import torch.nn as nn
from timer import timer
import torch
from GraphAttentionNetwork import GraphAttentionLayer, FusionGraphAttentionLayer
from Distill_Block import DistillBlock
from SharedSpareMoe import SharedSpareTimeMOE, BasicExpert
from show_attention import show_attention_maps


class AddAndNorm(nn.Module):
    def __init__(self, last_dim):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(last_dim)

    def forward(self, x, y):
        z = x + y
        z = self.norm(z)
        return z


class GAttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, input_length, num_heads_for_time,
                 num_heads_for_node, num_moe_layers, num_expert,
                 top_k, att_mode, activation, is_fusion):
        super(GAttentionBlock, self).__init__()
        """
        Args:
            in_dim: input dimension -- number of features
            out_dim: output dimension -- number of features
            input_length: input sequence length
            num_moe_layers: number of moe layers -- the linear layers in the moe block
            num_expert: number of experts -- the number of experts in the moe block
            top_k: top k experts -- the number of experts to be selected in the moe block
            num_heads_for_time: number of heads for time attention multi-head attention
            num_heads_for_node: number of heads for node attention gat
            att_mode: attention mode -- 'linear' or 'dot' mode for gat(attention mode)
            activation: activation function -- 'relu' or 'leaky_relu' or else
        """
        # 多头注意力层
        self.time_multi_head_att = nn.MultiheadAttention(in_dim, num_heads_for_time, batch_first=True)
        # 残差 + 层归一化
        self.add_norm_1 = AddAndNorm(in_dim)
        # Time-aware spare moe
        self.time_spare_moe = SharedSpareTimeMOE(in_dim, in_dim, num_expert,
                                                 num_moe_layers, input_length,
                                                 input_length, top_k)
        # 残差 + 层归一化
        self.add_norm_2 = AddAndNorm(in_dim)
        # DistillBlock - 蒸馏模块
        self.distill_block = DistillBlock(in_dim, in_dim)
        # GAT 层 --- 是否融合
        if is_fusion:
            self.gat_layers = FusionGraphAttentionLayer(in_dim, out_dim, num_heads_for_node, att_mode, activation=activation)
        else:
            self.gat_layers = GraphAttentionLayer(in_dim, out_dim, num_heads_for_node, att_mode, activation=activation)

        # 一些记录参数的变量
        self.time_att_weight = None
        self.node_att_weight = None
        self.out_dim = out_dim

    def forward(self, x, adj):
        """
        Args:
            x: input sequence -- (batch_size, num_nodes, input_length, in_dim)
            adj: adjacency matrix -- (batch_size, num_nodes, num_nodes)
        Returns:
            output sequence -- (batch_size, num_nodes, output_length, out_dim)
        """
        B, N, L, D = x.shape
        # 多头注意力层 - 将node维度和batch维度拼接
        node_concat_x = x.contiguous().view(B * N, L, D)
        multi_head_att_x, self.time_att_weight = self.time_multi_head_att(node_concat_x, node_concat_x, node_concat_x)
        self.time_att_weight = self.time_att_weight.detach().cpu().view(B, N, L, L)
        # 残差 + 层归一化
        multi_head_att_x = self.add_norm_1(node_concat_x, multi_head_att_x)
        # Time-aware spare moe --- 时序专家模块
        time_aware_moe_x = self.time_spare_moe(multi_head_att_x)
        # 残差 + 层归一化
        time_aware_moe_x = self.add_norm_2(multi_head_att_x, time_aware_moe_x)
        # DistillBlock - 蒸馏模块
        distill_x = self.distill_block(time_aware_moe_x).contiguous().view(B, N, L // 2, D)
        # GAT 层 -- 拼接time维度
        time_concat_x = distill_x.permute(0, 2, 1, 3).contiguous().view(-1, N, D)
        node_att_x, self.node_att_weight = self.gat_layers(time_concat_x, adj.repeat(L//2, 1, 1))
        # 复原所有维度
        node_att_x = node_att_x.contiguous().view(B, L // 2, N, self.out_dim).permute(0, 2, 1, 3)
        return node_att_x


class GAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, input_length, num_heads_for_time,
                 num_heads_for_node, num_moe_layers, num_expert,
                 top_k, att_mode, activation, is_fusion, num_blocks):
        super(GAttention, self).__init__()
        """
        Args:
            num_blocks: number of blocks in the model -- the number of GAttentionBlocks in the model
        """
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # assert top_k > i, "top_k must be greater than or equal to num_blocks"
            self.blocks.append(GAttentionBlock(in_dim, hidden_dim, input_length // 2**i, num_heads_for_time,
                                               num_heads_for_node, num_moe_layers, num_expert,
                                               max(top_k, 1), att_mode, activation, is_fusion))
            in_dim = hidden_dim

    @timer(False, "GAttention")
    def forward(self, x, adj):
        """
        Args:
            x: input sequence -- (batch_size, num_nodes, input_length, in_dim)
            adj: adjacency matrix -- (batch_size, num_nodes, num_nodes)
        Returns:
            output sequence -- (batch_size, num_nodes, output_length, out_dim)
        """
        blks_out = []
        for block in self.blocks:
            x = block(x, adj)
            blks_out.append(x)

        # 拼接所有block的输出 --- 汇聚全局变量-所有节点
        x = torch.sum(torch.cat(blks_out, dim=-2), keepdim=False, dim=1)
        return x


if __name__ == '__main__':
    # # 测试代码
    # gat = GAttentionBlock(in_dim=128, out_dim=128, input_length=64, num_heads_for_time=8,
    #                       num_heads_for_node=8, num_moe_layers=2, num_expert=4, top_k=2,
    #                       att_mode='linear', activation='relu', is_fusion=True)
    # x = torch.randn(32, 64, 64, 128)
    # adj = torch.randn(32, 64, 64)
    # out = gat(x, adj)
    # print(out.shape)

    # 测试代码
    gat = GAttention(in_dim=128, hidden_dim=256, input_length=64, num_heads_for_time=8,
                     num_heads_for_node=8, num_moe_layers=2, num_expert=4, top_k=2,
                     att_mode='linear', activation='relu', is_fusion=True, num_blocks=2)
    num_node = 7
    x = torch.randn(32, num_node, 64, 128)
    adj = torch.randn(32, num_node, num_node)
    out = gat(x, adj)
    print(out.shape)
