from RGNN import RGNN, RGNNBlock
from LSTM import LSTM
import torch.nn as nn
from GraphAttentionNetwork import FusionGraphAttentionLayer, GraphNeuralNetwork
from ConvBlock import GAttentionConvBlock
from STGNN import STGCN
from Graph_wave import GraphWaveNet
from GAT import GAT


class BackBone(nn.Module):
    def __init__(self, config, backbone_type):
        super(BackBone, self).__init__()
        self.node_encoder = nn.ModuleList()
        self.name = backbone_type
        self.backbone = nn.ModuleList()
        self.config = config
        if backbone_type == 'RGNN':
            # 选择不同动态系统的编码方式
            if config.representation == 'graph':
                # 是否使用区域化图表示
                if config.is_fusion:
                    self.node_encoder = nn.ModuleList([FusionGraphAttentionLayer(config.embed_dim,
                                                                                 config.hidden_dim,
                                                                                 config.num_head_for_node,
                                                                                 config.att_mode,
                                                                                 config.activation)
                                                       for _ in range(config.num_blocks)])
                else:
                    self.node_encoder = nn.ModuleList([GraphNeuralNetwork(config.embed_dim,
                                                                         config.activation)
                                                       for _ in range(config.num_blocks)])
            elif config.representation == 'grid':
                self.node_encoder = nn.ModuleList([GAttentionConvBlock(config.hidden_dim, config.hidden_dim, config.partial_ratio)
                                                   for _ in range(config.num_blocks)])
            else:
                raise ValueError('Invalid representation: {}'.format(config.representation))
            [self.backbone.append(RGNNBlock(config.embed_dim, config.hidden_dim,
                                                  config.in_len, config.num_head_for_time,
                                                  config.num_head_for_node, config.num_moe_layer,
                                                  config.num_expert, config.top_k, config.att_mode,
                                                  config.activation, config.is_fusion)) for i in
             range(config.num_blocks)]
        elif backbone_type == 'LSTM':
            self.backbone.add_module("LSTM", LSTM(config.embed_dim, config.hidden_dim))
        elif backbone_type == "STGNN":
            self.backbone.add_module("STGCN", STGCN(config.embed_dim, config.embed_dim))
        elif backbone_type == "GraphWave":
            self.backbone.append(GraphWaveNet(config.embed_dim, config.hidden_dim))
        elif backbone_type == "GAT":
            self.backbone.append(GAT(config.embed_dim, config.hidden_dim))
        else:
            raise ValueError('Invalid backbone type: {}'.format(backbone_type))

    def forward(self, *args, **kwargs):
        if self.name == 'RGNN':
            if self.config.representation == 'graph':
                x, adj = args[0], args[1]
                # 时间和B在一个维度
                B, N, L, D = x.shape
                adj = adj.repeat(L, 1, 1)
                for i in range(self.config.num_blocks):
                    # 先进行时序编码
                    # [B, N, L, D] -> [B*N, L, D]
                    x = x.contiguous().view(B * N, L, D)
                    x = self.backbone[i](x)
                    # 进行位置交换 [B, L, N, D] -> [B*N, L, D]
                    x = x.contiguous().view(B, N, L, -1).transpose(1, 2).contiguous().view(B * L, N, -1)
                    x, adj = self.node_encoder[i](x, adj)
                    # 还原[B, N, L, D]
                    x = x.contiguous().view(B, L, N, -1).transpose(1, 2)
            elif self.config.representation == 'grid':
                x, x_mask = args[0], args[1]  # [B, L, feature_dim, H, W]
                for i in range(self.config.num_blocks):
                    B, L, D, H, W = x.shape
                    # 先进行卷积，所以将时序L和B混合
                    x = x.contiguous().view(B*L, D, H, W)
                    x = self.node_encoder[i](x)  # [B*L, D, H, W] --> [B*L, D', H/4, W/4]
                    # 进行时序上面的提取，将H, W部分合并, 注意一下，这个维度D可能会变，因为是由卷积的通道数决定的, 但是代码不变
                    x = x.contiguous().view(B, L, D, H//4, W//4).permute(0, 3, 4, 1, 2).contiguous().view(-1, L, D)
                    x = self.backbone[i](x)  # [B*H*W/16, L, D'] --> [B*H*W/16, L, D]
                    # 回复成初始的数据格式
                    x = x.contiguous().view(B, H//4, W//4, L, D).permute(0, 3, 4, 1, 2)
        else:
            for module in self.backbone:
                args = module(*args, **kwargs)
            x = args[0]
        return x
