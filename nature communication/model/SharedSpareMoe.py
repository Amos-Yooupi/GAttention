import torch
import torch.nn as nn
import torch.nn.functional as F
from timer import timer


class BasicExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        """基础专家模型，包含多个全连接层"""
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for i in range(num_layer - 1):
            self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TimeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim, out_dim):
        """用于处理时间维度的模块 - 卷积层 + 门控线性单元"""
        """
        in_channel: 输入通道数 -- 特征的维度
        out_channel: 输出通道数 -- 输出的特征的维度
        in_dim: 输入的时序的维度 -- 时序长度
        out_dim: 输出时序的维度 -- 时序的长度，每个不同的周期的时序数据需要还原
        """
        super().__init__()
        self.time_conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.time_gate = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
        self.restore_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_length, dim]
        """
        x_transpose = x.transpose(1, 2)
        time_conv = self.time_conv(x_transpose)
        skip_conv = self.skip_conv(x_transpose)
        time_gate = F.sigmoid(self.time_gate(x_transpose))
        # 还原时序维度 [B, out_dim, out_channel]
        return self.restore_linear(F.gelu(time_conv * time_gate + skip_conv)).transpose(1, 2)


class BasicTimeExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, in_channel, out_channel):
        """基础专家模型，包含多个全连接层 + 时间维度处理模块"""
        """
        in_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        out_dim: 输出特征维度
        num_layer: 全连接层数
        out_channel: 输出通道数 -- 时序的长度，每个不同的周期的时序数据需要还原
        """
        super().__init__()
        self.linear_expert = BasicExpert(in_dim, hidden_dim, hidden_dim, num_layer)
        self.time_block = TimeBlock(hidden_dim, out_dim, in_channel, out_channel)

    def forward(self, x):
        """
             x: [batch_size, seq_length, dim]
        """
        # 先经过线性层
        x = self.linear_expert(x)
        # 再经过时间维度处理模块
        x = self.time_block(x)
        return x


class Router(nn.Module):
    def __init__(self, in_channel, in_dim, num_exper, top_k):
        super().__init__()
        self.top_k = top_k
        self.time_conv = nn.Conv1d(in_dim, 1, kernel_size=3, padding=1)
        self.feature_linear = nn.Sequential(nn.Linear(in_channel, in_channel*4), nn.GELU(), nn.Linear(in_channel*4, num_exper))

    def forward(self, x):
        """
        x: [batch_size, in_channel, in_dim]
        in_channel: 输入的时间序列维度
        in_dim: 输入的特征维度
        return: router_logit: [batch_size, top_k]  # 选中专家的门控得分
                top_k_idx: [batch_size, top_k]  # 选中专家的索引
        """
        # 先进行时序维度的卷积
        x = self.time_conv(x.transpose(1, 2)).squeeze(dim=1)  # [batch_size, in_channel]
        # 再进行特征维度的线性变换
        x = self.feature_linear(x)  # [batch_size, num_exper]
        # 得到每个batch选择的专家的logit和索引
        router_prob = F.softmax(x, dim=1)
        # 得到每个batch选择的专家的logit和索引
        router_logit, top_k_idx = torch.topk(router_prob, self.top_k, dim=1)
        # 对router_logit进行归一化
        router_logit /= router_logit.sum(dim=1, keepdim=True)

        return router_logit, top_k_idx


class SpareMOE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_expert, num_layer, in_channel, out_channel, top_k=None):
        """稀疏门控专家模型"""
        super().__init__()
        # 考虑不同周期的专家模型 -- 比如T=16 包含 16 8 4 2 1 这五个专家模型，但最开始那个时序长度为16是共享的
        assert in_channel >= num_expert, print("in_channel should be larger than num_expert")

        self.num_expert = num_expert
        self.experts = nn.ModuleList()
        for i in range(num_expert):
            # 每个专家用于处理不同的周期 --- 选择合适的专家模型 in_channel在变化
            expert = BasicTimeExpert(in_dim, hidden_dim, out_dim, num_layer, in_channel // (i+2), out_channel)
            self.experts.append(expert)

        # 默认 top_k为None则选取num_expert//2个专家
        self.router = Router(in_channel, in_dim, num_expert, num_expert // 2 if top_k is None else top_k)

    def downsample(self, x, factor):
        # 用于下采样的函数
        return F.avg_pool1d(x.transpose(1, 2), factor, stride=factor).transpose(1, 2)

    @timer(False, "SpareMOE")
    def forward(self, x):
        """
        x: [batch_size, seq_length, dim]
        """
        final_state = torch.zeros_like(x).to(x.device, x.dtype)  # 最终状态
        router_logit, top_k_idx = self.router(x)  # 得到每个batch选择的专家的logit和索引

        # 对专家进行循环，得到每个batch的最终状态
        for i in range(self.num_expert):
            # 得到当前周期的专家模型
            expert = self.experts[i]
            # 得到当前周期的输入
            current_x = self.downsample(x, i+2)  # [batch_size, seq_length//2**(i+1), dim]
            # 找出选择了当前专家的batch的索引 以及是top_k中的哪个 用于后续门控机制的计算
            select_batch_idx, select_top_idx = torch.where(top_k_idx == i)
            # 使用当前的专家模型进行计算
            current_state = expert(current_x[select_batch_idx]) * router_logit[select_batch_idx, select_top_idx].unsqueeze(dim=-1).unsqueeze(dim=-1)  # [num_select, seq_length, out_dim]
            # 添加到最后的状态中
            final_state.index_add_(dim=0, index=select_batch_idx, source=current_state)
        return final_state


class SharedSpareTimeMOE(nn.Module):
    def __init__(self, in_dim, out_dim, num_expert, num_layer, in_channel, out_channel, top_k=None, spare_mode=False):
        """共享稀疏门控专家模型"""
        super().__init__()
        # 考虑不同周期的专家模型 -- 比如T=16 包含 16 8 4 2 1 这五个专家模型，但最开始那个时序长度为16是共享的
        self.spare_moe = SpareMOE(in_dim, in_dim*4, out_dim, num_expert, num_layer, in_channel, out_channel, top_k)

        self.shared_expert = BasicTimeExpert(in_dim, in_dim*4, out_dim, num_layer, in_channel, out_channel)

        self.spare_mode = spare_mode

    @timer(False, "SharedSpareMOE")
    def forward(self, x):
        """
        x: [batch_size, seq_length, dim]
        """
        # 先经过共享的专家模型
        shared_state = self.shared_expert(x)  # [batch_size, out_channel, out_dim]

        if self.spare_mode:
            # 再经过稀疏的专家模型
            sparse_state = self.spare_moe(x)  # [batch_size, out_channel, out_dim]
        else:
            sparse_state = 0
        # 得到最终的状态
        final_state = shared_state + sparse_state
        return final_state


if __name__ == '__main__':
    x = torch.randn(64, 32, 128)
    shared_moe = SharedSpareTimeMOE(128, 128, 10, 3, 32, 32)
    output = shared_moe(x)
    print(output.shape)
