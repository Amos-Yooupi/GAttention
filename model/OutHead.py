import torch.nn as nn
# from Distill_Block import ConvBlock
from SharedSpareMoe import BasicExpert
from ConvBlock import UpConvBlock


class OutHeadTBC(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(OutHeadTBC, self).__init__()
        self.out_dim = out_dim
        self.feature_out = BasicExpert(in_dim, in_dim, out_dim, num_layers)

    def forward(self, x):
        """
        Args:
            x: (B, N, T, D)
        Returns:
            out: (B, N, T, d)
        """
        # 取出对应的节点
        return self.feature_out(x)


class OutHeadTraffic(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, region_infor, node_idx, is_region=True):
        super(OutHeadTraffic, self).__init__()
        self.out_dim = out_dim
        self.region_infor = region_infor
        self.num_node = region_infor.sum()
        self.node_idx = node_idx
        # 预测第一个区域的节点
        if is_region:
            self.linear = BasicExpert(in_dim, in_dim, out_dim * self.num_node, num_layers)
        else:
            self.linear = BasicExpert(in_dim, in_dim, out_dim, num_layers)
        self.is_region = is_region

    def forward(self, x):
        """
        Args:
            x: (B, N, T, D)
        Returns:
            out: (B, N, L, d)
        """
        # 取出第一个区域的节点
        B, _, T, D = x.shape
        if self.is_region:
            # 这里一定要和数据集对应
            region_node_x = x[:, self.node_idx]  # (B, T, D)
            # 预测第一个区域的节点
            region_node_out = self.linear(region_node_x)  # (B, T, num_node * out_dim)
            graph_out = region_node_out.contiguous().view(B, T, self.num_node, self.out_dim).transpose(1, 2)
        else:
            result = self.linear(x)  # (B, N, T, out_dim)
            graph_out = result[:, self.region_infor]

        return graph_out


class OutHeadWeather(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_blks):
        super(OutHeadWeather, self).__init__()
        self.out_dim = out_dim
        self.conv_blks = nn.ModuleList()
        for i in range(num_blks):
            self.conv_blks.append(UpConvBlock(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.linear_dim = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Tanh(), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        """
        Args:
            x: (B, L, D, H, W)
        Returns:
            out: (B, L, d)
        """
        # 将时序L叠在一起
        B, L, D, H, W = x.shape
        x = x.contiguous().view(-1, D, H, W)
        # 卷积恢复原来的尺寸
        for blk in self.conv_blks:
            x = blk(x)
        # 将D维度放在最后
        _, _, H, W = x.shape
        x = x.contiguous().view(B, L, -1, H, W).transpose(2, 4)  # (B, L, H, W, d)
        output = self.linear_dim(x).transpose(2, 4)  # (B, L, H, W, d) -> (B, L, d, H, W)
        return output

