import torch.nn as nn
from Distill_Block import ConvBlock
from SharedSpareMoe import BasicExpert


class OutHeadTBC(nn.Module):
    def __init__(self, in_dim, in_channels, out_dim, out_channels, num_layers):
        super(OutHeadTBC, self).__init__()
        self.num_train = 3
        self.out_dim = out_dim
        self.out_channels = out_channels
        self.time_conv = nn.Conv1d(in_dim, out_dim*self.num_train, kernel_size=3, stride=1, padding=1)
        self.feature_linear = BasicExpert(in_channels, in_channels*2, out_channels, num_layers)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, L, d)
        """
        B, _, _ = x.size()
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        x = self.time_conv(x)  # (B, D, T) -> (B, N*d, T)
        x = self.feature_linear(x)  # (B, N*d, T) -> (B, N*d, L)
        return x.contiguous().view(B, self.num_train, self.out_dim, -1).transpose(2, 3)  # (B, N*d, L) -> (B, N, d, L) -> (B, N, L, d)


class OutHeadTraffic(nn.Module):
    def __init__(self, in_dim, in_channels, out_dim, out_channels, num_layers, num_node):
        super(OutHeadTraffic, self).__init__()
        self.num_node = num_node
        self.out_dim = out_dim
        self.out_channels = out_channels
        self.time_conv = nn.Conv1d(in_dim, out_dim*self.num_node, kernel_size=3, stride=1, padding=1)
        self.feature_linear = BasicExpert(in_channels, in_channels*2, out_channels, num_layers)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, L, d)
        """
        B, _, _ = x.size()
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        x = self.time_conv(x)  # (B, D, T) -> (B, N*d, T)
        x = self.feature_linear(x)  # (B, N*d, T) -> (B, N*d, L)
        return x.contiguous().view(B, self.num_node, self.out_dim, -1).transpose(2, 3)  # (B, N*d, L) -> (B, N, d, L) -> (B, N, L, d)

