import torch
import torch.nn as nn
import torch.nn.functional as F


# This is implementation for graph wavenet

class GraphWaveBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dilation):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 3), padding=(0, dilation), dilation=(1, dilation))
        self.conv2 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 3), padding=(0, dilation), dilation=(1, dilation))
        self.skip_conv = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)

    def forward(self, x, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # 时序提取
        x = x.permute(0, 3, 1, 2)  # (batch_size, num_features, num_nodes, num_timesteps)
        gated = F.tanh(self.conv1(x))
        conv_x = F.sigmoid(self.conv2(x))
        conv_x = gated * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1)  # (batch_size, num_nodes, num_timesteps, num_features)
        # 空间提取
        lfs = torch.einsum("ij,jklm->kilm", [A, conv_x.permute(1, 0, 2, 3)])
        return lfs + self.skip_conv(x).permute(0, 2, 3, 1)


class GraphWaveNet(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        num_layers = 2
        dilation = [2**(i+1) for i in range(num_layers)]
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(GraphWaveBlock(in_dim, hidden_dim, dilation[i]))
            in_dim = hidden_dim

    def forward(self, x, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        A = A[0]
        result = []
        for block in self.blocks:
            x = block(x, A)
            result.append(x)
        return sum(result),

    def __str__(self):
        return "GraphWaveNet"

