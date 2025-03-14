import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        """
        in_channels: int, input channels --- 特征维度
        out_channels: int, output channels
        kernel_size: int, kernel size
        stride: int, stride
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        else:
            pass
        return x


class DistillBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DistillBlock, self).__init__()

        self.conv_blks = nn.Sequential(
            ConvBlock(in_channels, in_channels*4, 3, 1, 1, activation=nn.ReLU(inplace=True)),
            ConvBlock(in_channels*4, out_channels, 3, 1, 1, activation=None))

        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

        self.max_pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T/2, D)
        """
        x = x.transpose(1, 2)
        skip = self.skip_conv(x)
        conv_x = self.conv_blks(x)
        # 最大池化 - 降低时间维度
        return self.max_pool(skip + conv_x).transpose(1, 2)


if __name__ == '__main__':
    import torch
    x = torch.randn(1, 78, 128)
    model = DistillBlock(128, 128)
    print(model(x).shape)