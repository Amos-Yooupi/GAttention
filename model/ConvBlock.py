import torch.nn as nn
import torch


# 深度可分离卷积
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        # 深度分离卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)

    def forward(self, x, mask=1):
        x = self.depth_conv(x) * mask
        x = self.point_conv(x) * mask
        return x


# 轻量级卷积块
class PConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, partial_ratio):
        super().__init__()
        self.partial_len = int(in_channel * partial_ratio)
        self.residual_len = in_channel - self.partial_len
        # 对部分通道进行卷积，剩下用1x1卷积融合通道
        self.partial_conv = DepthWiseConv(self.partial_len, self.partial_len)
        self.connect_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x, mask=None):

        # mask -> [1, 1, H, W]
        # 部分卷积
        partial_x, residual_x = torch.split(x, [self.partial_len, self.residual_len], dim=1)
        partial_x = self.partial_conv(partial_x.clone(), mask)
        x = torch.cat([partial_x, residual_x], dim=1)
        # 剩下通道用1x1卷积融合
        return self.connect_conv(x) * mask


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, partial_ratio=0.5):
        super().__init__()
        # 卷积，batch_norm,激活函数
        self.conv1 = PConvBlock(in_channel, out_channel, partial_ratio)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, mask=1):
        # mask.shape -> [1, 1, H, W]  有元素的地方为0，没有元素的地方为1
        x = self.conv1(x, mask)
        x = self.batch_norm(x)
        return self.relu(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(UpConvBlock, self).__init__()
        self.conv_blks = nn.ModuleList()
        self.conv_blks.append(nn.ConvTranspose2d(in_dim, hidden_dim, kernel_size=2, stride=2, padding=0))
        self.conv_blks.append(nn.Tanh())
        in_dim = hidden_dim
        self.conv_blks.append(ConvBlock(in_dim, hidden_dim))
        self.conv_blks.append(nn.ConvTranspose2d(in_dim, hidden_dim, kernel_size=2, stride=2, padding=0))
        self.conv_blks.append(nn.Tanh())
        self.conv_blks.append(ConvBlock(in_dim, hidden_dim))

    def forward(self, x):
        for blk in self.conv_blks:
            x = blk(x)
        return x


class GAttentionConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, partial_ratio=0.5):
        super().__init__()
        # 卷积， pool, 卷积， pool
        self.conv1 = DepthWiseConv(in_channel, out_channel)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(in_channel, out_channel, partial_ratio)
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x, mask=None):
        # [B, C, H, W] -> [B, C', H/4. W/4]
        if mask is None:
            mask = 1
        else:
            mask = 1 - mask
        x = self.conv1(x, mask)
        x = self.pool(x)
        # 这里跟一个残差连接  后面大小变了没有mask了
        x += self.conv2(x)
        x = self.pool(x)
        return x


class PConvModule(nn.Module):
    def __init__(self, channels_set, partial_ratio):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.skip_connect = nn.ModuleList()
        self.skip_step = 2
        for i in range(len(channels_set) - 1):
            self.conv_blocks.append(ConvBlock(channels_set[i], channels_set[i+1], partial_ratio))
            if (i+1) % self.skip_step == 0:
                self.skip_connect.append(nn.Conv2d(channels_set[i], channels_set[i+1], kernel_size=1))

    def forward(self, x):
        for i, conv_block in enumerate(self.conv_blocks):
            if (i+1) % self.skip_step == 0:
                x = conv_block(x) + self.skip_connect[int((i+1)/self.skip_step - 1)](x)
                print("skip connect")
            else:
                x = conv_block(x)
        return x


if __name__ == '__main__':
    # 测试
    x = torch.randn(1, 64, 128, 128)
    pconv_module = PConvModule([64, 128, 256, 512, 64], 0.5)
    gatt_conv_block = GAttentionConvBlock(64, 64, 0.5)
    y = pconv_module(x)
    print(y.shape)
    y = gatt_conv_block(x)
    print(y.shape)