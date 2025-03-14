import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec


import matplotlib.pyplot as plt
import torch


def show_attention_maps(attention_maps, sep=2, name='attention', camp="Greens"):
    """
    显示 b 个 h×w 的注意力图。

    参数:
    attention_maps (torch.Tensor): 形状为 (b, h, w) 的张量，其中 b 是批大小，h 和 w 分别是注意力图的高度和宽度。
    """
    assert attention_maps.ndim == 3, "Input tensor must have three dimensions (b, h, w)"
    b, h, w = attention_maps.shape

    # 将 attention_maps 转为 NumPy 数组，便于使用 Matplotlib 可视化
    attention_maps = attention_maps.detach().cpu().numpy()

    # 创建图形和子图
    rows = b // sep if b >= sep else b
    fig, axs = plt.subplots(rows, sep, figsize=(8, 10))

    # 当 batch 只有一张图时，确保 axs 可迭代
    if sep == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # 显示所有子图的注意力图
    for i in range(b):
        im = axs[i].imshow(attention_maps[i], cmap=camp, aspect='equal')
        axs[i].axis('off')  # 隐藏坐标轴
        axs[i].set_title(f"Node {i+1}", fontsize=24)

    # 添加全局颜色条，放在左侧
    cax = fig.add_axes([0.02, 0.1, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)  # 将颜色条添加到指定的轴
    cbar.set_label('Attention Score', fontsize=24)

    # 调整布局
    plt.subplots_adjust(wspace=0.5)  # 调整子图间距
    plt.tight_layout()

    # 保存图像
    plt.savefig(name + '.eps')
    plt.show()


if __name__ == '__main__':

    # 示例用法
    b, h, w = 8, 6, 6  # 4 个 10x10 的注意力图
    attention_maps = torch.rand(b, h, w)  # 随机生成一些注意力图数据
    show_attention_maps(attention_maps, camp='Reds', name='attention')
