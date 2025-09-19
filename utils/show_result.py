import os.path
import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def show_result(true, predict, figsize=(8, 20), dpi=300):
    """
    可视化真实值与预测值对比（仅最后一行显示X轴标签，优化布局）
    """
    # 独立设置字体参数（可根据期刊要求微调）
    font_params = {
        'title': {'fontsize': 8, 'fontweight': 'bold', 'fontfamily': 'serif'},  # 子图标题
        'label': {'fontsize': 8, 'fontfamily': 'serif'},  # 坐标轴标签
        'legend': {'fontsize': 6, 'fontfamily': 'serif'},  # 图例
        'tick': {'fontsize': 8, 'fontfamily': 'serif'}  # 刻度值
    }

    # 仅保留兼容的全局配置
    plt.rcParams.update({
        'xtick.labelsize': font_params['tick']['fontsize'],
        'ytick.labelsize': font_params['tick']['fontsize'],
        'legend.fontsize': font_params['legend']['fontsize']
    })

    L, N, D = true.shape
    fig, axes = plt.subplots(N, D, figsize=figsize, dpi=dpi, sharex=True)

    # 确保axes为二维数组
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)


    for row in range(N):
        for col in range(D):
            ax = axes[row, col]
            # 绘制曲线
            ax.plot(true[:, row, col], label='True', linewidth=1.0, color='#1f77b4')
            ax.plot(predict[:, row, col], label='Predict', linewidth=1.0, color='#ff7f0e', linestyle='--')

            # 子图标题
            ax.set_title(f'Node {row + 1}, Feature {col + 1}',
                         fontsize=font_params['title']['fontsize'],
                         fontweight=font_params['title']['fontweight'],
                         fontfamily=font_params['title']['fontfamily'])

            # 仅最后一行添加X轴标签，其余行隐藏
            if row == N - 1:
                ax.set_xlabel('Time Step',
                              fontsize=font_params['label']['fontsize'],
                              fontfamily=font_params['label']['fontfamily'])

            else:
                ax.set_xlabel('')  # 清空非最后一行的X轴标签

            if col == 0:
                # 所有列添加Y轴标签（第一列可保留，其余可根据需要隐藏）
                ax.set_ylabel('predict value',
                              fontsize=font_params['label']['fontsize'],
                              fontfamily=font_params['label']['fontfamily'])
            else:
                ax.set_ylabel('')  # 清空非最后一行的y轴标签

            ax.legend(frameon=True, edgecolor='gray', facecolor='white', framealpha=0.6,
                      prop={'family': font_params['legend']['fontfamily']})

            # 刻度字体族设置
            ax.xaxis.set_tick_params(labelsize=font_params['tick']['fontsize'])
            ax.yaxis.set_tick_params(labelsize=font_params['tick']['fontsize'])
            for label in ax.get_xticklabels():
                label.set_fontfamily(font_params['tick']['fontfamily'])
            for label in ax.get_yticklabels():
                label.set_fontfamily(font_params['tick']['fontfamily'])

            # 网格线
            ax.grid(linestyle='--', alpha=0.7, linewidth=0.8)

            # Y轴科学计数法
            ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')

    plt.subplots_adjust(wspace=0.25, hspace=0.62)  # 调整间距避免标签重叠
    plt.savefig("predict.png", dpi=500)
    plt.show()

    return fig, axes


def show_weather_result(true, predict):
    L, D, H, W = true.shape
    for i in range(L):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        for ax in axes:
            ax.axis('off')

        # 显示真实数据
        im0 = axes[0].imshow(true[i, 0])
        # 显示预测数据
        im1 = axes[1].imshow(predict[i, 0])
        # 显示真实数据和预测数据的误差
        im2 = axes[2].imshow(torch.abs(true - predict)[i, 0])

        # 为每个子图添加色条，并调整字体大小
        for i, im in enumerate([im2]):
            cbar = fig.colorbar(im, ax=axes[i])  # 添加色条
            cbar.ax.tick_params(labelsize=30)  # 设置色条的字体大小

        plt.show()


def show_loss(train_loss, val_loss):
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()


def read_loss_file(path):
    with open(path, 'r') as f:
        loss = [float(item) for item in f.readlines()[:-1]]
    return loss[1:]


def show_loss_compare(no_regional_path, regional_path):
    data_regional = [read_loss_file(os.path.join(regional_path, "train_loss.txt")),
                     read_loss_file(os.path.join(regional_path, "val_loss.txt"))]
    data_no_regional = [read_loss_file(os.path.join(no_regional_path, "train_loss.txt")),
                         read_loss_file(os.path.join(no_regional_path, "val_loss.txt"))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    else:
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

    title_font_size = 16
    label_font_size = 14
    axes[0, 0].plot(data_no_regional[0], label="-Train")
    axes[0, 0].plot(data_no_regional[1], label="-Validation")
    axes[0, 1].plot(data_regional[0], label="-Train")
    axes[0, 1].plot(data_regional[1], label="-Validation")
    axes[0, 0].set_title('No Regional Graph Representation', fontsize=title_font_size)
    axes[0, 1].set_title('Regional Graph Representation', fontsize=title_font_size)
    axes[0, 0].set_xlabel('Epoch', fontsize=label_font_size)
    axes[0, 0].set_ylabel('Loss', fontsize=label_font_size)
    axes[0, 1].set_xlabel('Epoch', fontsize=label_font_size)
    axes[0, 1].set_ylabel('Loss', fontsize=label_font_size)
    axes[0, 0].grid()
    axes[0, 1].grid()
    axes[0, 0].legend()
    axes[0, 1].legend()
    plt.savefig("loss_compare.eps")
    plt.show()
