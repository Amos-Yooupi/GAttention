import os.path

import matplotlib.pyplot as plt
import numpy as np


def show_result(true, predict):
    L, N, D = true.shape

    fig, axes = plt.subplots(N, D, figsize=(10, 10))
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    else:
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

    for row in range(N):
        for col in range(D):
            axes[row, col].plot(true[:, row, col], label='True')
            axes[row, col].plot(predict[:, row, col], label='Predict')
            axes[row, col].set_title(f'Node {row+1}, Feature {col+1}')
            axes[row, col].grid()
            axes[row, col].legend()
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
