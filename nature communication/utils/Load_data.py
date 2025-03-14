import os.path
import numpy as np
import torch


def read_file_TBC(dir_file_path, time_freq=10):
    train_path = os.path.join(dir_file_path, 'Train.opt')
    pier_path = os.path.join(dir_file_path, 'Earthquake.opt')
    bridge_path = os.path.join(dir_file_path, 'Span.opt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} \n Train.opt file not found in the given directory")
    if not os.path.exists(pier_path):
        raise FileNotFoundError(f"{pier_path} \nEarthquake.opt file not found in the given directory")
    if not os.path.exists(bridge_path):
        raise FileNotFoundError(f"{bridge_path} \nSpan.opt file not found in the given directory")

    train_data = torch.load(train_path).transpose(1, 2)
    bridge_data = torch.load(bridge_path)
    _, d, L = bridge_data.size()
    pier_data = torch.load(pier_path).view(-1, 3, L).transpose(1, 2)
    bridge_data = bridge_data.view(-1, 3, d, L).mean(dim=1, keepdim=False).transpose(1, 2)

    return train_data[:, ::time_freq], bridge_data[:, ::time_freq], pier_data[:, ::time_freq]


def read_file_traffic(traffic_data_path, time_freq=1):
    traffic_data = np.load(traffic_data_path)['data']
    # 只需要流量数据
    return torch.tensor(traffic_data[:, ::time_freq]).transpose(0, 1)[:, :, 0:1]


if __name__ == '__main__':
    train_data, pier_data, bridge_data = read_file_TBC(r'E:\DeskTop\深度学习\nature communication\data\TBC\Span-3')
    print(train_data.size(), pier_data.size(), bridge_data.size())
    traffic_data_path = r'E:\DeskTop\深度学习\nature communication\data\traffic\PEMS04\pems04.npz'
    print(read_file_traffic(traffic_data_path).shape)
