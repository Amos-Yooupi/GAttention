import os.path
import numpy as np
import torch


def read_file_TBC(dir_file_path, time_freq=10):
    train_path = os.path.join(dir_file_path, 'Train.pt')
    pier_path = os.path.join(dir_file_path, 'Earthquake.pt')
    bridge_path = os.path.join(dir_file_path, 'Span.pt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} \n Train.opt file not found in the given directory")
    if not os.path.exists(pier_path):
        raise FileNotFoundError(f"{pier_path} \nEarthquake.opt file not found in the given directory")
    if not os.path.exists(bridge_path):
        raise FileNotFoundError(f"{bridge_path} \nSpan.opt file not found in the given directory")
    # 这里两种数据格式，重写！！！！
    if int(dir_file_path.split("Car_")[-1]) > 3:
        train_data = torch.load(train_path)
    else:
        train_data = torch.load(train_path).transpose(1, 2)
    bridge_data = torch.load(bridge_path)
    pier_data = torch.load(pier_path)
    L = pier_data.shape[-1]
    pier_data = pier_data.reshape(-1, 3, L).transpose(1, 2)
    try:
        bridge_data = bridge_data.reshape(-1, 3, 6, L).mean(dim=1, keepdim=False).transpose(1, 2)
    except:
        bridge_data = bridge_data
        time_freq = 1

    return train_data[:, ::time_freq], bridge_data[:, ::time_freq], pier_data[:, ::time_freq]


def read_file_traffic(traffic_data_path, time_freq=1):
    traffic_data = np.load(traffic_data_path)['data']
    # 只需要流量数据
    return torch.tensor(traffic_data[:, ::time_freq]).transpose(0, 1)[:, :, 0:1]


if __name__ == '__main__':
    train_data, pier_data, bridge_data = read_file_TBC(r'/nature communication/data/TBC/Span_3_Car_3')
    print(train_data.size(), pier_data.size(), bridge_data.size())
    traffic_data_path = r'E:\DeskTop\深度学习\nature communication\data\traffic\PEMS04\pems04.npz'
    print(read_file_traffic(traffic_data_path).shape)
