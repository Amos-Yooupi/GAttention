from torch.utils.data import DataLoader, Dataset
import torch
from Load_data import *
from Create_adj import *


class DataSetTBC(Dataset):
    def __init__(self, vehicle, bridge, pier, input_length, output_length, num_task_node, mode="complex"):
        self.vehicle = vehicle
        self.bridge = bridge
        self.pier = pier
        self.in_len = input_length
        self.out_len = output_length
        assert mode in ["simple", "complex"], "Mode should be simple or complex"
        self.mode = mode
        self.num_task_node = num_task_node

    def __len__(self):
        return int((self.vehicle.size()[1] - self.in_len - self.out_len) / (self.out_len
                                                                            if self.mode == "simple"
                                                                            else 1))

    def __getitem__(self, idx):
        start_idx = idx * (self.out_len if self.mode == "simple" else 1)
        end_idx = start_idx + self.in_len
        vehicle_dta = self.vehicle[:, start_idx:end_idx]
        bridge_dta = self.bridge[:, start_idx:end_idx]
        pier_dta = self.pier[:, start_idx:end_idx]
        label = self.vehicle[:, end_idx:end_idx + self.out_len]
        return [vehicle_dta, bridge_dta, pier_dta], label[:self.num_task_node]


class DataSetTraffic(Dataset):
    def __init__(self, traffic_data, region_graph_x, in_len, out_len, num_task_node, mode="complex"):
        self.traffic_data = traffic_data
        self.region_graph_x = region_graph_x
        self.in_len = in_len
        self.out_len = out_len
        assert mode in ["simple", "complex"], "Mode should be simple or complex"
        self.mode = mode
        self.num_task_node = num_task_node

    def __len__(self):
        return int((self.traffic_data.size()[1] - self.in_len - self.out_len) / (self.out_len
                                                                                 if self.mode == "simple"
                                                                                 else 1))

    def __getitem__(self, idx):
        start_idx = idx * (self.out_len if self.mode == "simple" else 1)
        end_idx = start_idx + self.in_len
        graph_x = self.region_graph_x[:, start_idx:end_idx]
        label = self.traffic_data[:, end_idx:end_idx + self.out_len]
        return [graph_x], label[:self.num_task_node]


class DataIter(object):
    def __init__(self):
        pass

    def norm_data(self, *datas):
        # 归一化数据
        mean = [data.mean(dim=1, keepdim=True) for data in datas]
        std = [data.std(dim=1, keepdim=True) for data in datas]
        return [(data - mean[i]) / std[i] for i, data in enumerate(datas)], [mean, std]

    def re_norm_data(self, data, norm_record):
        mean, std = norm_record
        return data * std + mean

    def norm_dict(self, dataset_dict):
        new_dataset_dict = {}
        for key, values in dataset_dict.items():
            norm_data, norm_record = self.norm_data(*values)
            new_dataset_dict[key] = {"data": norm_data, "norm_record": norm_record}
        return new_dataset_dict

    def split_data(self, *original_data, split_ratio: list):
        """
        Return a dictionary with keys 'train', 'val', 'test' and values are lists of data.
        The data is splitted according to the split_ratio.
        {
        'train': {'data':[vehicle, bridge, pier], 'norm_record':[mean_list, std_list]},
        'val': {'data':[vehicle, bridge, pier], 'norm_record':[mean_list, std_list]},
        'test': {'data':[vehicle, bridge, pier], 'norm_record':[mean_list, std_list]}
        }
        """
        train_ratio, val_ratio, test_ratio = split_ratio
        train_data, val_data, test_data = [], [], []
        for data in original_data:
            total_data_length = data.shape[1]
            train_length = int(total_data_length * train_ratio)
            val_length = int(total_data_length * val_ratio)
            train_data.append(data[:, :train_length])
            val_data.append(data[:, train_length:train_length + val_length])
            test_data.append(data[:, train_length + val_length:])
        return {"train": train_data, "val": val_data, "test": test_data}

    def choose_dataiter(self):
        return self

    def __len__(self):
        return 1


class DataIterTBC(DataIter):
    def __init__(self, data_file_path, split_ratio: list):
        super().__init__()
        assert sum(split_ratio) == 1, "Split ratio should sum up to 1"
        # 读取数据
        self.vehicle, self.bridge, self.pier = read_file_TBC(data_file_path, time_freq=20)
        # 划分数据集
        dataset_dict = self.split_data(self.vehicle,
                                       self.bridge, self.pier,
                                       split_ratio=split_ratio)
        # 归一化数据
        self.dataset_dict = self.norm_dict(dataset_dict)

        # 设置参数
        self.adj, self.num_train, self.num_bridge = None, None, None
        self.get_adj()
        # 只预测列车节点的响应
        self.num_task = self.vehicle.shape[0]

    def get_adj(self):
        # 取出列车的数量 和  桥梁的数量
        self.num_train = self.vehicle.shape[0]
        self.num_bridge = self.bridge.shape[0]

        self.adj = create_adj_tbc(self.num_train, self.num_bridge)

    def generate_dataset(self, dataset_keyword, batch_size, in_len, out_len, load_mode="complex", shuffle=True):
        dataset = DataSetTBC(*self.dataset_dict[dataset_keyword]["data"], in_len, out_len, self.num_task, load_mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return dataloader

    def re_norm_data(self, data, keyword):
        norm_record = self.dataset_dict[keyword]["norm_record"]
        re_normed_data = super().re_norm_data(data, norm_record)
        return re_normed_data


class DataIterTraffic(DataIter):
    def __init__(self, traffic_data_file_path, traffic_adj_file_path, split_ratio: list, is_region=False):
        super().__init__()
        self.traffic_data = read_file_traffic(traffic_data_file_path)
        print("Ori:", self.traffic_data.element_size() * self.traffic_data.numel() / 1024 ** 2, "MB")
        # 设置参数
        self.adj = self.get_adj(traffic_adj_file_path)
        self.is_region = is_region

        # 生成区域图
        if is_region:
            self.region_adj, self.region_graph_x = create_regional_graph(self.adj, self.traffic_data)
            print("Regional graph:", self.region_graph_x.element_size() * self.region_graph_x.numel() / 1024 ** 2, "MB")
            self.adj = self.region_adj
            dataset_dict_region = self.split_data(self.region_graph_x,
                                                  split_ratio=split_ratio)
            self.dataset_dict_region = self.norm_dict(dataset_dict_region)

        # 划分数据集
        dataset_dict = self.split_data(self.traffic_data,
                                       split_ratio=split_ratio)

        # 归一化数据
        self.dataset_dict = self.norm_dict(dataset_dict)

        # 只预测第一个节点
        # self.num_task = self.traffic_data.shape[0]
        self.num_task = 1

    def get_adj(self, traffic_adj_file_path):
        return create_adj_traffic(traffic_adj_file_path)

    def generate_dataset(self, dataset_keyword, batch_size, in_len, out_len, load_mode="complex", shuffle=True):
        if self.is_region:
            graph_x = self.dataset_dict_region[dataset_keyword]["data"][0]
        else:
            graph_x = self.dataset_dict[dataset_keyword]["data"][0]
        dataset = DataSetTraffic(*self.dataset_dict[dataset_keyword]["data"],
                                 graph_x, in_len, out_len, self.num_task,
                                 load_mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return dataloader

    def re_norm_data(self, data, keyword):
        norm_record = self.dataset_dict[keyword]["record"]
        re_normed_data = super().re_norm_data(data, norm_record)
        return re_normed_data

class DataIterMixerTBC(object):
    def __init__(self):
        self.data_iter_list = []
        self.count = 0
        self.adj = None

    def add_data_iter(self, data_iter):
        self.data_iter_list.append(data_iter)

    def choose_dataiter(self):
        choose_one = self.data_iter_list[self.count]
        if self.count < len(self.data_iter_list) - 1:
            self.count += 1
        else:
            self.count = 0
        return choose_one

    def __len__(self):
        return self.data_iter_list.__len__()


def init_data_iter(config):
    choose = config.embedding_choose
    if choose == "TBC":
        data_mixer = DataIterMixerTBC()
        for span in config.span:
            tbc_file_path = rf"data\TBC\Span-{span}"
            config.model_path = r"model_parameter/tbc/GAttention.pth"
            data_iter = DataIterTBC(tbc_file_path, config.split_ratio)
            data_mixer.add_data_iter(data_iter)
        config.out_dim = 4
        config.num_node = data_iter.num_task
        return data_mixer
    elif choose == "Traffic":
        num = 8
        traffic_file_path = rf"data\traffic\PEMS0{num}\pems0{num}.npz"
        adj_file_path = rf"data\traffic\PEMS0{num}\distance.csv"
        config.model_path = r"model_parameter/traffic/GAttention.pth"
        config.out_dim = 1
        data_iter = DataIterTraffic(traffic_file_path, adj_file_path, config.split_ratio, config.is_region)
        config.num_node = data_iter.num_task
        return data_iter
    else:
        raise ValueError("choose should be TBC or Traffic")


if __name__ == '__main__':
    # 测试
    file_path = r'E:\DeskTop\深度学习\nature communication\data\TBC\Span-3'
    data_iter = DataIterTBC(file_path, split_ratio=[0.8, 0.1, 0.1])
    train_loader = data_iter.generate_dataset("train", 64, 74,
                                              28, "simple", shuffle=True)
    for i, data in enumerate(train_loader):
        x, label = data
        print(i, end="  ")
        [print(item.shape, end=" " * 3) for item in x]
        print(label.shape)

    # 测试
    traffic_file_path = r'E:\DeskTop\深度学习\nature communication\data\traffic\PEMS04\pems04.npz'
    traffic_adj_file_path = r'E:\DeskTop\深度学习\nature communication\data\traffic\PEMS04\distance.csv'
    data_iter = DataIterTraffic(traffic_file_path, traffic_adj_file_path, split_ratio=[0.8, 0.1, 0.1], is_region=True)
    train_loader = data_iter.generate_dataset("train", 64, 74,
                                              28, "simple", shuffle=True)
    for i, data in enumerate(train_loader):
        x, label = data
        print(i, end="  ")
        [print(item.shape, end=" " * 3) for item in x]
        print(label.shape)
