import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from Load_data import *
from Create_adj import *
import matplotlib.pyplot as plt
import torch.nn as nn
import tqdm as tqdm
import time
import datetime


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
        self.complex_step = 1

    def __len__(self):
        return int((self.vehicle.size()[1] - self.in_len - self.out_len) / (self.out_len
                                                                            if self.mode == "simple"
                                                                            else self.complex_step)) + 1

    def __getitem__(self, idx):
        start_idx = idx * (self.out_len if self.mode == "simple" else self.complex_step)
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
        self.complex_step = 2

    def __len__(self):
        return int((self.region_graph_x.size()[1] - self.in_len - self.out_len) / (self.out_len
                                                                                   if self.mode == "simple"
                                                                                   else self.complex_step))

    def __getitem__(self, idx):
        start_idx = idx * (self.out_len if self.mode == "simple" else self.complex_step)
        end_idx = start_idx + self.in_len
        graph_x = self.region_graph_x[:, start_idx:end_idx]
        label = self.traffic_data[:, end_idx:end_idx + self.out_len]
        num_task = torch.where(self.num_task_node == 1)[0]
        return [graph_x], label[num_task]


class DataSetWeather(Dataset):
    def __init__(self, weather_data, weather_mask, in_len, out_len, mode="complex"):
        # [time_steps, 1, longitude, latitude]
        self.weather_data = weather_data
        self.weather_mask = weather_mask
        self.in_len = in_len
        self.out_len = out_len
        assert mode in ["simple", "complex"], "Mode should be simple or complex"
        self.mode = mode
        self.complex_step = 1

    def __len__(self):
        return int((self.weather_data.size()[0] - self.in_len - self.out_len) / (self.out_len
                                                                                 if self.mode == "simple"
                                                                                 else self.complex_step))

    def __getitem__(self, idx):
        start_idx = idx * (self.out_len if self.mode == "simple" else self.complex_step)
        end_idx = start_idx + self.in_len
        weather_x = self.weather_data[start_idx:end_idx]
        label = self.weather_data[end_idx:end_idx + self.out_len]
        return [weather_x.unsqueeze(dim=1), self.weather_mask], label.unsqueeze(dim=1)


class DataIter(object):
    def __init__(self, time_dim=1):
        self.scaler = Scaler(time_dim)
        self.time_dim = time_dim
        self.label_node_idx = None

    def _norm_data(self, data):
        # 归一化数据
        return self.scaler.transform(data)

    def re_norm_data(self, data, *args):
        return self.scaler.inverse_transform(data, self.label_node_idx)

    def _get_mean_std(self, data):
        self.scaler.fit(data)

    def norm_dict(self, dataset_dict):
        for key, values in dataset_dict.items():
            for idx, item in enumerate(values):
                dataset_dict[key][idx] = self._norm_data(item)
        return dataset_dict

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
            original_slices = [slice(None)] * (self.time_dim + 1)
            train_slice = original_slices.copy()
            val_slice = original_slices.copy()
            test_slice = original_slices.copy()
            train_slice[self.time_dim] = slice(0, train_length)
            val_slice[self.time_dim] = slice(train_length, train_length + val_length)
            test_slice[self.time_dim] = slice(train_length + val_length, None)
            train_data.append(data[tuple(train_slice)])
            val_data.append(data[tuple(val_slice)])
            test_data.append(data[tuple(test_slice)])
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
        self.vehicle, self.bridge, self.pier = read_file_TBC(data_file_path, time_freq=40)
        self.scaler_train = Scaler(1)
        self.scaler_bridge = Scaler(1)
        self.scaler_pier = Scaler(1)
        # # 划分数据集
        self.dataset_dict = self.split_data(self.vehicle,
                                            self.bridge, self.pier,
                                            split_ratio=split_ratio)
        # 归一化数据
        self._get_mean_std()
        self._norm_dcit()

        # 设置参数
        self.adj, self.num_train, self.num_bridge = None, None, None
        self.get_adj()
        # 只预测列车节点的响应
        self.num_task = self.vehicle.shape[0]
        self.label_node_idx = torch.tensor([0, 1, 2])

        # self.set_train_dis_to_zero()
        self.name = f"TBC"

    def _get_mean_std(self, *args):
        train, bridge, pier = self.dataset_dict["train"]
        self.scaler_train.fit(self.dataset_dict["train"][0])
        self.scaler_bridge.fit(self.dataset_dict["train"][1])
        self.scaler_pier.fit(self.dataset_dict["train"][2])

    def _norm_dcit(self):
        maps = {0: "train", 1: "bridge", 2: "pier"}
        for key, value in self.dataset_dict.items():
            for idx, item in enumerate(value):
                scaler = self._return_correspond_scaler(maps[idx])
                self.dataset_dict[key][idx] = scaler.transform(item)

    def set_train_dis_to_zero(self):
        self.vehicle[:, :, 2:] = 0

    def get_adj(self):
        # 取出列车的数量 和  桥梁的数量
        self.num_train = self.vehicle.shape[0]
        self.num_bridge = self.bridge.shape[0]

        self.adj = create_adj_tbc(self.num_train, self.num_bridge)

    def generate_dataset(self, dataset_keyword, batch_size, in_len, out_len, load_mode="complex", shuffle=True):
        # dataset = DataSetTBC(*self.dataset_dict[dataset_keyword], in_len, out_len, self.num_task, load_mode)
        dataset = DataSetTBC(*self.dataset_dict[dataset_keyword], in_len, out_len, self.num_task, load_mode)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return dataloader

    def _re_norm_data(self, data, keyword):
        scaler = self._return_correspond_scaler(keyword)
        return scaler.inverse_transform(data)

    def _return_correspond_scaler(self, keyword_):
        if keyword_ == "train":
            return self.scaler_train
        elif keyword_ == "bridge":
            return self.scaler_bridge
        elif keyword_ == "pier":
            return self.scaler_pier
        else:
            raise ValueError("Invalid keyword")

    def __str__(self):
        return f"DataIterTBC-Span-{self.num_bridge}-Train-{self.num_train}"


class DataIterTraffic(DataIter):
    def __init__(self, traffic_data_file_path, traffic_adj_file_path, split_ratio: list, is_region=False,
                 region_order=1):
        super().__init__()
        self.traffic_data = read_file_traffic(traffic_data_file_path)
        print("Ori:", self.traffic_data.element_size() * self.traffic_data.numel() / 1024 ** 2, "MB")
        # 设置参数
        self.adj = self.get_adj(traffic_adj_file_path)
        self.is_region = is_region

        # 归一化数据 --- 划分数据集
        dataset_dict = self.split_data(self.traffic_data,
                                       split_ratio=split_ratio)
        self._get_mean_std(dataset_dict["train"][0])
        self.traffic_data = self.scaler.transform(self.traffic_data)  # 这个是label，也需要归一化
        self.label_dataset_dict = self.norm_dict(dataset_dict)

        # 生成区域图
        self.region_adj, self.region_graph_x, self.region_infor, self.node_list = create_regional_graph(self.adj,
                                                                                                        self.traffic_data,
                                                                                                        k=region_order)
        print(self.region_adj.shape, self.adj.shape)
        print("Regional graph:", self.region_graph_x.element_size() * self.region_graph_x.numel() / 1024 ** 2, "MB")
        if self.is_region:
            self.adj = self.region_adj
            self.graph_x = self.region_graph_x
        else:
            self.graph_x = self.traffic_data
        # 生成新的数据集
        self.dataset_dict = self.split_data(self.graph_x,
                                            split_ratio=split_ratio)  # 已经归一化了，不需要归一化
        # 只预测第一个区域的节点
        self.region_order = region_order
        region_idx = 0  # 预测区域的索引
        print("Region:", "--" * 3, region_idx)
        self.num_task = self.region_infor[region_idx].int()
        self.name = "Traffic"
        self.node_idx = torch.Tensor(self.node_list[region_idx])
        self.label_node_idx = torch.where(self.num_task > 0)[0]

    def __str__(self):
        return f"DataIterTraffic-Region-{self.is_region}-Region-Order-{self.region_order}-Train-{self.num_task.sum()}-Node idx-{self.node_idx}"

    def show_adj(self, adj):
        plt.imshow(adj)
        # 调整x轴和y轴刻度的字体大小
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def get_adj(self, traffic_adj_file_path):
        return create_adj_traffic(traffic_adj_file_path)

    def generate_dataset(self, dataset_keyword, batch_size, in_len, out_len, load_mode="complex", shuffle=True):
        graph_x = self.dataset_dict[dataset_keyword][0]
        dataset = DataSetTraffic(self.label_dataset_dict[dataset_keyword][0],
                                 graph_x, in_len, out_len, self.num_task,
                                 load_mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return dataloader


class DataIterWeather(DataIter):
    def __init__(self, weather_data_file_path, weather_data_mask_file_path, split_ratio: list):
        super().__init__(time_dim=0)
        # 读取天气数据 [time, longitude, latitude]
        self.adj = torch.zeros([1])
        self.weather = torch.load(weather_data_file_path)
        # 读取天气数据mask [1, longitude, latitude]
        self.weather_mask = torch.load(weather_data_mask_file_path)[0].unsqueeze(0)
        time_steps, num_longitude, num_latitude = self.weather.shape
        # 划分数据集
        dataset_dict = self.split_data(self.weather,
                                       split_ratio=split_ratio)
        # 归一化数据
        self._get_mean_std(dataset_dict["train"][0])
        self.dataset_dict = self.norm_dict(dataset_dict)

        # 设置参数  --- 预测温度
        self.num_task = 1
        self.name = "Weather"

    def generate_dataset(self, dataset_keyword, batch_size, in_len, out_len, load_mode="complex", shuffle=True):
        conv_x = self.dataset_dict[dataset_keyword][0]
        dataset = DataSetWeather(conv_x, self.weather_mask, in_len, out_len, load_mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return dataloader

    def split_data(self, *original_data, split_ratio: list):
        train_ratio, val_ratio, test_ratio = split_ratio
        train_data, val_data, test_data = [], [], []
        for data in original_data:
            total_data_length = data.shape[0]
            train_length = int(total_data_length * train_ratio)
            val_length = int(total_data_length * val_ratio)
            train_data.append(data[:train_length])
            val_data.append(data[train_length:train_length + val_length])
            test_data.append(data[train_length + val_length:])
        return {"train": train_data, "val": val_data, "test": test_data}


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

    def __str__(self):
        for data_iter in self.data_iter_list:
            print(data_iter)
        return ""


def init_data_iter(config):
    choose = config.embedding_choose
    if choose == "TBC":
        data_mixer = DataIterMixerTBC()
        for span in config.span:
            for train in config.train:
                tbc_file_path = rf"data\TBC\Span_{span}_Car_{train}"
                # config.model_path = r"model_parameter/tbc/GAttention.pth"
                data_iter = DataIterTBC(tbc_file_path, config.split_ratio)
                data_mixer.add_data_iter(data_iter)
        print(data_mixer)
        config.out_dim = 4
        config.num_node = data_iter.num_task
        return data_mixer
    elif choose == "Traffic":
        num = 8
        traffic_file_path = rf"data\traffic\PEMS0{num}\pems0{num}.npz"
        adj_file_path = rf"data\traffic\PEMS0{num}\distance.csv"
        # config.model_path = r"model_parameter/traffic/GAttention.pth"
        config.out_dim = 1
        data_iter = DataIterTraffic(traffic_file_path, adj_file_path, config.split_ratio, config.is_region,
                                    config.region_order)
        config.node_idx = data_iter.node_idx
        config.num_node = data_iter.num_task
        return data_iter
    elif choose == "Weather":
        weather_file_path = r"data\Weather\weather_data.pt"
        weather_mask_path = r"data\Weather\weather_mask.pt"
        # config.model_path = r"model_parameter/weather/GAttention.pth"
        config.out_dim = 1
        data_iter = DataIterWeather(weather_file_path, weather_mask_path, config.split_ratio)
        config.num_task = data_iter.num_task
        return data_iter
    else:
        raise ValueError("choose should be TBC or Traffic， Weather")


class Scaler(object):
    def __init__(self, scale_dim=-1):
        self.scale_mean = None
        self.scale_std = None
        self.scale_dim = scale_dim

    def fit(self, data):
        self.scale_mean = torch.mean(data, dim=self.scale_dim, keepdim=True)
        self.scale_std = torch.std(data, dim=self.scale_dim, keepdim=True)

    def transform(self, data):
        return (data - self.scale_mean) / self.scale_std

    def inverse_transform(self, data, node_idx=None):
        return data * self.scale_std[node_idx] + self.scale_mean[node_idx]


if __name__ == '__main__':
    # 测试
    file_path = r"E:\DeskTop\深度学习\nature communication\data\TBC\Span_5_Car_4"
    data_iter = DataIterTBC(file_path, split_ratio=[0.8, 0.1, 0.1])
    train_loader = data_iter.generate_dataset("train", 64, 60,
                                              60, "simple", shuffle=True)
    for i, data in enumerate(train_loader):
        x, label = data
        print(i, end="  ")
        [print(item.shape, end=" " * 3) for item in x]
        print(label.shape)

    # 测试
    traffic_file_path = r'E:\DeskTop\深度学习\nature communication\data\traffic\PEMS08\pems08.npz'
    traffic_adj_file_path = r'E:\DeskTop\深度学习\nature communication\data\traffic\PEMS08\distance.csv'
    data_iter = DataIterTraffic(traffic_file_path, traffic_adj_file_path, split_ratio=[0.8, 0.1, 0.1], is_region=True,
                                region_order=1)
    train_loader = data_iter.generate_dataset("train", 64, 60,
                                              60, "simple", shuffle=True)
    weather_loader = DataIterWeather(r'E:\DeskTop\深度学习\nature communication\data\Weather\weather_data.pt',
                                     r'E:\DeskTop\深度学习\nature communication\data\Weather\weather_mask.pt',
                                     split_ratio=[0.8, 0.1, 0.1]).generate_dataset("train", 64, 12, 12)

    print("Weather", "--" * 10)
    for i in weather_loader:
        x, label = i
        [print(item.shape, end=" " * 3) for item in x]
        print(label.shape)
    print("Traffic", "--" * 10)
    for i, data in enumerate(train_loader):
        x, label = data
        print(i, end="  ")
        [print(item.shape, end=" " * 3) for item in x]
        print(label.shape)
