from Embedding import *
from GAttention import GAttention
import torch.nn as nn
from OutHead import *
from LSTM import LSTM


class BasicModel(nn.Module):
    def __init__(self, config, backbone):
        super(BasicModel, self).__init__()
        self.config = config
        final_len = config.in_len
        if config.embedding_choose == 'TBC':
            self.embedding = TBCEmbedding(config.vehicle_dim, config.bridge_dim,
                                          config.pier_dim, config.embed_dim)
            self.out = OutHeadTBC(config.hidden_dim, config.out_dim, config.num_moe_layer)
        elif config.embedding_choose == 'Traffic':
            self.embedding = TrafficEmbedding(config.traffic_dim, config.embed_dim)
            self.out = OutHeadTraffic(config.hidden_dim, config.out_dim, config.num_moe_layer, config.num_node)
        elif config.embedding_choose == 'Weather':
            self.embedding = WeatherEmbedding(config.weather_dim, config.embed_dim)
            self.out = OutHeadWeather(config.embed_dim, config.out_dim, config.hidden_dim, config.num_blocks)
        else:
            raise ValueError("embedding_choose must be 'TBCEmbedding'")

        self.backbone = backbone

    def forward(self, *args, adj):
        graph_x = self.embedding(*args)
        if self.config.embedding_choose == 'Weather':
            # 对于grid没有adj，需要传入的参数是mask
            adj = args[1]
            # 只对部分地方进行mask  - 天气
            mask = 1 - args[1].unsqueeze(1)
        else:
            mask = 1
        graph_out_x = self.backbone(graph_x, adj)
        return self.out(graph_out_x) * mask
