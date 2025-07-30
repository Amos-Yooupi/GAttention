import torch.nn as nn
import torch
from PositionEmbedding import PositionEmbedding
from ConvBlock import DepthWiseConv
import matplotlib.pyplot as plt
import os


class TBCEmbedding(nn.Module):
    def __init__(self, vehicle_dim, bridge_dim, pier_dim, embed_dim):
        super(TBCEmbedding, self).__init__()

        self.vehicle_embedding = nn.Linear(vehicle_dim, embed_dim)
        self.bridge_embedding = nn.Linear(bridge_dim, embed_dim)
        self.pier_embedding = nn.Linear(pier_dim, embed_dim)
        self.pe = PositionEmbedding(embed_dim)

    def forward(self, vehicle_features, bridge_features, pier_features):
        """
        Args:
            vehicle_features: (batch_size, num_node, time_step, vehicle_dim) -> (B, N, T, D)
            bridge_features: (batch_size, num_node,time_step, bridge_dim)
            pier_features: (batch_size, num_node,time_step, pier_dim)
        Returns:
            graph_embedding_x: (batch_size, total_num_node, time_step, embed_dim)
        """

        vehicle_embedding = self.vehicle_embedding(vehicle_features)
        bridge_embedding = self.bridge_embedding(bridge_features)
        bridge_embedding[:] = 0  # 遮掩桥梁的数据，不能用到 可以用其他固定属性特征代替
        pier_embedding = self.pier_embedding(pier_features)

        graph_embedding_x = torch.cat([vehicle_embedding, bridge_embedding, pier_embedding], dim=1)
        B, N, L, D = graph_embedding_x.shape
        graph_embedding_x = self.pe(graph_embedding_x.view(B*N, L, D)).view(B, N, L, D)

        return graph_embedding_x


class TrafficEmbedding(nn.Module):
    def __init__(self, traffic_dim, embed_dim):
        super(TrafficEmbedding, self).__init__()
        self.traffic_embedding = nn.Sequential(nn.Linear(traffic_dim, embed_dim), nn.Tanh())
        self.count = 1

    def forward(self, traffic_features):
        """
        traffic_features: (batch_size, num_node, traffic_dim, time_step)
        """
        traffic_embedding = self.traffic_embedding(traffic_features)
        self.count += 1
        return traffic_embedding


class WeatherEmbedding(nn.Module):
    def __init__(self, weather_dim, embed_dim):
        super(WeatherEmbedding, self).__init__()
        self.weather_embedding = nn.Sequential(nn.Conv2d(weather_dim, embed_dim, kernel_size=5, padding=2), nn.Tanh())

    def forward(self, weather_features, mask, *ergs):
        """
        weather_features: (batch_size, time_step, feature_dim, H, W)
        """
        B, L, D, H, W = weather_features.shape
        mask = mask.repeat(L, 1, 1, 1)
        weather_features = weather_features.contiguous().view(B*L, D, H, W)
        weather_embedding = self.weather_embedding(weather_features) * (1-mask)
        weather_embedding = weather_embedding.contiguous().view(B, L, -1, H, W)
        return weather_embedding