import torch.nn as nn
import torch


class TBCEmbedding(nn.Module):
    def __init__(self, vehicle_dim, bridge_dim, pier_dim, embed_dim):
        super(TBCEmbedding, self).__init__()

        self.vehicle_embedding = nn.Linear(vehicle_dim, embed_dim)
        self.bridge_embedding = nn.Linear(bridge_dim, embed_dim)
        self.pier_embedding = nn.Linear(pier_dim, embed_dim)

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

        return graph_embedding_x


class TrafficEmbedding(nn.Module):
    def __init__(self, traffic_dim, embed_dim):
        super(TrafficEmbedding, self).__init__()
        self.traffic_embedding = nn.Linear(traffic_dim, embed_dim)

    def forward(self, traffic_features):
        """
        traffic_features: (batch_size, num_node, traffic_dim, time_step)
        """
        traffic_embedding = self.traffic_embedding(traffic_features)
        return traffic_embedding