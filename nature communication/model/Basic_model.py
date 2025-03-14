from Embedding import *
from GAttention import GAttention
import torch.nn as nn
from OutHead import *
from LSTM import LSTM


class BasicModel(nn.Module):
    def __init__(self, config, backbone):
        super(BasicModel, self).__init__()
        if backbone.name == "GAttention":
            final_len = int(sum([config.in_len / 2 ** (i+1) for i in range(config.num_blocks)]))
        else:
            final_len = config.in_len
        if config.embedding_choose == 'TBC':
            self.embedding = TBCEmbedding(config.vehicle_dim, config.bridge_dim,
                                          config.pier_dim, config.embed_dim)
            self.out = OutHeadTBC(config.hidden_dim, final_len, config.out_dim,
                                  config.out_len, config.num_moe_layer)
        elif config.embedding_choose == 'Traffic':
            self.embedding = TrafficEmbedding(config.traffic_dim, config.embed_dim)
            self.out = OutHeadTraffic(config.hidden_dim, final_len, config.out_dim,
                                  config.out_len, config.num_moe_layer, config.num_node)
        else:
            raise ValueError("embedding_choose must be 'TBCEmbedding'")

        self.backbone = backbone

    def forward(self, *args, adj):
        graph_x = self.embedding(*args)
        graph_out_x = self.backbone(graph_x, adj)
        return self.out(graph_out_x)
