import torch.nn as nn
import torch
from GraphAttentionNetwork import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GAT, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.gat_layers = nn.ModuleList(GraphAttentionLayer(hidden_dim, hidden_dim, 8) for i in range(2))

    def forward(self, x, adj):
        """
        params: x: [batch_size, num_node, num_length, in_dim]
        adj: [batch_size, num_node, num_node]
        """
        x = self.linear(x)
        b, n, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, n, d)
        adj = adj.repeat(l, 1, 1)
        for i in range(2):
            x, _ = self.gat_layers[i](x, adj)
        x = x.contiguous().view(b, l, n, d).transpose(1, 2)
        return x,

    def __str__(self):
        return "GAT"