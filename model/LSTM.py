import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, graph_x, adj, *args):
        B, N, L, D = graph_x.shape
        x = graph_x.contiguous().view(B*N, L, D)
        result, _ = self.lstm(x)
        # 变成(B*N, L, D)
        result = result.contiguous().view(B, N, L, -1).sum(dim=1)
        return result

    def __repr__(self):
        return "LSTM"

    def name(self):
        return "LSTM"

