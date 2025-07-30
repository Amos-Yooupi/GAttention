import torch.nn as nn
import torch
import math


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size, dropout=0, max_len=500):
        super().__init__()
        self.pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        assert x.shape.__len__() == 3, print("the shape of input x should be 3")
        B, L, D = x.shape
        assert D == self.pe.shape[-1], print("The dim of x should be equal to embedding size")
        return self.dropout(x + self.pe[:L, :][None, :, :].to(x.device))


if __name__ == '__main__':
    x = torch.rand((64, 20, 512))
    PE = PositionEmbedding(512)
    print(PE(x).shape)
