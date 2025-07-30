import torch.nn as nn
import torch.nn.functional as F
import torch
from timer import timer
from show_attention import show_attention_maps


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, att_mode="linear", activation="relu"):
        super(GraphAttentionLayer, self).__init__()
        assert in_dim % n_heads == 0, print("Number of features should divide by heads")
        assert out_dim % n_heads == 0, print("Number of features should divide by heads")
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU()
        else:
            raise ValueError("Invalid activation function")
        self.in_dim = in_dim // n_heads
        self.out_dim = out_dim // n_heads
        self.n_heads = n_heads
        self.fc = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), activation)
        if att_mode == "linear":
            self.att_layer = nn.Linear(self.in_dim*2, 1)
        elif att_mode == "dot":
            self.att_layer = None
        else:
            raise ValueError("Invalid attention mode")
        self.att_mode = att_mode
        self.att_weight = None

    @timer(False, "GraphAttentionLayer")
    def forward(self, x, adj=None):
        """
        Args:
            x: input features with shape (batch_size, num_nodes, in_features)
            adj: adjacency matrix with shape (batch_size, num_nodes, num_nodes)
        """
        x, adj = self.transpose_x(x, adj)  # (batch_size * n_heads, num_nodes, in_dim)
        if self.att_mode == "linear":
            att_prob = self.linear_attention(x, adj)  # (batch_size * n_heads, num_nodes, num_nodes)
        else:
            att_prob = self.dot_attention(x, adj)  # (batch_size * n_heads, num_nodes, num_nodes)
        # self.att_weight = att_prob.detach().cpu().view(-1, self.n_heads, att_prob.size(1), att_prob.size(2))
        # 进行注意力操作
        att_result_x = torch.bmm(att_prob, x)  # (batch_size * n_heads, num_nodes, in_dim)
        return self.restore_x(self.fc(att_result_x)), att_prob.detach().cpu()

    def linear_attention(self, x, adj):
        """
        Args:
            x: input features with shape (batch_size * n_heads, num_nodes, in_dim)
            adj: adjacency matrix with shape (batch_size * n_heads, num_nodes, num_nodes)
        """
        x_i = x @ self.att_layer.weight.t()[:self.in_dim]  # (batch_size * n_heads, num_nodes, 1)
        x_j = (x @ self.att_layer.weight.t()[self.in_dim:]).squeeze().unsqueeze(dim=1)  # (batch_size * n_heads, num_nodes, 1)
        att_prob = F.leaky_relu(x_i + x_j)  # (batch_size * n_heads, num_nodes, num_nodes)
        # 进行mask操作，将不相邻的节点的注意力权重置为0
        if adj is not None:
            mask_score = torch.zeros_like(att_prob).fill_(-1e9).to(x.device, dtype=x.dtype)
            att_prob = torch.where(adj.bool(), att_prob, mask_score)  # (batch_size * n_heads, num_nodes, num_nodes)
        else:
            pass
        att_prob = F.softmax(att_prob, dim=2)  # (batch_size * n_heads, num_nodes, num_nodes)
        return att_prob

    def dot_attention(self, x, adj):
        """
        Args:
            x: input features with shape (batch_size * n_heads, num_nodes, in_dim)
            adj: adjacency matrix with shape (batch_size * n_heads, num_nodes, num_nodes)
        """
        att_prob = x @ x.transpose(1, 2) / torch.sqrt(torch.tensor(self.in_dim))  # (batch_size * n_heads, num_nodes, num_nodes)
        if adj is not None:
            mask_score = torch.zeros_like(att_prob).fill_(-1e9).to(x.device, dtype=x.dtype)
            att_prob = torch.where(adj.bool(), att_prob, mask_score)  # (batch_size * n_heads, num_nodes, num_nodes)
        else:
            pass
        att_prob = F.softmax(att_prob, dim=2)  # (batch_size * n_heads, num_nodes, num_nodes)
        return att_prob

    def transpose_x(self, x, adj):
        b, n, _ = x.size()
        x = x.contiguous().view(b, n, self.n_heads, self.in_dim).permute(0, 2, 1, 3)
        return x.contiguous().view(b * self.n_heads, n, self.in_dim), adj.repeat(self.n_heads, 1, 1) if adj is not None else None

    def restore_x(self, x):
        b_hat, n, _ = x.size()
        x = x.contiguous().view(-1, self.n_heads, n, self.out_dim).permute(0, 2, 1, 3)
        return x.contiguous().view(-1, n, self.n_heads * self.out_dim)


class FusionGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, att_mode="linear", activation="relu"):
        super(FusionGraphAttentionLayer, self).__init__()
        self.gat_layer = GraphAttentionLayer(in_features, out_features, n_heads, att_mode, activation)

        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU()
        else:
            raise ValueError("Invalid activation function")
        self.fc = nn.Sequential(nn.Linear(in_features, out_features), activation)
        self.gate = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
        self.att_score = None

    @timer(False, "FusionGraphAttentionLayer")
    def forward(self, *args):
        x, adj = args[0], args[1]
        gate_values = self.gate(x)  # (B, N, 1)
        # 混合注意力机制
        att_out, att_prob = self.gat_layer(x, adj=None)
        self.att_score = att_prob.detach().cpu()
        graph_message_x = torch.bmm(adj, x)
        # 融合注意力机制
        fusion_out = self.fc(graph_message_x) * gate_values + att_out * (1 - gate_values)
        return fusion_out, adj


class GraphNeuralNetwork(nn.Module):
    def __init__(self, in_features, activation="relu"):
        super(GraphNeuralNetwork, self).__init__()
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU()
        else:
            raise ValueError("Invalid activation function")
        self.linear = nn.Sequential(nn.Linear(in_features, in_features), activation)

    def forward(self, x, adj):
        return self.linear(torch.bmm(adj, x)), adj


if __name__ == '__main__':
    num_nodes = 20
    torch.seed()
    x = torch.randn(64, num_nodes, 256)
    adj = torch.ones(64, num_nodes, num_nodes)
    model = FusionGraphAttentionLayer(256, 128, 8)
    out, att_prob = model(x, adj)
    print(out.shape)