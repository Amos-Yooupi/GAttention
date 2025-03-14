import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from reginal_graph_represention import regional_graph


def create_adj_tbc(num_train_node, num_bridge_node):
    # 创建对应结构tbc的邻接矩阵
    train_idx_list = list(range(0, num_train_node))
    bridge_idx_list = list(range(num_train_node, num_bridge_node + num_train_node))
    pier_idx_list = list(
        range(num_train_node + num_bridge_node, num_train_node + num_bridge_node + num_bridge_node + 1))
    total_node_num = num_train_node + num_bridge_node + num_bridge_node + 1

    edges = []
    for i, train_node_idx in enumerate(train_idx_list):
        # 列车节点和桥梁节点之间有边
        edges.extend([[train_node_idx, bridge_node_idx] for bridge_node_idx in bridge_idx_list])
        # 列车节点和列车节点之间有边
        if i < num_train_node - 1:
            edges.append([train_node_idx, train_node_idx + 1])
        if i > 0:
            edges.append([train_node_idx, train_node_idx - 1])
    for i, bridge_node_idx in enumerate(bridge_idx_list):
        # 桥梁节点和相邻的桥墩节点相互连接
        edges.extend([bridge_node_idx, pier_idx_list[i + j]] for j in range(0, 2))
        # 桥梁节点和相邻的桥梁节点相互连接
        if i < num_bridge_node - 1:
            edges.append([bridge_node_idx, bridge_node_idx+1])
        if i > 0:
            edges.append([bridge_node_idx, bridge_node_idx-1])
    edges = torch.tensor(edges, dtype=torch.int64).T
    adj_matrix = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]),
                                         torch.Size([total_node_num, total_node_num])).to_dense()
    adj_matrix += adj_matrix.t().multiply(adj_matrix.t() > adj_matrix)
    adj_matrix[torch.arange(total_node_num), torch.arange(total_node_num)] = 1   # 加上自环
    return adj_matrix


def create_adj_traffic(adj_file_path):
    edges =pd.read_csv(adj_file_path, encoding="utf-8").to_numpy()[:, :2]
    edges = torch.tensor(edges, dtype=torch.int64).T
    total_node_num = edges.max().item() + 1  # 节点数
    adj_matrix = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]),
                                         torch.Size([total_node_num, total_node_num])).to_dense()
    adj_matrix += adj_matrix.t().multiply(adj_matrix.t() > adj_matrix)
    adj_matrix[torch.arange(total_node_num), torch.arange(total_node_num)] = 1  # 加上自环

    return adj_matrix


def create_regional_graph(adj, graph_x):
    # 形成一个新的图  -- 区域图
    edge_list, node_list, region_list = regional_graph(adj)
    edges = torch.tensor(edge_list, dtype=torch.int64).T
    total_node_num = adj.shape[0]  # 节点数
    adj_matrix = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]),
                                         torch.Size([total_node_num, total_node_num])).to_dense()
    adj_matrix += adj_matrix.t().multiply(adj_matrix.t() > adj_matrix)
    adj_matrix[torch.arange(total_node_num), torch.arange(total_node_num)] = 1  # 加上自环

    node_idx = torch.tensor(node_list, dtype=torch.int64)

    region_infor = torch.stack(region_list, dim=0).to(torch.float64)  # (regional_num, total_num)
    new_graph_x = torch.einsum('ij,jkl->ikl', region_infor, graph_x)  # (regional_num, feature_dim)
    new_graph_x = new_graph_x / torch.sum(region_infor, dim=1, keepdim=True).unsqueeze(dim=-1)  # (regional_num, feature_dim) 求平均
    return adj_matrix[node_idx[:, None], node_idx[None, :]], new_graph_x


if __name__ == '__main__':
    num_train_node = 3
    num_bridge_node = 3
    adj_matrix = create_adj_tbc(num_train_node, num_bridge_node)
    print(adj_matrix)
    traffic_adj_matrix = create_adj_traffic(r"E:\DeskTop\深度学习\nature communication\data\traffic\PEMS04\distance.csv")
    print(traffic_adj_matrix.shape, traffic_adj_matrix)
    # 创建图（使用NetworkX库）
    G = nx.from_numpy_array(traffic_adj_matrix.numpy())
    # 画出图的布局
    plt.figure(figsize=(12, 12))  # 设置图像大小

    # 你可以选择不同的布局方式，这里使用 spring_layout，表示力导向布局
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # 绘制节点和边
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='blue', font_size=8, font_color='black', width=0.5)

    # 显示图形
    plt.title("Graph Visualization from Adjacency Matrix")
    plt.show()
    edge_list, node_list, region_list = regional_graph(traffic_adj_matrix)
    print(edge_list.__len__())
    print(node_list.__len__())
