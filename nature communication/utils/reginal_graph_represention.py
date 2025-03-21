import torch


def regional_graph(adj: torch.Tensor):
    original_adj = adj.clone()
    region_node = []
    node_list = []
    edge_list = []
    while True:
        degree_matrix = torch.sum(adj, dim=1)
        # 找出连接最多的节点
        _, node_idx = torch.max(degree_matrix, dim=0)
        # 将这个作为作为作为一个大节点
        region_node.append(original_adj[node_idx])
        node_list.append(node_idx)
        # 直接将已经进行过得节点清零
        selected_node = torch.where(region_node[-1]== 1)[0]
        adj[selected_node, :] = 0
        adj[node_idx, :] = 0
        # 判断这个节点是否和其他节点相互连接
        for sequence, exist_node in enumerate(node_list):
            # 得到exist的节点，只要他们有重复的部分
            equal_item = region_node[-1] == region_node[sequence]
            # 判断是否存在1
            equal_item = torch.where(equal_item)[0]
            result = region_node[-1][equal_item].sum()
            if result:
                edge_list.append([exist_node.item(), node_idx.item()])
        if torch.sum(adj) == 0:
            break

    return edge_list, node_list, region_node


if __name__ == '__main__':
    adj = torch.tensor([[1., 1., 0., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 1., 1., 0.],
        [1., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 1.]])
    edge_list, node_list, region_node = regional_graph(adj)
    print(edge_list)
    print(node_list)
    print(region_node)