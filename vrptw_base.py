import numpy as np


class Node:
    def __init__(self, x, y, demand, earliest_time, latest_time, service_time):
        super()
        self.x = x
        self.y = y
        self.demand = demand
        self.earliest_time = earliest_time
        self.latest_time = latest_time
        self.service_time = service_time


class Graph:
    def __init__(self, file_path):
        super()
        # node_num 结点个数
        # node_dist_mat 节点之间的距离（矩阵）
        # pheromone_mat 节点之间路径上的信息度浓度
        self.node_num, self.nodes, self.node_dist_mat = self.create_from_file(file_path)

    def create_from_file(self, file_path):
        # 从文件中读取服务点、客户的位置
        with open(file_path, 'rt') as f:
            node_list = list(line.split() for line in f)
        node_num = len(node_list)
        nodes = list(Node(float(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])) for item in node_list)

        # 创建距离矩阵
        node_dist_mat = np.zeros((node_num, node_num))
        for i in range(node_num):
            node_a = nodes[i]
            for j in range(i, node_num):
                node_b = nodes[j]
                node_dist_mat[i][j] = Graph.calculate_dist(node_a, node_b)
                node_dist_mat[j][i] = node_dist_mat[i][j]

        return node_num, nodes, node_dist_mat

    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))