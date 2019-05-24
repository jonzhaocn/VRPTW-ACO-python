import numpy as np


class Node:
    def __init__(self, id:  int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float):
        super()
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


class VPRTW_Graph:
    def __init__(self, file_path):
        super()
        # node_num 结点个数
        # node_dist_mat 节点之间的距离（矩阵）
        # pheromone_mat 节点之间路径上的信息度浓度
        self.node_num, self.nodes, self.node_dist_mat, self.vehicle_num, self.vehicle_capacity \
            = self.create_from_file(file_path)

    def create_from_file(self, file_path):
        # 从文件中读取服务点、客户的位置
        node_list = []
        with open(file_path, 'rt') as f:
            count = 1
            for line in f:
                if count == 5:
                    vehicle_num, vehicle_capacity = line.split()
                    vehicle_num = int(vehicle_num)
                    vehicle_capacity = int(vehicle_capacity)
                elif count >= 10:
                    node_list.append(line.split())
                count += 1
        node_num = len(node_list)
        nodes = list(Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]), float(item[6])) for item in node_list)

        # 创建距离矩阵
        node_dist_mat = np.zeros((node_num, node_num))
        for i in range(node_num):
            node_a = nodes[i]
            node_dist_mat[i][i] = np.inf
            for j in range(i+1, node_num):
                node_b = nodes[j]
                node_dist_mat[i][j] = VPRTW_Graph.calculate_dist(node_a, node_b)
                node_dist_mat[j][i] = node_dist_mat[i][j]

        return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity

    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))


class Ant:
    def __init__(self, node_num):
        super()
        self.current_index = 0
        self.vehicle_load = 0
        self.vehicle_travel_distance = 0
        self.current_time = 0
        self.travel_path = [0]
        self.arrival_time = [0]
        self.index_to_visit = list(range(1, node_num))

    def move_to_next_index(self, graph, vehicle_speed, next_index):
        # 更新蚂蚁路径
        self.travel_path.append(next_index)
        self.arrival_time.append(self.current_time)

        if next_index == 0:
            # 如果一下个位置为服务器点，则要将车辆负载等清空
            self.vehicle_load = 0
            self.vehicle_travel_distance = 0
            self.current_time = 0

        else:
            # 更新车辆负载、行驶距离、时间
            self.vehicle_load += graph.nodes[next_index].demand
            self.vehicle_travel_distance += graph.node_dist_mat[self.current_index][next_index]
            self.current_time += (graph.node_dist_mat[self.current_index][next_index] / vehicle_speed + graph.nodes[next_index].service_time)
            self.index_to_visit.remove(next_index)

        self.current_index = next_index

    def index_to_visit_empty(self):
        return len(self.index_to_visit) == 0