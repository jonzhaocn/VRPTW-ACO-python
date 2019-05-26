import numpy as np


class Node:
    def __init__(self, id:  int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float):
        super()
        self.id = id

        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False

        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


class VrptwGraph:
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
                node_dist_mat[i][j] = VrptwGraph.calculate_dist(node_a, node_b)
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
        self.vehicle_travel_time = 0
        self.travel_path = [0]
        self.arrival_time = [0]
        self.index_to_visit = list(range(1, node_num))
        self.total_travel_distance = 0

    def move_to_next_index(self, graph, next_index):
        # 更新蚂蚁路径
        self.travel_path.append(next_index)
        self.total_travel_distance += graph.node_dist_mat[self.current_index][next_index]

        dist = graph.node_dist_mat[self.current_index][next_index]
        self.arrival_time.append(self.vehicle_travel_time + dist)

        if next_index == 0:
            # 如果一下个位置为服务器点，则要将车辆负载等清空
            self.vehicle_load = 0
            self.vehicle_travel_time = 0

        else:
            # 更新车辆负载、行驶距离、时间
            self.vehicle_load += graph.nodes[next_index].demand
            # 如果早于客户要求的时间窗(ready_time)，则需要等待

            self.vehicle_travel_time += dist + max(graph.nodes[next_index].ready_time - self.vehicle_travel_time - dist, 0) + graph.nodes[next_index].service_time
            self.index_to_visit.remove(next_index)

        self.current_index = next_index

    def index_to_visit_empty(self):
        return len(self.index_to_visit) == 0


class NearestNeighborHeuristic:
    def __init__(self, graph: VrptwGraph):
        self.graph = graph

    def construct_path(self):
        """
        不考虑使用的车辆的数目，调用近邻点算法构造路径
        :return:
        """
        ant = Ant(self.graph.node_num)
        while not ant.index_to_visit_empty():
            customers_meet_constrains = self._cal_customers_meet_constrains(ant)
            if len(customers_meet_constrains) == 0:
                next_index = 0
            else:
                next_index = self._cal_nearest_customer(customers_meet_constrains, ant)

            ant.move_to_next_index(self.graph, next_index)
        ant.move_to_next_index(self.graph, 0)

        return ant

    def cal_init_pheromone_val(self):
        ant = self.construct_path()
        init_pheromone = (1 / (self.graph.node_num * ant.total_travel_distance))
        return init_pheromone

    def _cal_customers_meet_constrains(self, ant: Ant):
        """
        找出所有从当前位置（ant.current_index）可达的customer
        :param ant:
        :return:
        """
        customers_meet_constrains = []
        current_ind = ant.current_index
        for next_ind in ant.index_to_visit:
            condition1 = ant.vehicle_travel_time + self.graph.node_dist_mat[current_ind][next_ind] <= self.graph.nodes[next_ind].due_time
            condition2 = ant.vehicle_load + self.graph.nodes[next_ind].demand <= self.graph.vehicle_capacity
            if condition1 and condition2:
                customers_meet_constrains.append(next_ind)
        return customers_meet_constrains

    def _cal_nearest_customer(self, customers, ant: Ant):
        """
        从待选的customers中选择，离当前位置（ant.current_index）最近的customer
        :param customers:
        :param ant:
        :return:
        """
        current_ind = ant.current_index

        nearest_ind = customers[0]
        min_dist = self.graph.node_dist_mat[current_ind][customers[0]]

        for next_ind in customers[1:]:
            dist = self.graph.node_dist_mat[current_ind][next_ind]
            if dist < min_dist:
                min_dist = dist
                nearest_ind = next_ind

        return nearest_ind
