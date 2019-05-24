import numpy as np
from vrptw_base import VPRTW_Graph, Ant
import random


class VRPTW_ACO:
    def __init__(self, graph: VPRTW_Graph, ants_num=50, max_iter=300, alpha=1, beta=1, rho=0.1):
        super()
        # 结点的位置、服务时间信息
        self.graph = graph
        # 蚂蚁数量
        self.ants_num = ants_num
        # 最大迭代次数
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        # 信息素挥发速度
        self.rho = rho
        # q0 表示直接选择概率最大的下一点的概率
        self.q0 = 0.1
        # Q 表示每辆车的最大载重
        self.max_load = graph.vehicle_capacity
        # L 表示每辆车的最远行驶距离
        self.max_travel_distance = 2000
        # 出车的单位成本
        self.car_cost = 50
        # 行驶的单位成本
        self.travel_cost = 1
        # 早到时间成本
        self.before_ready_time_cost = 1
        # 晚到时间成本
        self.after_due_time_cost = 2
        # 信息素强度
        self.Q = 1
        # 车辆行驶速度
        self.vehicle_speed = 60
        # 创建信息素矩阵
        self.pheromone_mat = np.ones((graph.node_num, graph.node_num))
        self.eta_mat = 1 / graph.node_dist_mat

        # best path
        self.best_path_distance = None
        self.best_path = None

    def run(self):
        """
        运行蚁群优化算法
        :return:
        """
        # 最大迭代次数
        for iter in range(self.max_iter):
            transition_prob = np.zeros((self.graph.node_num, self.graph.node_num))
            # 为每只蚂蚁设置当前车辆负载，当前旅行距离，当前时间
            ants = list(Ant(self.graph.node_num) for _ in range(self.ants_num))
            for k in range(self.ants_num):

                # 蚂蚁需要访问完所有的客户
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k], transition_prob)
                    # 判断加入该位置后，是否还满足约束条件, 如果不满足，则再选择一次，然后再进行判断
                    if not self.check_condition(ants[k], next_index):
                        next_index = self.select_next_index(ants[k], transition_prob)
                        if not self.check_condition(ants[k], next_index):
                            next_index = 0

                    # 更新蚂蚁路径
                    ants[k].move_to_next_index(self.graph, self.vehicle_speed, next_index)

                # 最终回到0位置
                ants[k].move_to_next_index(self.graph, self.vehicle_speed, 0)

            # 计算所有蚂蚁的路径长度
            paths_distance = self.calculate_all_path_distance(ants)

            # 记录当前的最佳路径
            best_index = np.argmin(paths_distance)
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[best_index].travel_path
                self.best_path_distance = paths_distance[best_index]
            print('[iteration %d]: best_cost %f' % (iter, self.best_path_distance))
            # 更新信息素表
            self.update_pheromone_mat(ants, paths_distance)

    def select_next_index(self, ant: Ant, transition_prob):
        """
        选择下一个结点
        :param ant:
        :param transition_prob:
        :return:
        """
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        transition_prob[ant.current_index][ant.index_to_visit] = np.power(self.pheromone_mat[current_index][index_to_visit], self.alpha) * \
            np.power(self.eta_mat[current_index][index_to_visit], self.beta)
        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob[current_index][index_to_visit])
            next_index = index_to_visit[max_prob_index]
        else:
            # 使用轮盘度
            next_index = self.stochastic_accept(index_to_visit, transition_prob[current_index][index_to_visit])
        return next_index

    def check_condition(self, ant: Ant, next_index) -> bool:
        """
        检查移动到下一个点是否满足约束条件
        :param ant:
        :param next_index:
        :return:
        """
        current_index = ant.current_index
        if ant.vehicle_load + self.graph.nodes[next_index].demand > self.max_load:
            return False
        if ant.vehicle_travel_distance + self.graph.node_dist_mat[current_index][next_index] + self.graph.node_dist_mat[next_index][0] > self.max_travel_distance:
            return False
        return True

    def update_pheromone_mat(self, ants, paths_distance):
        """
        更新信息素矩阵
        :return:
        """
        self.pheromone_mat = (1 - self.rho) * self.pheromone_mat
        for k in range(self.ants_num):
            path = ants[k].travel_path
            current_index = path[0]
            for index in path[1:]:
                self.pheromone_mat[current_index][index] += self.Q / paths_distance[k]

    def calculate_all_path_cost(self, ants):
        """
        计算所有蚂蚁行走路径的cost
        :param paths:
        :return:
        """
        # 注意路径是否是从0开始到0结束的
        costs = np.zeros(self.ants_num)
        for k in range(self.ants_num):
            path = ants[k].travel_path
            current_index = path[0]
            for index in path[1:]:
                if index == 0:
                    costs[k] += self.car_cost + self.travel_cost * self.graph.node_dist_mat[current_index][index]
                else:
                    costs[k] += self.travel_cost * self.graph.node_dist_mat[current_index][index]
                    # 这里如何计算是否是早到了，还是迟到了，我还没有记录时间
                    costs[k] += 0
                current_index = index

        return costs

    def calculate_all_path_distance(self, ants: list):
        """
        计算所有蚂蚁的行走路径的长度
        :param paths:
        :return:
        """
        distances = np.zeros(self.ants_num)
        for k in range(self.ants_num):
            path = ants[k].travel_path
            current_index = path[0]
            for index in path[1:]:
                distances[k] += self.graph.node_dist_mat[current_index][index]
                current_index = index
        return distances

    def stochastic_accept(self, index_to_visit, transition_prob):
        """
        轮盘赌
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        max_tran_prob = np.max(transition_prob)
        transition_prob = transition_prob/max_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= transition_prob[ind]:
                return index_to_visit[ind]


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    graph = VPRTW_Graph(file_path)

    vrptw = VRPTW_ACO(graph, ants_num=50, max_iter=300, alpha=1, beta=1, rho=0.1)
    vrptw.run()
