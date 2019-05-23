import numpy as np
from vrptw_base import Graph


class VRPTW_ACO:
    def __init__(self, graph: Graph, ants_num=50, max_iter=300, alpha=1, beta=1, rho=0.1):
        super()
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        # 信息素挥发速度
        self.rho = rho
        # q0 表示直接选择概率最大的下一点的概率
        self.q0 = 0.1
        # Q 表示每辆车的最大载重
        self.max_load = 1
        # L 表示每辆车的最远行驶距离
        self.max_travel_distance = 100
        # 出车的单位成本
        self.car_cost = 1
        # 行驶的单位成本
        self.travel_cost = 1
        # 早到时间成本
        self.early_arrival_cost = 1
        # 晚到时间成本
        self.late_arrival_cost = 1
        # 信息素强度
        self.Q = 1
        # 创建信息素矩阵
        self.pheromone_mat = np.ones((graph.node_num, graph.node_num))
        self.eta_mat = 1 / graph.node_dist_mat

    def run(self):
        # 最大迭代次数
        for iter in range(self.max_iter):
            probability = np.zeros((self.graph.node_num, self.graph.node_num))
            vehicle_load = np.zeros(self.graph.node_num)
            travel_distance = np.zeros(self.graph.node_num)

            # 遍历所有的蚂蚁
            ants_path = list([] for _ in range(self.ants_num))
            for k in range(self.ants_num):
                index_to_visit = list(range(graph.node_num))

                # 每只蚂蚁都从0位置出发，最终要回到0位置
                current_index = 0
                ants_path[k] = [current_index]
                index_to_visit.remove(current_index)

                # 蚂蚁需要访问完所有的客户
                while len(index_to_visit) > 0:
                    next_index = self.select_next_index(probability, current_index, index_to_visit)
                    # 判断加入该位置后，是否还满足约束条件, 如果不满足，则再选择一次，然后再进行判断
                    if not self.check_condition(current_index, next_index, vehicle_load[k], travel_distance[k]):
                        next_index = self.select_next_index(probability, current_index, index_to_visit)
                        if not self.check_condition(current_index, next_index, vehicle_load[k], travel_distance[k]):
                            next_index = 0

                    # 更新蚂蚁路径
                    if next_index == 0:
                        ants_path[k].append(next_index)
                        vehicle_load[k] = 0
                        travel_distance[k] = 0
                    else:
                        ants_path[k].append(next_index)
                        index_to_visit.remove(next_index)
                        vehicle_load[k] += self.graph.nodes[next_index].demand
                        travel_distance[k] += self.graph.node_dist_mat[current_index][next_index]

                # 最终回到0位置
                ants_path[k].append(0)

            # 计算所有蚂蚁的路径长度
            paths_distance = self.calculate_all_path_distance(ants_path)

            # 记录当前的最佳路径
            paths_cost = self.calculate_all_path_cost(ants_path)
            best_index = np.argmin(paths_cost)

            # 更新信息素表
            self.update_pheromone_mat(ants_path, paths_distance, )

    def select_next_index(self, probability, current_index, index_to_visit):
        probability[current_index][index_to_visit] = np.power(self.pheromone_mat[current_index][index_to_visit], self.alpha) * \
            np.power(self.eta_mat[current_index][index_to_visit], self.beta)
        if np.random.rand() < self.q0:
            # 这里的index不能这么写
            max_prob_index = np.argmax(probability[current_index][index_to_visit])
            next_index = index_to_visit[max_prob_index]
        else:
            # 使用轮盘度
            prob_sum = np.sum(probability[current_index][index_to_visit])
            next_index = np.random.choice(index_to_visit, probability[current_index][index_to_visit]/prob_sum)
        return next_index

    def check_condition(self, current_index, next_index, vehicle_load, travel_distance) -> bool:
        """
        检查移动到下一个点是否满足约束条件
        :param current_index:
        :param next_index:
        :param vehicle_load:
        :param travel_distance:
        :return:
        """
        if vehicle_load + self.graph.nodes[next_index].demand > self.max_load:
            return False
        if travel_distance + self.graph.node_dist_mat[current_index][next_index] + self.graph.node_dist_mat[next_index][0] > self.max_travel_distance:
            return False
        return True

    def update_pheromone_mat(self, paths, paths_distance):
        """
        更新信息素矩阵
        :return:
        """
        self.pheromone_mat = (1 - self.rho) * self.pheromone_mat
        for k in range(self.graph.node_num):
            current_index = paths[k][0]
            for index in paths[k][1:]:
                self.pheromone_mat[current_index][index] += self.Q / paths_distance[k]

    def calculate_all_path_cost(self, paths):
        # 注意路径是否是从0开始到0结束的
        costs = np.zeros(self.graph.node_num)
        for k in range(self.graph.node_num):
            current_index = paths[k][0]
            for index in paths[k][1:]:
                if index == 0:
                    costs[k] += self.car_cost + self.travel_cost * self.graph.node_dist_mat[current_index][index]
                else:
                    costs[k] += self.travel_cost * self.graph.node_dist_mat[current_index][index]
                    # 这里如何计算是否是早到了，还是迟到了，我还没有记录时间
                    costs[k] += 0
                current_index = index

        return costs

    def calculate_all_path_distance(self, paths):
        distances = np.zeros(self.graph.node_num)
        for k in range(self.graph.node_num):
            current_index = paths[k][0]
            for index in paths[k][1:]:
                distances[k] += self.graph.node_dist_mat[current_index][index]
                current_index = index
        return distances


if __name__ == '__main__':
    file_path = ''
    graph = Graph(file_path)

    vrptw = VRPTW_ACO(graph, ants_num=50, max_iter=300, alpha=1, beta=1, rho=0.1)
    vrptw.run()
