import numpy as np
import copy
from vrptw_base import VrptwGraph
from threading import Event


class Ant:
    def __init__(self, graph: VrptwGraph, start_index=0):
        super()
        self.graph = graph
        self.current_index = start_index
        self.vehicle_load = 0
        self.vehicle_travel_time = 0
        self.travel_path = [start_index]
        self.arrival_time = [0]

        self.index_to_visit = list(range(graph.node_num))
        self.index_to_visit.remove(start_index)

        self.total_travel_distance = 0

    def clear(self):
        self.travel_path.clear()
        self.index_to_visit.clear()

    def move_to_next_index(self, next_index):
        # 更新蚂蚁路径
        self.travel_path.append(next_index)
        self.total_travel_distance += self.graph.node_dist_mat[self.current_index][next_index]

        dist = self.graph.node_dist_mat[self.current_index][next_index]
        self.arrival_time.append(self.vehicle_travel_time + dist)

        if self.graph.nodes[next_index].is_depot:
            # 如果一下个位置为服务器点，则要将车辆负载等清空
            self.vehicle_load = 0
            self.vehicle_travel_time = 0

            # remove duplicated_depot
            if next_index in self.index_to_visit:
                self.index_to_visit.remove(next_index)
        else:
            # 更新车辆负载、行驶距离、时间
            self.vehicle_load += self.graph.nodes[next_index].demand
            # 如果早于客户要求的时间窗(ready_time)，则需要等待

            self.vehicle_travel_time += dist + max(self.graph.nodes[next_index].ready_time - self.vehicle_travel_time - dist, 0) + self.graph.nodes[next_index].service_time
            self.index_to_visit.remove(next_index)

        self.current_index = next_index

    def index_to_visit_empty(self):
        return len(self.index_to_visit) == 0

    def get_active_vehicles_num(self):
        return self.travel_path.count(0)-1

    def check_condition(self, next_index) -> bool:
        """
        检查移动到下一个点是否满足约束条件
        :param next_index:
        :return:
        """
        if self.vehicle_load + self.graph.nodes[next_index].demand > self.graph.vehicle_capacity:
            return False

        dist = self.graph.node_dist_mat[self.current_index][next_index]
        wait_time = max(self.graph.nodes[next_index].ready_time - self.vehicle_travel_time - dist, 0)
        service_time = self.graph.nodes[next_index].service_time

        # 检查访问某一个旅客之后，能否回到服务店
        if self.vehicle_travel_time + dist + wait_time + service_time + self.graph.node_dist_mat[next_index][0] > self.graph.nodes[0].due_time:
            return False

        # 不可以服务due time之外的旅客
        if self.vehicle_travel_time + dist > self.graph.nodes[next_index].due_time:
            return False

        return True

    def cal_next_index_meet_constrains(self):
        """
        找出所有从当前位置（ant.current_index）可达的customer
        :return:
        """
        next_index_meet_constrains = []
        for next_ind in self.index_to_visit:
            if self.check_condition(next_ind):
                next_index_meet_constrains.append(next_ind)
        return next_index_meet_constrains

    def cal_nearest_next_index(self, next_index_list):
        """
        从待选的customers中选择，离当前位置（ant.current_index）最近的customer

        :param next_index_list:
        :return:
        """
        current_ind = self.current_index

        nearest_ind = next_index_list[0]
        min_dist = self.graph.node_dist_mat[current_ind][next_index_list[0]]

        for next_ind in next_index_list[1:]:
            dist = self.graph.node_dist_mat[current_ind][next_ind]
            if dist < min_dist:
                min_dist = dist
                nearest_ind = next_ind

        return nearest_ind

    def cal_total_travel_distance(self, travel_path):
        distance = 0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance += self.graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance

    def try_insert_on_path(self, node_id, stop_event: Event):
        """
        尝试性地将node_id插入当前的travel_path中
        插入的位置不能违反载重，时间，行驶距离的限制
        如果有多个位置，则找出最优的位置
        :param node_id:
        :return:
        """
        feasible_insert_index = []
        feasible_distance = []

        for insert_index in range(len(self.travel_path)):

            if stop_event.is_set():
                # print('[try_insert_on_path]: receive stop event')
                return

            if self.graph.nodes[self.travel_path[insert_index]].is_depot:
                continue

            front_depot_index = insert_index
            while front_depot_index >= 0 and not self.graph.nodes[self.travel_path[front_depot_index]].is_depot:
                front_depot_index -= 1
            front_depot_index = max(front_depot_index, 0)

            check_ant = Ant(self.graph, self.travel_path[front_depot_index])

            # 让check_ant 走过 path中下标从front_depot_index开始到insert_index-1的点
            for i in range(front_depot_index+1, insert_index):
                check_ant.move_to_next_index(self.travel_path[i])

            # 开始尝试性地对排序后的index_to_visit中的结点进行访问
            if check_ant.check_condition(node_id):
                check_ant.move_to_next_index(node_id)
            else:
                continue

            # 如果可以到node_id，则要保证vehicle可以行驶回到depot
            for next_ind in self.travel_path[insert_index:]:

                if stop_event.is_set():
                    # print('[try_insert_on_path]: receive stop event')
                    return

                if check_ant.check_condition(next_ind):

                    check_ant.move_to_next_index(next_ind)

                    # 如果回到了depot
                    if self.graph.nodes[next_ind].is_depot:
                        feasible_insert_index.append(insert_index)
                        # 计算距离
                        path = copy.deepcopy(self.travel_path)
                        path.insert(insert_index, node_id)
                        feasible_distance.append(self.cal_total_travel_distance(path))
                        break

                # 如果不可以回到depot，则返回上一层
                else:
                    break

        if len(feasible_distance) == 0:
            return None
        else:
            feasible_distance = np.array(feasible_distance)
            min_insert_ind = np.argmin(feasible_distance)
            best_ind = feasible_insert_index[int(min_insert_ind)]
            return best_ind

    def insertion_procedure(self, stop_even: Event):
        """
        为每个未访问的结点尝试性地找到一个合适的位置，插入到当前的travel_path
        插入的位置不能违反载重，时间，行驶距离的限制
        :return:
        """
        if self.index_to_visit_empty():
            return

        ind_to_visit = np.array(copy.deepcopy(self.index_to_visit))

        demand = np.zeros(len(ind_to_visit))
        for i in range(len(ind_to_visit)):
            demand[i] = self.graph.nodes[i].demand

        sorted_ind = np.argsort(demand)
        ind_to_visit = ind_to_visit[sorted_ind]

        for node_id in ind_to_visit:

            if stop_even.is_set():
                # print('[insertion_procedure]: receive stop event')
                return

            best_insert_index = self.try_insert_on_path(node_id, stop_even)
            if best_insert_index is not None:
                self.travel_path.insert(best_insert_index, node_id)
                self.index_to_visit.remove(node_id)

        self.total_travel_distance = self.cal_total_travel_distance(self.travel_path)

    def local_search_procedure(self, stop_event: Event):
        """
        对当前的已经访问完graph中所有节点的travel_path使用cross进行局部搜索
        :return:
        """
        # 找出path中所有的depot的位置
        depot_ind = []
        for ind in range(len(self.travel_path)):
            if self.graph.nodes[self.travel_path[ind]].is_depot:
                depot_ind.append(ind)

        new_path_travel_distance = self.total_travel_distance
        new_path = self.travel_path

        # 将self.travel_path分成多段，每段以depot开始，以depot结束，称为route
        for i in range(1, len(depot_ind)):
            for j in range(i+1, len(depot_ind)):

                if stop_event.is_set():
                    return

                position = []
                for start_a in range(depot_ind[i-1]+1, depot_ind[i]):
                    for end_a in range(start_a, min(depot_ind[i], start_a+7)):
                        for start_b in range(depot_ind[j-1]+1, depot_ind[j]):
                            for end_b in range(start_b, min(depot_ind[j], start_b+7)):
                                position.append([start_a, end_a, start_b, end_b])

                for posi in position:
                    start_a, end_a, start_b, end_b = posi
                    path = []
                    path.extend(self.travel_path[:start_a])
                    path.extend(self.travel_path[start_b:end_b+1])
                    path.extend(self.travel_path[end_a:start_b])
                    path.extend(self.travel_path[start_a:end_a])
                    path.extend(self.travel_path[end_b+1:])

                    for k in range(1, len(path)):
                        if path[i-1] == 0 and path[i] == 0:
                            path.remove(i)
                            break

                    depot_before_start_a = start_a
                    while not self.graph.nodes[path[depot_before_start_a]].is_depot:
                        depot_before_start_a -= 1

                    depot_before_start_b = start_b
                    while not self.graph.nodes[path[depot_before_start_b]].is_depot:
                        depot_before_start_b -= 1

                    # 判断发生改变的route a是否是feasible的
                    success_route_a = False
                    check_ant = Ant(self.graph, path[depot_before_start_a])
                    for ind in path[depot_before_start_a+1:]:
                        if check_ant.check_condition(ind):
                            check_ant.move_to_next_index(ind)
                            if self.graph.nodes[ind].is_depot:
                                success_route_a = True
                                break
                        else:
                            break
                    check_ant.clear()
                    del check_ant

                    # 判断发生改变的route b是否是feasible的
                    success_route_b = False
                    check_ant = Ant(self.graph, path[depot_before_start_b])
                    for ind in path[depot_before_start_b + 1:]:
                        if check_ant.check_condition(ind):
                            check_ant.move_to_next_index(ind)
                            if self.graph.nodes[ind].is_depot:
                                success_route_b = True
                                break
                        else:
                            break
                    check_ant.clear()
                    del check_ant

                    if success_route_a and success_route_b:
                        total_travel_distance = self.cal_total_travel_distance(path)
                        if total_travel_distance < new_path_travel_distance:
                            # print('success to search')
                            new_path_travel_distance = total_travel_distance
                            new_path = path
                            print('[local_search_procedure]: found a path in local search, its distance is %f' % new_path_travel_distance)
                    else:
                        path.clear()

        # 找出新生成的path中，路程最小的
        self.travel_path = new_path
        self.total_travel_distance = new_path_travel_distance
        print('[local_search_procedure]: local search finished')