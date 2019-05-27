import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, VrptwMessage
from ant import Ant
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy


class VrptwAco:
    def __init__(self, graph: VrptwGraph, ants_num=10, max_iter=200, alpha=1, beta=2):
        super()
        # graph 结点的位置、服务时间信息
        self.graph = graph
        # ants_num 蚂蚁数量
        self.ants_num = ants_num
        # max_iter 最大迭代次数
        self.max_iter = max_iter
        # vehicle_capacity 表示每辆车的最大载重
        self.max_load = graph.vehicle_capacity
        # 信息素强度
        self.Q = 1
        # alpha 信息素信息重要新
        self.alpha = alpha
        # beta 启发性信息重要性
        self.beta = beta
        # q0 表示直接选择概率最大的下一点的概率
        self.q0 = 0.1
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = self.graph.nnh_travel_path.count(0)-1

        self.whether_or_not_to_show_figure = False

        if self.whether_or_not_to_show_figure:
            # figure
            self.figure = VrptwAcoFigure(self.graph)

    def basic_aco(self):
        """
        最基本的蚁群算法
        :return:
        """
        # 最大迭代次数
        for iter in range(self.max_iter):

            # 为每只蚂蚁设置当前车辆负载，当前旅行距离，当前时间
            ants = list(Ant(self.graph) for _ in range(self.ants_num))
            for k in range(self.ants_num):

                # 蚂蚁需要访问完所有的客户
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    # 判断加入该位置后，是否还满足约束条件, 如果不满足，则再选择一次，然后再进行判断
                    if not ants[k].check_condition(next_index):
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0

                    # 更新蚂蚁路径
                    ants[k].move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)

                # 最终回到0位置
                ants[k].move_to_next_index(0)
                self.graph.local_update_pheromone(ants[k].current_index, 0)

            # 计算所有蚂蚁的路径长度
            paths_distance = np.array([ant.total_travel_distance for ant in ants])

            # 记录当前的最佳路径
            best_index = np.argmin(paths_distance)
            if self.best_path is None:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                if self.whether_or_not_to_show_figure:
                    self.figure.init_figure(self.best_path)

            elif paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                if self.whether_or_not_to_show_figure:
                    self.figure.update_figure(self.best_path)

            print('[iteration %d]: best distance %f' % (iter, self.best_path_distance))
            # 更新信息素表
            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

    def select_next_index(self, ant):
        """
        选择下一个结点
        :param ant:
        :return:
        """
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        transition_prob = np.power(self.graph.pheromone_mat[current_index][index_to_visit], self.alpha) * \
            np.power(self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta)

        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            # 使用轮盘赌算法
            next_index = VrptwAco.stochastic_accept(index_to_visit, transition_prob)
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        轮盘赌
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob/sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    @staticmethod
    def new_active_ant(ant: Ant, local_search: bool, IN: np.numarray, q0: float, what_to_do_list: list):
        print('[new_active_ant]: start, start_index %d' % ant.travel_path[0])
        # 计算从当前位置可以达到的下一个位置
        next_index_meet_constrains = ant.cal_next_index_meet_constrains()

        while len(next_index_meet_constrains) > 0:

            if len(what_to_do_list) > 0:
                info = what_to_do_list[0]
                if info.is_to_stop():
                    return

            index_num = len(next_index_meet_constrains)
            ready_time = np.zeros(index_num)
            due_time = np.zeros(index_num)
            for i in range(index_num):
                ready_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].ready_time
                due_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.array([max(i, j) for i, j in zip(ant.vehicle_travel_time + ant.graph.node_dist_mat[ant.current_index][next_index_meet_constrains], ready_time)])

            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.array([max(1.0, j) for j in distance-IN[next_index_meet_constrains]])
            closeness = 1/distance

            # 按照概率选择下一个点next_index
            if np.random.rand() < q0:
                max_prob_index = np.argmax(closeness)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # 使用轮盘赌算法
                next_index = VrptwAco.stochastic_accept(next_index_meet_constrains, closeness)
            ant.move_to_next_index(next_index)

            # 更新信息素矩阵

            # 重新计算可选的下一个点
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

        ant.insertion_procedure(what_to_do_list)

        # ant.index_to_visit_empty()==True就是feasible的意思
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(what_to_do_list)

        return ant.get_path_without_duplicated_depot()

    @staticmethod
    def acs_time(new_graph: VrptwGraph, vehicle_num, ants_num, q0, what_to_do: Queue, what_is_found: Queue):
        print('[acs_time]: start, vehicle_num %d' % vehicle_num)
        # 初始化信息素矩阵
        global_best_path = []
        global_best_distance = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        ants_what_to_do_list = []
        while True:
            for k in range(ants_num):
                ant = Ant(new_graph, random.randint(0, vehicle_num - 1))
                what_to_do_list = []
                thread = ants_pool.submit(VrptwAco.new_active_ant, ant, True, np.zeros(new_graph.node_num), q0, what_to_do_list)

                ants_thread.append(thread)
                ants.append(ant)
                ants_what_to_do_list.append(what_to_do_list)

            # 这里可以使用result方法，等待线程跑完
            for thread in ants_thread:

                while not what_to_do.empty():
                    info = what_to_do.get()
                    if info.is_to_stop():
                        print('[acs_time]: receive stop info')
                        # 顺便要让蚂蚁线程都结束
                        for what_to_do_list in ants_what_to_do_list:
                            what_to_do_list.append(VrptwMessage('stop', None, None))
                        ants_pool.shutdown()
                        return
                    else:
                        print('[acs_time]: receive global path info')
                        global_best_path, global_best_distance = info.get_path_info()

                thread.result()

            # 判断蚂蚁找出来的路径是否是feasible的，并且比全局的路径要好
            for ant in ants:
                # 如果比全局的路径要好，则要将该路径发送到macs中
                while not what_to_do.empty():
                    info = what_to_do.get()
                    if info.is_to_stop():
                        print('[acs_time]: receive stop info')
                        # 顺便要让蚂蚁线程都结束
                        for what_to_do_list in ants_what_to_do_list:
                            what_to_do_list.append(VrptwMessage('stop', None, None))
                        ants_pool.shutdown()
                        return
                    else:
                        print('[acs_time]: receive global path info')
                        global_best_path, global_best_distance = info.get_path_info()

                if ant.index_to_visit_empty() and ant.total_travel_distance < global_best_distance:
                    print('[acs_time]: found a improved feasible path, send path info to macs')
                    what_is_found.put(VrptwMessage('path_info', ant.travel_path, ant.total_travel_distance))

            # 在这里执行信息素的全局更新
            new_graph.global_update_pheromone(global_best_path, global_best_distance)

    @staticmethod
    def acs_vehicle(new_graph: VrptwGraph, vehicle_num, ants_num, q0, what_to_do: Queue, what_is_found: Queue):
        print('[acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = []
        global_best_distance = None

        # 使用邻近点算法初始化path 和distance
        current_path, current_path_distance = new_graph.nearest_neighbor_heuristic()

        # 找出当前path中未访问的结点
        current_index_to_visit = list(range(new_graph.node_num))
        for ind in set(current_path):
            current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        ants_what_to_do_list = []
        IN = np.zeros(new_graph.node_num)
        while True:
            for k in range(ants_num):
                ant = Ant(new_graph, random.randint(0, vehicle_num-1))
                what_to_do_list = []
                thread = ants_pool.submit(VrptwAco.new_active_ant, ant, True, IN, q0, what_to_do_list)

                ants_thread.append(thread)
                ants.append(ant)
                ants_what_to_do_list.append(what_to_do_list)

            # 这里可以使用result方法，等待线程跑完
            for thread in ants_thread:

                while not what_to_do.empty():
                    info = what_to_do.get()
                    if info.is_to_stop():
                        # 顺便要让蚂蚁线程都结束
                        print('[acs_vehicle]: receive stop info')
                        for what_to_do_list in ants_what_to_do_list:
                            what_to_do_list.append(VrptwMessage('stop', None, None))
                        ants_pool.shutdown()
                        return
                    else:
                        print('[acs_vehicle]: receive global path info')
                        global_best_path, global_best_distance = info.get_path_info()

                thread.result()

            for ant in ants:
                index_to_visit = ant.index_to_visit
                IN[index_to_visit] = IN[index_to_visit]+1
                path = ant.get_path_without_duplicated_depot()

                # 判断蚂蚁找出来的路径是否比current_path，能使用vehicle_num辆车访问到更多的结点
                if len(index_to_visit) < len(current_index_to_visit):
                    current_path = path
                    current_index_to_visit = index_to_visit
                    current_path_distance = ant.total_travel_distance
                    # 并且将IN设置为0
                    IN = np.zeros(new_graph.node_num)

                    # 如果这一条路径是feasible的话，就要发到macs_vrptw中
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: found a feasible path, send path info to macs')
                        what_is_found.put(VrptwMessage('path_info', ant.travel_path, ant.total_travel_distance))

            # 更新new_graph中的信息素，global
            new_graph.global_update_pheromone(current_path, current_path_distance)

            while not what_to_do.empty():
                info = what_to_do.get()
                if info.is_to_stop():
                    # 顺便要让蚂蚁线程都结束
                    print('[acs_vehicle]: receive stop info')
                    for what_to_do_list in ants_what_to_do_list:
                        what_to_do_list.append(VrptwMessage('stop', None, None))
                    ants_pool.shutdown()
                    return
                else:
                    print('[acs_vehicle]: receive global path info')
                    global_best_path, global_best_distance = info.get_path_info()

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

    def multiple_ant_colony_system(self):
        # 在这里需要两个队列，一个队列是macs告诉acs_time和acs_vehicle这两个线程，当前的best path是什么、告诉他们停止计算
        # 另外的一个线程就是用来，接收他们两个计算出来的比best path还要好的feasible path
        time_what_to_do = Queue()
        vehicle_what_to_do = Queue()
        what_is_found = Queue()

        while True:
            # acs_vehicle
            graph_for_acs_vehicle = self.graph.construct_graph_with_duplicated_depot(self.best_vehicle_num-1,
                                                                                     self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(target=self.acs_vehicle, args=(graph_for_acs_vehicle, self.best_vehicle_num-1,
                                                                       self.ants_num, self.q0, vehicle_what_to_do,
                                                                       what_is_found))

            # acs_time
            graph_for_acs_time = self.graph.construct_graph_with_duplicated_depot(self.best_vehicle_num,
                                                                                  self.graph.init_pheromone_val)
            acs_time_thread = Thread(target=self.acs_time, args=(graph_for_acs_time, self.best_vehicle_num,
                                                                 self.ants_num, self.q0, time_what_to_do, what_is_found))

            # 启动acs_vehicle_thread和acs_time_thread，当他们找到feasible、且是比best path好的路径时，就会发送到macs中来
            print('[macs]: start acs_vehicle and acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():
                path_info = what_is_found.get()
                print('[macs]: receive found path info')
                found_path, found_path_distance = path_info.get_path_info()
                # 如果，这两个线程找到的路径用的车辆更少了，就停止这两个线程，开始下一轮迭代
                # 向acs_time和acs_vehicle中发送停止信息

                if found_path_distance < self.best_path_distance:
                    print('[macs]: distance of found path better than best path\'s')
                    self.best_path = found_path
                    self.best_vehicle_num = found_path.count(0) - 1
                    self.best_path_distance = 0

                if found_path.count(0)-1 < best_vehicle_num:
                    print('[macs]: vehicle num of found path better than best path\'s')
                    self.best_path = found_path
                    self.best_vehicle_num = found_path.count(0)-1
                    self.best_path_distance = 0

                    # 停止acs_time 和 acs_vehicle 两个线程
                    print('[macs]: send stop info to acs_time and acs_vehicle')
                    time_what_to_do.put(VrptwMessage(info_type='stop', path=None, distance=None))
                    vehicle_what_to_do.put(VrptwMessage(info_type='stop', path=None, distance=None))


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    graph = VrptwGraph(file_path)

    vrptw = VrptwAco(graph)
    vrptw.multiple_ant_colony_system()
