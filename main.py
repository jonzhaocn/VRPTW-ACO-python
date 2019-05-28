import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread, Event
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
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = True

    def run_basic_aco(self):
        # 开启一个线程来跑_basic_aco，使用主线程来绘图
        path_queue_for_figure = Queue()
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()

        # 是否要展示figure
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph, path_queue_for_figure)
            figure.run()
        basic_aco_thread.join()

        # 传入None作为结束标志
        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _basic_aco(self, path_queue_for_figure: Queue):
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
                    path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

            elif paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                if self.whether_or_not_to_show_figure:
                    path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

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
    def new_active_ant(ant: Ant, vehicle_num: int, local_search: bool, IN: np.numarray, q0: float, stop_event: Event):
        # print('[new_active_ant]: start, start_index %d' % ant.travel_path[0])

        # 在new_active_ant中，最多可以使用vehicle_num个车，即最多可以包含vehicle_num+1个depot结点，由于出发结点用掉了一个，所以只剩下vehicle个depot
        unused_depot_count = vehicle_num

        # 如果还有未访问的结点，并且还可以回到depot中
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: receive stop event')
                return

            # 计算所有满足载重等限制的下一个结点
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            # 如果没有满足限制的下一个结点，则回到depot中
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # 开始计算满足限制的下一个结点，选择各个结点的概率
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

            # 按照概率直接选择closeness最大的结点
            if np.random.rand() < q0:
                max_prob_index = np.argmax(closeness)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # 使用轮盘赌算法
                next_index = VrptwAco.stochastic_accept(next_index_meet_constrains, closeness)

            # 更新信息素矩阵
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # 如果走完所有的点了，需要回到depot
        if ant.index_to_visit_empty():
            ant.move_to_next_index(0)

        # 对未访问的点进行插入，保证path是可行的
        ant.insertion_procedure(stop_event)

        # ant.index_to_visit_empty()==True就是feasible的意思
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    @staticmethod
    def acs_time(new_graph: VrptwGraph, vehicle_num, ants_num, q0, global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):
        # 最多可以使用vehicle_num辆车，即在path中最多包含vehicle_num+1个depot中，找到路程最短的路径，
        # vehicle_num设置为与当前的best_path一致
        print('[acs_time]: start, vehicle_num %d' % vehicle_num)
        # 初始化信息素矩阵
        global_best_path = None
        global_best_distance = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:

            if stop_event.is_set():
                print('[acs_time]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(VrptwAco.new_active_ant, ant, vehicle_num, True, np.zeros(new_graph.node_num), q0, stop_event)

                ants_thread.append(thread)
                ants.append(ant)

            # 这里可以使用result方法，等待线程跑完
            for thread in ants_thread:
                thread.result()

            # 判断蚂蚁找出来的路径是否是feasible的，并且比全局的路径要好
            for ant in ants:

                if stop_event.is_set():
                    print('[acs_time]: receive stop event')
                    return

                # 如果比全局的路径要好，则要将该路径发送到macs中
                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    print('[acs_time]: receive global path info')
                    global_best_path, global_best_distance = info.get_path_info()

                if ant.index_to_visit_empty() and ant.total_travel_distance < global_best_distance:
                    print('[acs_time]: found a improved feasible path, send path info to macs')
                    path_found_queue.put(PathMessage(ant.travel_path, ant.total_travel_distance))

            # 在这里执行信息素的全局更新
            new_graph.global_update_pheromone(global_best_path, global_best_distance)

    @staticmethod
    def acs_vehicle(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):
        # 最多可以使用vehicle_num辆车，即在path中最多包含vehicle_num+1个depot中，找到路程最短的路径，
        # vehicle_num设置为比当前的best_path少一个
        print('[acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
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
        IN = np.zeros(new_graph.node_num)
        while True:

            if stop_event.is_set():
                print('[acs_vehicle]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(VrptwAco.new_active_ant, ant, vehicle_num, True, IN, q0, stop_event)

                ants_thread.append(thread)
                ants.append(ant)

            # 这里可以使用result方法，等待线程跑完
            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_vehicle]: receive stop event')
                    return

                index_to_visit = copy.deepcopy(ant.index_to_visit)
                IN[index_to_visit] = IN[index_to_visit]+1

                # 判断蚂蚁找出来的路径是否比current_path，能使用vehicle_num辆车访问到更多的结点
                if len(index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = index_to_visit
                    current_path_distance = ant.total_travel_distance
                    # 并且将IN设置为0
                    IN = np.zeros(new_graph.node_num)

                    # 如果这一条路径是feasible的话，就要发到macs_vrptw中
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: found a feasible path, send path info to macs')
                        path_found_queue.put(PathMessage(ant.travel_path, ant.total_travel_distance))

            # 更新new_graph中的信息素，global
            new_graph.global_update_pheromone(current_path, current_path_distance)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: receive global path info')
                global_best_path, global_best_distance = info.get_path_info()

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

    def run_multiple_ant_colony_system(self):
        # _multiple_ant_colony_system，使用主线程来绘图
        path_queue_for_figure = Queue()
        multiple_ant_colony_system_thread = Thread(target=self._multiple_ant_colony_system, args=(path_queue_for_figure,))
        multiple_ant_colony_system_thread.start()

        # 是否要展示figure
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph, path_queue_for_figure)
            figure.run()
        multiple_ant_colony_system_thread.join()

        # 传入None作为结束标志
        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _multiple_ant_colony_system(self, path_queue_for_figure: Queue):
        # 在这里需要两个队列，time_what_to_do、vehicle_what_to_do， 用来告诉acs_time、acs_vehicle这两个线程，当前的best path是什么，或者让他们停止计算
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # 另外的一个队列， path_found_queue就是接收acs_time 和acs_vehicle计算出来的比best path还要好的feasible path
        path_found_queue = Queue()

        # 使用近邻点算法初始化
        self.best_path, self.best_path_distance = self.graph.nearest_neighbor_heuristic()
        self.best_vehicle_num = self.best_path.count(0) - 1

        while True:
            # 当前best path的信息
            global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
            global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

            # acs_vehicle
            stop_event = Event()
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(target=VrptwAco.acs_vehicle,
                                        args=(graph_for_acs_vehicle, self.best_vehicle_num-1, self.ants_num, self.q0,
                                              global_path_to_acs_vehicle, path_found_queue, stop_event))

            # acs_time
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)
            acs_time_thread = Thread(target=VrptwAco.acs_time,
                                     args=(graph_for_acs_time, self.best_vehicle_num, self.ants_num, self.q0,
                                           global_path_to_acs_time, path_found_queue, stop_event))

            # 启动acs_vehicle_thread和acs_time_thread，当他们找到feasible、且是比best path好的路径时，就会发送到macs中来
            print('[macs]: start acs_vehicle and acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                print('[macs]: receive found path info')
                found_path, found_path_distance = path_info.get_path_info()

                # 如果找到的路径（feasible）的距离更短，则更新当前的最佳path的信息
                if found_path_distance < self.best_path_distance:
                    print('-' * 50)
                    print('[macs]: distance of found path (%f) better than best path\'s (%f)' % (found_path_distance, self.best_path_distance))
                    print('-' * 50)
                    self.best_path = found_path
                    self.best_vehicle_num = found_path.count(0) - 1
                    self.best_path_distance = found_path_distance

                    # 如果需要绘制图形，则要找到的best path发送给绘图程序
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                    # 通知acs_vehicle和acs_time两个线程，当前找到的best_path和best_path_distance
                    global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
                    global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

                # 如果，这两个线程找到的路径用的车辆更少了，就停止这两个线程，开始下一轮迭代
                # 向acs_time和acs_vehicle中发送停止信息
                if found_path.count(0)-1 < best_vehicle_num:
                    print('-' * 50)
                    print('[macs]: vehicle num of found path (%d) better than best path\'s (%d)' % (found_path.count(0)-1, best_vehicle_num))
                    print('-' * 50)
                    self.best_path = found_path
                    self.best_vehicle_num = found_path.count(0)-1
                    self.best_path_distance = found_path_distance

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                    # 停止acs_time 和 acs_vehicle 两个线程
                    print('[macs]: send stop info to acs_time and acs_vehicle')
                    stop_event.set()


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    graph = VrptwGraph(file_path)

    vrptw = VrptwAco(graph)
    vrptw.run_multiple_ant_colony_system()
