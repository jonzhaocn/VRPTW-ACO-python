import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue


class VrptwAcoFigure:
    def __init__(self, nodes: list, path_queue: MPQueue):
        """
        matplotlib绘图计算需要放在主线程，寻找路径的工作建议另外开一个线程，
        当寻找路径的线程找到一个新的path的时候，将path放在path_queue中，图形绘制线程就会自动进行绘制
        queue中存放的path以PathMessage（class）的形式存在
        nodes中存放的结点以Node（class）的形式存在，主要使用到Node.x, Node.y 来获取到结点的坐标

        :param nodes: nodes是各个结点的list，包括depot
        :param path_queue: queue用来存放工作线程计算得到的path，队列中的每一个元素都是一个path，path中存放的是各个结点的id
        """

        self.nodes = nodes
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.path_queue = path_queue
        self._depot_color = 'k'
        self._customer_color = 'steelblue'
        self._line_color = 'darksalmon'

    def _draw_point(self):
        # 画出depot
        self.figure_ax.scatter([self.nodes[0].x], [self.nodes[0].y], c=self._depot_color, label='depot', s=40)

        # 画出customer
        self.figure_ax.scatter(list(node.x for node in self.nodes[1:]),
                               list(node.y for node in self.nodes[1:]), c=self._customer_color, label='customer', s=20)
        plt.pause(0.5)

    def run(self):
        # 先绘制出各个结点
        self._draw_point()
        self.figure.show()

        # 从队列中读取新的path，进行绘制
        while True:
            if not self.path_queue.empty():
                # 取队列中最新的一个path，其他的path丢弃
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()

                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[draw figure]: exit')
                    break

                # 需要先记录要移除的line，不能直接在第一个循环中进行remove，
                # 不然self.figure_ax.lines会在循环的过程中改变，导致部分line无法成功remove
                remove_obj = []
                for line in self.figure_ax.lines:
                    if line._label == 'line':
                        remove_obj.append(line)

                for line in remove_obj:
                    self.figure_ax.lines.remove(line)
                remove_obj.clear()

                # 重新绘制line
                self.figure_ax.set_title('travel distance: %0.2f, number of vehicles: %d ' % (distance, used_vehicle_num))
                self._draw_line(path)
            plt.pause(1)

    def _draw_line(self, path):
        # 根据path中index进行路径的绘制
        for i in range(1, len(path)):
            x_list = [self.nodes[path[i - 1]].x, self.nodes[path[i]].x]
            y_list = [self.nodes[path[i - 1]].y, self.nodes[path[i]].y]
            self.figure_ax.plot(x_list, y_list, color=self._line_color, linewidth=1.5, label='line')
            plt.pause(0.2)
