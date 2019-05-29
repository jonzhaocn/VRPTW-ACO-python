import matplotlib.pyplot as plt
from queue import Queue


class VrptwAcoFigure:
    def __init__(self, nodes: list, path_queue: Queue):
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
        self._dot_color = 'k'
        self._line_color_list = ['r', 'y', 'g', 'c', 'b', 'm']

    def _draw_point(self):
        # 先画出图中的点
        self.figure_ax.plot(list(node.x for node in self.nodes),
                            list(node.y for node in self.nodes), '%s.' % self._dot_color)
        self.figure.show()
        plt.pause(0.5)

    def run(self):
        self._draw_point()

        # 从队列中读取新的path，进行绘制
        while True:
            if not self.path_queue.empty():
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()
                path, distance, used_vehicle_num = info.get_path_info()

                if path is None:
                    break

                self.figure_ax.clear()
                self._draw_point()
                self.figure.canvas.draw()
                plt.pause(1)

                self._draw_line(path)
            plt.pause(1)

    def _draw_line(self, path):
        # 根据path中index进行路劲的绘制
        for i in range(1, len(path)):
            x_list = [self.nodes[path[i - 1]].x, self.nodes[path[i]].x]
            y_list = [self.nodes[path[i - 1]].y, self.nodes[path[i]].y]
            self.figure_ax.plot(x_list, y_list, '%s-' % self._line_color_list[0])
            plt.pause(0.05)
