import matplotlib.pyplot as plt
from vrptw_base import VrptwGraph
from queue import Queue


class VrptwAcoFigure:
    def __init__(self, graph: VrptwGraph, path_queue: Queue):
        self.graph = graph
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.path_queue = path_queue
        self._dot_color = 'k'
        self._line_color_list = ['r', 'y', 'g', 'c', 'b', 'm']

    def _draw_point(self):
        # 先画出图中的点
        self.figure_ax.plot(list(node.x for node in self.graph.nodes),
                            list(node.y for node in self.graph.nodes), '%s.' % self._dot_color)
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
                path, distance = info.get_path_info()

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
            x_list = [self.graph.nodes[path[i - 1]].x, self.graph.nodes[path[i]].x]
            y_list = [self.graph.nodes[path[i - 1]].y, self.graph.nodes[path[i]].y]
            self.figure_ax.plot(x_list, y_list, '%s-' % self._line_color_list[0])
            plt.pause(0.05)
