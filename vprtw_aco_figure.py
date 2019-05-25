import matplotlib.pyplot as plt
from vrptw_base import VrptwGraph
import time


class VrptwAcoFigure:
    def __init__(self, graph: VrptwGraph):
        self.graph = graph
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)

        self._dot_color = 'k'
        self._line_color_list = ['r', 'y', 'g', 'c', 'b', 'm']

    def init_figure(self, path):
        self.figure_ax.plot(list(self.graph.nodes[index].x for index in path),
                            list(self.graph.nodes[index].y for index in path), '%s.' % self._dot_color)
        self._draw_line(path)
        self.figure.show()
        time.sleep(0.2)

    def update_figure(self, path):
        for line in self.figure_ax.lines:
            if line._color != self._dot_color:
                self.figure_ax.lines.remove(line)

        self._draw_line(path)

        self.figure.canvas.draw()
        time.sleep(0.2)

    def _draw_line(self, path):
        color_ind = 0
        x_list = []
        y_list = []
        i = 0
        while i < len(path):
            x_list.append(self.graph.nodes[path[i]].x)
            y_list.append(self.graph.nodes[path[i]].y)

            if path[i] == 0 and len(x_list) > 1:
                self.figure_ax.plot(x_list, y_list, '%s-' % self._line_color_list[color_ind])
                color_ind = (color_ind + 1) % len(self._line_color_list)
                x_list.clear()
                y_list.clear()
                i -= 1
            i += 1
