from vrptw_base import VrptwGraph
from multiple_ant_colony_system import MultipleAntColonySystem
import os


if __name__ == '__main__':
    ants_num = 10
    beta = 1
    q0 = 0.1
    show_figure = False
    for file_name in os.listdir('./solomon-100'):
        file_path = os.path.join('./solomon-100', file_name)
        print('-' * 100)
        print('file_path: %s' % file_path)
        print('\n')
        file_to_write_path = os.path.join('./result', file_name.split('.')[0] + '-result.txt')
        graph = VrptwGraph(file_path)
        macs = MultipleAntColonySystem(graph, ants_num=ants_num, beta=beta, q0=q0, whether_or_not_to_show_figure=show_figure)
        macs.run_multiple_ant_colony_system(file_to_write_path)
        print('\n' * 10)
