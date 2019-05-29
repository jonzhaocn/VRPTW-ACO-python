from vrptw_base import VrptwGraph
from basic_aco import BasicACO


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    graph = VrptwGraph(file_path)

    vrptw = BasicACO(graph)
    vrptw.run_basic_aco()