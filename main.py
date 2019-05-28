from vrptw_base import VrptwGraph
from basic_aco import BasicACO
from multiple_ant_colony_system import MultipleAntColonySystem


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    graph = VrptwGraph(file_path)

    # vrptw = BasicACO(graph)
    # vrptw.run_basic_aco()

    macs = MultipleAntColonySystem(graph)
    macs.run_multiple_ant_colony_system()
