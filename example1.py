from vrptw_base import VrptwGraph
from multiple_ant_colony_system import MultipleAntColonySystem


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    graph = VrptwGraph(file_path)

    macs = MultipleAntColonySystem(graph)
    macs.run_multiple_ant_colony_system()
