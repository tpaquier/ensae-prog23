from graph import Graph, graph_from_file


data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g)

from graph import Graph, Union_Find, graph_from_file, graph_into_pdf, kruskal, min_power_kruskal_V1
import os
import time
import random


def performance_estimation(file):
    g = graph_from_file(file)
    MST = kruskal(g)
    #determine_parents(MST)
    individual_performances = []
    N = 30
    paths = []
    nb_paths = 200000
    for index in range(N):
        paths.append((random.randint(1, MST.nb_nodes), random.randint(1, MST.nb_nodes)))

    for n1, n2 in paths:
        start = time.perf_counter()
        print(min_power_kruskal_V1(MST, n1, n2))
        stop = time.perf_counter()
        individual_performances.append(stop-start)
    print(f'Moyenne de temps de traitement du trajet: {(1/N)*sum(list_of_times)}')
    tps = nb_paths*(1/N)*sum(list_of_times)
    print(f'{tps} secondes')

performance_estimation("/home/onyxia/work/ensae-prog23/input/network.03.in")
