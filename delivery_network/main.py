from graph import Graph, Union_Find, graph_from_file, kruskal, min_power_kruskal_V1, min_power, get_path_with_power, dfs, connected_components, bfs, connected_components_set 
import os
import time
import random

# See below different rudimentary time estimation functions
# They estimate only the actual min_power functions processing time (not e.g. of writing routes.out, etc.)

def performance_estimation_min_power(file, routes):
    g = graph_from_file(file)
    individual_performances = []
    paths = open(routes, "r")
    nb_paths = int(paths.readline().strip())
    for i, line in enumerate(paths):
        if i == 0:
            continue  # Skip the first line
        if i > 30:
            break  # Stop reading after 30 lines
        n1, n2 = map(int, line.split()[:2])
        start = time.perf_counter()
        min_power(g, n1, n2)
        stop = time.perf_counter()
        individual_performances.append(stop - start)
    estimation = sum(individual_performances)
    print("Temps estimé:", nb_paths*(estimation / 30))

    """
    Estimated times with DFS:
    network/routes 1
    network/routes 5
    network/routes 9
    """
    
g = graph_from_file("/home/onyxia/work/ensae-prog23/input/network.01.in")
print(min_power(g, 1, 2))
performance_estimation_min_power("/home/onyxia/work/ensae-prog23/input/network.1.in", "/home/onyxia/work/ensae-prog23/input/routes.1.in")

def performance_estimation_kruskal(file, routes):
    g = graph_from_file(file)
    out_route = open("routes.1.out", "w")
    MST = kruskal(g)
    individual_performances = []
    paths = open(routes, "r")
    nb_paths = int(paths.readline().strip())
    for i, line in enumerate(paths):
        if i == 0:
            continue # Skip the first line
        if i > 30:
            break # Stop reading after 30 lines
        n1, n2 = map(int, line.split())
        start = time.perf_counter()
        output = min_power_kruskal_V1(g, n1, n2)
        stop = time.perf_counter()
        individual_performances.append(stop - start)
        out_route.write(str(output) + "\n")
    out_route.close()
    estimation = sum(individual_performances)
    print("Temps estimé:", nb_paths*(estimation / 30))

    """
    Estimated times with a pre-processed graph using Kruskal:
    network/routes 1
    network/routes 5
    network/routes 9
    """

performance_estimation_kruskal("/home/onyxia/work/ensae-prog23/input/network.2.in", "/home/onyxia/work/ensae-prog23/input/routes.2.in")
