from graph import Graph, Union_Find, graph_from_file, kruskal, min_power_kruskal_V1, time_perf_min_power
import os
import time
import random



g = graph_from_file('input/network.1.in')
gk = kruskal(g)
'''
min_power = gk.min_power(6,11)
print(gk.get_path_with_power(6, 11, min_power[1]))
'''

time_perf_min_power(2)