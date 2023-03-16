from graph import Graph, graph_from_file, min_power_kruskal_V1, kruskal, Union_Find

data_path = "input/"
file_name = "network.9.in"

g = graph_from_file(data_path + file_name)
print(min_power_kruskal_V1(g, 1, 50))