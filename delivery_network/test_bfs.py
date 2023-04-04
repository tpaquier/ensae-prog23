from graph import Graph, Union_Find, graph_from_file, kruskal, min_power_kruskal_V1, vitesse

data_path = "input/"
file_name = "network.4.in"

g = graph_from_file(data_path + file_name)
gk = kruskal(g)
ancetres = gk.bfs(1, 100000)
a=vitesse(9, 10, ancetres)
print("la réponse finale est : \n", a)