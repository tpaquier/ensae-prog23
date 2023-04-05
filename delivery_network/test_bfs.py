
from graph import Graph, Union_Find, graph_from_file, kruskal, min_power_kruskal_V1, vitesse

data_path = "input/"
file_name = "network.10.in"

g = graph_from_file(data_path + file_name)


#b= g.bfs(1,7)
gk = kruskal(g)
truc = gk.bfs(1, 100000)
a=vitesse(1, 100000, truc)
#c=g.min_power(1, 153789)
#d=g.get_path_with_power(1, 12, 500000000)
print("la réponse finale est : \n", a, "\n, c")




