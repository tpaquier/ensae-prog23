
from graph import Graph, Union_Find, graph_from_file, kruskal, min_power_kruskal_V1, vitesse

data_path = "input/"
file_name = "network.00.in"

g = graph_from_file(data_path + file_name)


#b= g.bfs(1,7)

gk = kruskal(g)
truc = gk.bfs(1, 100000)
a=vitesse(9, 10, truc)

#c=gk.min_power(9,10)
#d=g.get_path_with_power(1, 7, 500000000)
print("la réponse finale est : \n", a)

"""
u=[1,2,3,4,5,6,7,8,9]
for i in range (0,9) :
    print("u is :", u)
    u.reverse()
    n=u.pop()
    if 1 in u : 
        print("trou")
    elif 1 in u or 2 in u :
        print("ouaf")
    elif 6 in u and 8 in u :
        print("pouet")
    elif 6 in u or 8 in u :
        print("trompette")
"""


