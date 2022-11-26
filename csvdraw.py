import numpy as np
# import csv
import matplotlib.pyplot as plt
import networkx as nx


# csvfile=open("./test.csv")
# for i in csvfile:
#    print (i)


def toarray(fichier):
    myarray = np.loadtxt(fichier, delimiter=";", dtype=str)
    return myarray


toarray('./test.csv')

# print(arr)
# print("l'élément 2 de la ligne 7 est:")
# print(arr[7][2])

# with open("./test.csv", 'r') as parse:
#    sample=list(csv.reader(parse,delimiter=";"))

# arr=np.array(sample)
# print(arr)

arr = toarray('./test.csv')

print(arr)


def construitgraphe(array):
    nombreelements = len(array)
    print("nombre d'éléments:", nombreelements)

    g = nx.DiGraph()

    for i in range(nombreelements):
        print("intégration de : ", (array[i][0]), (array[i][1]), (array[i][2]))
        #   g.add_node(arr[i][0])
        if bool(array[i][2]):
            liste_voisins = (array[i][2]).split(',')
            for vois in liste_voisins:
                print("traitement de : ", array[i][0], ", ajout de : ", vois)
                g.add_edge(vois, array[i][0], weight=float(array[i][1]))
    return g


g = construitgraphe(arr)

# prints out the nature of our digraph: 
print(g)

# prints out the complete set of edges with their attributes
print("listing des edges en format brut:")
print(g.edges.data())

print("impression du degré des edges : non pondéré")
print(g.degree(nbunch=None, weight=None))

# Le tableau est maintenant construit, on peut attaquer l'exploitation des données... mais d'abord, un petit tableau ?


pos = nx.spring_layout(g, seed=13, k=1.5)  # positions for all nodes - seed for reproducibility

# nodes

nx.draw_networkx_nodes(g, pos, node_size=500)

# edges : the following two examples need to prepare a liste of large (elarge) and one of small (esmall) edges
# nx.draw_networkx_edges(g, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(g, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
# this one is more straightforward
nx.draw_networkx_edges(g, pos, width=3)

# node labels
nx.draw_networkx_labels(g, pos)
# edge weight labels
edge_labels = nx.get_edge_attributes(g, "weight")
# edge_labels = nx.draw_networkx_labels(g,pos=nx.spring_layout(g))
nx.draw_networkx_edge_labels(g, pos, edge_labels)

# nx.draw_spring(g,with_labels = True)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()
