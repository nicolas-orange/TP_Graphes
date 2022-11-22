import numpy as np
import csv
import matplotlib.pyplot as plt
import networkx as nx 

# csvfile=open("./test.csv")
# for i in csvfile:
#    print (i)


def toArray(fichier):
    arr=np.loadtxt(fichier,delimiter=";",dtype=str)
    return arr  



toArray('./test.csv')

# print(arr)
# print("l element 2 de la ligne 7 est:") 
# print(arr[7][2])

#with open("./test.csv", 'r') as parse:
#    sample=list(csv.reader(parse,delimiter=";"))

#arr=np.array(sample)
#print(arr)
arr=toArray('./test.csv')

print(arr)

def construitgraphe(arr):

    nombreelements=len(arr)
    print("nombre d'elements:",nombreelements)

    g = nx.DiGraph()

    for i in range(nombreelements):
        print("intégration de : ",(arr[i][0]),(arr[i][1]),(arr[i][2]))
    #   g.add_node(arr[i][0])
        if (bool(arr[i][2])):
            liste_voisins=(arr[i][2]).split(',')
            for vois in liste_voisins:
                print("traitement de : ",arr[i][0],", ajout de : ",vois)
                g.add_edge(vois,arr[i][0],weight=float(arr[i][1]))
    return g

g=construitgraphe(arr)

# prints out the nature of our digraph: 
print (g)

# prints out the complete set of edges with their attributes
print ("listing des edges en format brut:")
print(g.edges.data())

print ("impression du degré des edges : non pondéré")
print(g.degree( nbunch=None, weight=None))

## le tableau est maintenant constuit, on peut attaquer l'exploitation des données... mais d'abord, un petit tableau ?






pos = nx.spring_layout(g,seed=13,k=1.5)  # positions for all nodes - seed for reproducibility


# nodes

nx.draw_networkx_nodes(g, pos, node_size=500)

# edges : the fllowing two examples need to prepare a liste of large (elarge) and one of small (esmall) edges
# nx.draw_networkx_edges(g, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(g, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
## this one is more straightforward 
nx.draw_networkx_edges(g, pos, width=3)

# node labels
nx.draw_networkx_labels(g, pos )
# edge weight labels
edge_labels = nx.get_edge_attributes(g, "weight")
#edge_labels = nx.draw_networkx_labels(g,pos=nx.spring_layout(g))
nx.draw_networkx_edge_labels(g, pos, edge_labels)

# nx.draw_spring(g,with_labels = True)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()

