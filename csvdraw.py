import numpy as np
# import csv
import matplotlib.pyplot as plt
import networkx as nx

import pprint


# csvfile=open("./test.csv")
# for i in csvfile:
#    print (i)


def toarray(fichier):
    myarray = np.loadtxt(fichier, delimiter=";", dtype=str)
    return myarray


# toarray('./test.csv')

# print(arr)
# print("l'élément 2 de la ligne 7 est:")
# print(arr[7][2])

# with open("./test.csv", 'r') as parse:
#    sample=list(csv.reader(parse,delimiter=";"))

# arr=np.array(sample)
# print(arr)

arr = toarray('./test.csv')

print(arr)

print("longueur du premier array : ", len(arr))


def flip_and_chop_array(myarray):
    # retournement du tableau par la diagonale, les lignes deviennent les colonnes
    newarray = np.stack(myarray, axis=-1)

    # print(len(newarray)) # longueur du tableau en nbre de lignes
    # print (newarray[3]) # affichage de la derniere ligne pour un tableau à 4 lignes (de 0 à 3)
    # print(newarray[(len(newarray) - 1)])  # affichage de la dernière ligne d'un tableau à N entrée

    # flag = np.any(newarray[(len(newarray)-1)]) # on teste si la derniere ligne est vide !!! ne marche pas !!!
    if all(s == '' for s in (newarray[(len(newarray) - 1)])):  # si la derniere ligne ne contient que des champs vides
        newarray = newarray[:-1]  # on supprime la dernière ligne du tableau
    # if flag:  # si elle est vide

    return newarray  # on rend le tableau


reduced_array = flip_and_chop_array(arr)

print(reduced_array)


def construitgraphe(myarray):
    nombreelements = len(myarray)
    print("nombre d'éléments:", nombreelements)

    g = nx.DiGraph()

    for i in range(nombreelements):
        print("intégration de : ", (myarray[i][0]), (myarray[i][1]), (myarray[i][2]))
        #   g.add_node(arr[i][0])
        if bool(myarray[i][2]):
            liste_voisins = (myarray[i][2]).split(',')
            for vois in liste_voisins:
                print("traitement de : ", myarray[i][0], ", ajout de : ", vois)
                g.add_edge(vois, myarray[i][0], weight=float(myarray[i][1]))
    return g


g = construitgraphe(arr)

# prints out the nature of our digraph:
print(g)

# prints out the complete set of edges with their attributes
print("listing des edges en format brut:")
print(g.edges.data())

print("impression du degré des noeuds : non pondéré")
print(g.degree(nbunch=None, weight=None))


def triparnbreliens(graphe):
    liste_avec_nbr_noeuds = list(graphe.degree(nbunch=None, weight=None))
    return sorted(liste_avec_nbr_noeuds, key=lambda noeud: noeud[1], reverse=True)  # sort by age


print("Liste des noeuds par nombre d'arcs (degrés) décroissant:")
print(triparnbreliens(g))


def triparpoidstotal(graphe):
    liste_avec_nbr_noeuds = list(graphe.degree(nbunch=None, weight='weight'))
    return sorted(liste_avec_nbr_noeuds, key=lambda noeud: noeud[1], reverse=True)  # sort by age


print("Liste des noeuds par degré total (arcs pondérés) décroissant:")
print(triparpoidstotal(g))

print("impression des liens descendants de chaque noeud")
# list the neighbors of each node from our digraph, following directions:
print("descendants de : ", [(n, ":", list(g.out_edges(n))) for n in g.nodes()])

print(" impression des liens ascendants de chaque noeud ")
print("ascendants de : ", [(n, ":", list(g.in_edges(n))) for n in g.nodes()])

print(list(g.adjacency()))

print("successeurs de : ", [(n, ":", list(g.successors(n))) for n in g.nodes()])
print("prédécesseurs de: ", [(n, ":", list(g.predecessors(n))) for n in g.nodes()])

print("re-affichage des nodes")
print(g.nodes.data())


### a revoir, boucle mal
def calcul_niveaux(graph):  # function qui prends un graph en entrée, et rends une table de noeuds par niveau
    n = int(0)  # niveau en cours
    # nx.set_node_attributes(graph, '', 'niveau')
    cherche = []
    for n in range(3):
        for nod in graph.nodes():
            # print(nod[0])
            if len(list(graph.predecessors(nod))) == 0 and nod[0] not in cherche:
                nx.set_node_attributes(graph, {nod[0]: {"niveau": n+1}})
                cherche.append(nod[0])
            elif cherche in list(graph.predecessors(nod)) :
                nx.set_node_attributes(graph, {nod[0]: {"niveau":n+1}})
                cherche.append(nod[0])
    print("fini")
    return graph


calcul_niveaux(g)

print("re-affichage des nodes")
print(g.nodes.data())

# print("calcul des niveaux:", calcul_niveaux(g))

# Compute the degree of every node: degrees
degrees = [len(list(g.neighbors(n))) for n in g.nodes()]

# Print the degrees
print("affichage des degrés des noeud:")
print(degrees)
'''
def sortbynodesload(mygraph):
    for i in range(len(mygraph)):
    return mygraph
'''

# Le tableau est maintenant construit, on peut attaquer l'exploitation des données... mais d'abord, un petit tableau ?


pos = nx.spring_layout(g, seed=13, k=1.666)  # positions for all nodes - seed for reproducibility
# pos = nx.nx_pydot.graphviz_layout(g)

# nodes

# nx.draw_networkx_nodes(g, pos, node_size=500)

# edges : the following two examples need to prepare a liste of large (elarge) and one of small (esmall) edges
# nx.draw_networkx_edges(g, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(g, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
# this one is more straightforward
nx.draw_networkx_edges(g, pos, width=3)
# affiche les noeuds et leur nom
nx.draw_networkx_nodes(g, pos)

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
