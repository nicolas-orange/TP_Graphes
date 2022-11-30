import copy

# import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import DiGraph


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

'''
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
'''


def construitgraphe(myarray):
    nombreelements = len(myarray)
    print("nombre d'éléments:", nombreelements)

    graph: DiGraph = nx.DiGraph()

    for i in range(nombreelements):
        print("intégration de : ", (myarray[i][0]), (myarray[i][1]), (myarray[i][2]))
        graph.add_node(myarray[i][0], weight=float(myarray[i][1]))
        if bool(myarray[i][2]):
            liste_voisins = (myarray[i][2]).split(',')
            for vois in liste_voisins:
                print("traitement de : ", myarray[i][0], ", ajout de : ", vois)
                graph.add_edge(vois, myarray[i][0])
    return graph


G: DiGraph = construitgraphe(arr)

# prints out the nature of our digraph:
print(G)

# prints out the complete set of edges with their attributes
print("listing des edges en format brut:")
print(G.edges.data())

print("impression du degré des noeuds : non pondéré")
print(G.degree(nbunch=None, weight=None))


def triparnbreliens(graphe):
    liste_avec_nbr_noeuds = list(graphe.degree(nbunch=None, weight=None))
    return sorted(liste_avec_nbr_noeuds, key=lambda noeud: noeud[1], reverse=True)  # sort by age


print("Liste des noeuds par nombre d'arcs (degrés) décroissant:")
print(triparnbreliens(G))


def triparpoidstotal(graphe):
    liste_avec_nbr_noeuds = list(graphe.degree(nbunch=None, weight='weight'))
    return sorted(liste_avec_nbr_noeuds, key=lambda noeud: noeud[1], reverse=True)  # sort by degree


print("Liste des noeuds par degré total (arcs pondérés) décroissant:")
print(triparpoidstotal(G))

print("impression des liens descendants de chaque noeud")
# list the neighbors of each node from our digraph, following directions:
print("descendants de : ", [(n, ":", list(G.out_edges(n))) for n in G.nodes()])

print(" impression des liens ascendants de chaque noeud ")
print("ascendants de : ", [(n, ":", list(G.in_edges(n))) for n in G.nodes()])

print(list(G.adjacency()))

print("successeurs de : ", [(n, ":", list(G.successors(n))) for n in G.nodes()])
print("prédécesseurs de: ", [(n, ":", list(G.predecessors(n))) for n in G.nodes()])

print("re-affichage des nodes")
print(G.nodes.data())


# a revoir, boucle mal
def calcul_niveaux(graph):  # function qui prend un graph en entrée et rend une table de noeuds par niveau
    # nx.set_node_attributes(graph, '', 'niveau')
    cherche = []
    for n in range(3):  # on parcourt les différents niveaux
        for nod in graph.nodes():
            # print(nod[0])
            if len(list(graph.predecessors(nod))) == 0 and nod[0] not in cherche:
                nx.set_node_attributes(graph, {nod[0]: {"niveau": n + 1}})
                cherche.append(nod[0])
            elif cherche in list(graph.predecessors(nod)):
                nx.set_node_attributes(graph, {nod[0]: {"niveau": n + 1}})
                cherche.append(nod[0])
    print("fini")
    return graph


calcul_niveaux(G)

print("re-affichage des nodes")
print(G.nodes.data())

print("re-affichage des noms et des poids des nodes")
print(nx.get_node_attributes(G, 'weight'))

# print("calcul des niveaux:", calcul_niveaux(g))

# Compute the degree of every node: degrees
degrees = [len(list(G.neighbors(n))) for n in G.nodes()]

# Print the degrees
print("affichage des degrés des noeud:")
print(degrees)
'''
def sortbynodesload(mygraph):
    for i in range(len(mygraph)):
    return mygraph
'''

# Le tableau est maintenant construit, on peut attaquer l'exploitation des données... mais d'abord, un petit tableau ?


# nodes
# node label and colors definition
labels = {nod: G.nodes[nod]['weight'] for nod in G.nodes}


# colors = [g.nodes[nod]['weight'] for nod in g.nodes]

# pos = nx.spring_layout(g, seed=13, k=1.666)  # positions for all nodes - seed for reproducibility
# pos = nx.nx_pydot.graphviz_layout(g)


# nx.draw_networkx_nodes(g, pos, node_size=500)

# edges : the following two examples need to prepare a liste of large (elarge) and one of small (esmall) edges
# nx.draw_networkx_edges(g, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(g, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
# pos=nx.spring_layout(g, seed=13, k=1.666)

def ajoutlayers(graph):
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            graph.nodes[node]["layer"] = layer
    return graph


ajoutlayers(G)

def ajoutdebutfin(graph):
    graph.add_node('Debut')
    graph.add_edge('Debut',nodnam) for nodnam in list(graph.nodes.keys)

    return graph

# Compute the multipartite_layout using the "layer" node attribute
pos = nx.multipartite_layout(G, subset_key="layer")

# ajoutlayers(G)
# Compute the multipartite_layout using the "layer" node attribute
pos = nx.multipartite_layout(G, subset_key="layer")

print("calcul des layers avec la fonction topological_generations")
print(list(nx.topological_generations(G)))

print("comparaison avec la fonction bfs_layers : ")
print(list(enumerate(nx.bfs_layers(G, ['A', 'B', 'C']))))

# this one is more straightforward

# affiche les noeuds et leur nom
# nx.draw_networkx_nodes(g, with_labels=True, labels=labels, node_color=colors)

# node labels
# nx.draw_networkx_labels(g, pos)

# edge weight labels
# edge_labels = nx.get_edge_attributes(g, "weight")
# edge_labels = nx.draw_networkx_labels(g,pos=nx.spring_layout(g))
# nx.draw_networkx_edge_labels(g, pos, edge_labels)
# nx.draw_networkx_edge_labels(g, edge_labels)

# print("liste des positions: ")
# print(list(pos.items()))
# Décalage des positions des labels des nodes prrrrrr
newpos = copy.deepcopy(pos)

for px in list(newpos):
    # print(px)
    # print(newpos[px])
    # print(px[1][0])
    # print(px[1][1])
    # newpos[px][0] = newpos[px][0] -0.1
    newpos[px][1] = newpos[px][1] + 0.1

print("liste des positions: ")
print(list(pos.items()))

print("type de pos:")
print(type(pos))
print("nouvelle liste de positions: ")
print(list(newpos.items()))

# on dessine les noeuds et les arcs:
nx.draw_networkx_nodes(G, pos=pos)
nx.draw_networkx_edges(G, pos=pos, width=3)
# on dessine les labels :
nx.draw_networkx_labels(G, pos=pos, horizontalalignment='center')

# modification de la position entre deux affichages :

nx.draw_networkx_labels(G, pos=newpos, labels=labels, verticalalignment='bottom')

# nx.draw_networkx_edge_labels(g, pos=pos)


# preparation de l'affichage secondaire ordonné:
# ax = plt.subplots()
# ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()
