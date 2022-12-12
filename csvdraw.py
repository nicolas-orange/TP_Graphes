import copy
# from typing import Dict

import numpy as np
# import csv
import matplotlib.pyplot as plt
import networkx as nx


# import array

# import pprint


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

# TODO : demander le nom du fichier à parser
#read string from user
fichier = input('Entrez le fichier à parser : (default test.csv)')
if 'csv' not in fichier:
  fichier='test.csv'

arr = toarray(fichier)


# print(arr)

# print("longueur du premier array : ", len(arr))


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


# reduced_array = flip_and_chop_array(arr)

# print(reduced_array)


def construitgraphe(myarray):
  nombreelements = len(myarray)
  print("nombre d'éléments:", nombreelements)

  for i in range(nombreelements):
    # print("intégration de : ", (myarray[i][0]), (myarray[i][1]), (myarray[i][2]))
    g.add_node(myarray[i][0], weight=float(myarray[i][1]))
    if bool(myarray[i][2]):
      liste_voisins = (myarray[i][2]).split(',')
      for vois in liste_voisins:
        # print("traitement de : ", myarray[i][0], ", ajout de : ", vois)
        g.add_edge(vois, myarray[i][0])
  return g

g = nx.DiGraph()
g = construitgraphe(arr)


# prints out the nature of our digraph:
# print(g)

# prints out the complete set of edges with their attributes
# print("listing des edges en format brut:", g.edges.data())

# print("impression du degré des noeuds : non pondéré", g.degree(nbunch=None, weight=None))


def triparnbreliens(graphe):
  liste_avec_nbr_noeuds = list(graphe.degree(nbunch=None, weight=None))
  return sorted(liste_avec_nbr_noeuds, key=lambda noeud: noeud[1], reverse=True)  # sort by age


# print("Liste des noeuds par nombre d'arcs (degrés) décroissant:", triparnbreliens(g))


def triparpoidstotal(graphe):
  liste_avec_nbr_noeuds = list(graphe.degree(nbunch=None, weight='weight'))
  return sorted(liste_avec_nbr_noeuds, key=lambda noeud: noeud[1], reverse=True)  # sort by age


# list the neighbors of each node from our digraph, following directions:
# print("descendants de : ", [(n, ":", list(g.out_edges(n))) for n in g.nodes()])

# print(" impression des liens ascendants de chaque noeud ")
# print("ascendants de : ", [(n, ":", list(g.in_edges(n))) for n in g.nodes()])

# print(list("impression des adjacences", g.adjacency()))

# print("successeurs de : ", [(n, ":", list(g.successors(n))) for n in g.nodes()])
# print("prédécesseurs de: ", [(n, ":", list(g.predecessors(n))) for n in g.nodes()])

# print("re-affichage des nodes", g.nodes.data())

# calcul_niveaux(g)


# print("re-affichage des nodes")
# print(g.nodes.data())

# print("re-affichage des noms et des poids des nodes")
# print(nx.get_node_attributes(g, 'weight'))

# print("calcul des niveaux:", calcul_niveaux(g))

# Compute the degree of every node: degrees
# degrees = [len(list(g.neighbors(n))) for n in g.nodes()]

# Print the degrees
# print("affichage des degrés des noeud:", degrees)

# Le tableau est maintenant construit, on peut attaquer l'exploitation des données... mais d'abord, un petit tableau ?


# nodes
# node label and colors definition



# colors = [g.nodes[nod]['weight'] for nod in g.nodes]

# pos = nx.spring_layout(g, seed=13, k=1.666)  # positions for all nodes - seed for reproducibility
# pos = nx.nx_pydot.graphviz_layout(g)


# nx.draw_networkx_nodes(g, pos, node_size=500)

# edges : the following two examples need to prepare a liste of large (elarge) and one of small (esmall) edges
# nx.draw_networkx_edges(g, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(g, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
# pos=nx.spring_layout(g, seed=13, k=1.666)


####CALCUL des niveaux de taches sans networkx

def graphelvl(graphe):
  for node in graphe.nodes:
    graphe.nodes[node]['layer']=-1
  nodelayer = []
  while len(list(nodelayer)) < len(graphe.nodes):
    for node in graphe.nodes:
      if node not in nodelayer:
        if not list(graphe.predecessors(node)):
          graphe.nodes[node]['layer'] = 0
          nodelayer.append(node)
        else:
          max = 0
          traite = True
          for predecessor in graphe.predecessors(node):
            if graphe.nodes[predecessor]['layer'] < 0:
              traite=False
            else:
              predecessorlvl = int(graphe.nodes[predecessor]['layer'])
              if predecessorlvl > max :
                max = predecessorlvl
                traite=traite&True
          if traite:
            graphe.nodes[node]['layer'] = max + 1
            nodelayer.append(node)
            
  return graphe


graphelvl(g)

'''
# Compute the multipartite_layout using the "layer" node attribute
for layer, nodes in enumerate(nx.topological_generations(g)):
  # `multipartite_layout` expects the layer as a node attribute, so add the
  # numeric layer value as a node attribute
  for node in nodes:
    g.nodes[node]['layer'] = layer

'''


def niveaumax(graph):
  maxlayer = 1
  # for layer in g.nodes.data('layer'): # ajout d'un début et d'une fin ...
  listlayer = [(graph.nodes[node]['layer']) for node in graph.nodes()]
  # print("liste des niveaux des noeuds : ", listlayer)
  maxlayer = max(listlayer)
  return maxlayer


# ajout des nodes début et fin :
# ajout d'un début et d'une fin ...

print("nombre de niveaux de g : ", niveaumax(g) + 1)
nbreniveauxgraphinitial = niveaumax(g)


# TODO afficher debut et fin avec une couleur différente

def creerdebutfin(graph):
  noeudssanssucc = [nod for nod in g.nodes if not list(graph.successors(nod))]
  # print(list(noeudssanssucc))
  graph.add_node('debut', layer=-1, weight=0)
  graph.add_node('fin', layer=niveaumax(graph) + 1, weight=0)

  for node in graph.nodes():
    if graph.nodes[node]['layer'] == 0:
      graph.add_edge('debut', node)
  for lastnode in noeudssanssucc:
    graph.add_edge(lastnode, 'fin')
  return graph


creerdebutfin(g)


# print("affichage des nodes avec ajout des liens debut fin : ", g.edges())

# calcul de la date au plus tot

# print("tentative d affichage du chemin le plus court en partant de 'B' (les arcs ont tous un poids de 1) :" , nx.shortest_path(g)['B'])
# print("tentative d affichage du pagerank : ", nx.pagerank(g))


# print("exploration des poids : ", list(g.nodes.data('weight')))

# print("prédécesseurs de: ", [(nod, ":", list(g.predecessors(nod))) for nod in g.nodes()])


# dictpredecesseurs['D'] = list(g.predecessors('D'))


def calculdateauplustot(graph):
  dateauplustot: Dict = {}
  while len(dateauplustot) < len(graph.nodes):
    for nod in graph.nodes():
      # print("on récupère le nom du nodes : ", nod)
      # print("exploration des predecesseurs", list(graph.predecessors(nod)))
      listepredecesseurs = list(graph.predecessors(nod))
      if nod not in dateauplustot:
        # print("on traitre le noeud : ", nod, "type : ", type(nod))
        cheminmax = 0  # on initialise le cheminmax à 0 pour les noeuds du rang 0 qui n'ont pas d'antécédents
        traite = True  # on initialise une variable qui suppose qu'on a tous les antécédents dispos
        for pred in listepredecesseurs:  # on boucle sur la liste des prédécesseurs de nod
          # print("on traite le predecesseur :", pred)
          if pred in dateauplustot:  # si le predecesseurs à bien une date, on traite
            #  print("noeud : ", nod, "predecesseur : ",pred)
            # affichage du predecesseur en cours de traitement.
            cheminmax = max(cheminmax, (dateauplustot[pred] + graph.nodes.data('weight')[
              pred]))  # le chemin max , c'est le max entre (la plus grosse valeur connue)
            # et (la  valeur en chemin du predecesseur qu'on teste + son poids)
          else:  # si on trouve un predecesseur qui n'a pas de date au plus tot, on annule tout !
            traite = False
        if traite:  # on n'inscrit la valeur du nœud dans la table que si on a traité tous ses predecesseurs !
          dateauplustot[nod] = cheminmax
          graph.nodes[nod]['dateauplustot'] = cheminmax
          # print("dateauplustot de : ", nod, ": ", dateauplustot[nod])

  return graph


calculdateauplustot(g)


def calculdateauplustard(graph):
  listesuccesseurs = []
  dateauplustard = {}
  # dateauplustard['fin'] = calculdateauplustot(graph)['fin']
  dateauplustard['fin'] = (graph.nodes['fin']['dateauplustot'])
  graph.nodes['fin']['dateauplustard'] = graph.nodes['fin']['dateauplustot']
  print("date au plus tot fin = date au plus tard = ", dateauplustard['fin'])

  while len(dateauplustard) < len(graph):
    for nod in graph.nodes():
      listesuccesseurs = list(graph.successors(nod))
      if nod not in dateauplustard:
        mindepuisfin = dateauplustard['fin']
        traite = True
        for succ in listesuccesseurs:
          if succ in dateauplustard:
            mindepuisfin = min(mindepuisfin, (dateauplustard[succ] - graph.nodes.data('weight')[nod]))
          else:  # si on trouve un successeur qui n'a pas de date au plus tard, on annule tout !
            traite = False
        if traite:  # on n'inscrit la valeur du nœud dans la table que si on a traité tous ses predecesseurs !
          dateauplustard[nod] = mindepuisfin
          graph.nodes[nod]['dateauplustard'] = mindepuisfin
          # print (graph.nodes[nod]['dateauplustard'])
          # print("dateauplustard de : ", nod, ": ", dateauplustard[nod])
  return graph


calculdateauplustard(g)


def calculmarges(graph):
  for nod in g.nodes():
    graph.nodes[nod]['margedate'] = (graph.nodes[nod]['dateauplustard'] - graph.nodes[nod]['dateauplustot'])
  return graph


calculmarges(g)

listemarges = [(node, g.nodes[node]['dateauplustot'], g.nodes[node]['dateauplustard'], g.nodes[node]['margedate']) for
               node in g.nodes()]
# print("liste des marges des noeuds: ", listemarges)
# print(g.nodes['A']['layer'])
# print("affichage des noeud par layer :", sorted(g.nodes.data('layer'), key=lambda layer: layer[1]))
# print(nx.get_node_attributes(g,'layer')['A'])


cheminecritique = []


### PARCOURS d'un dict g par double index node et layer, trié par le deuxieme champs de l'attribut 'layer'
# for nodename, layer in sorted(g.nodes.data('layer'), key=lambda layer: layer[1]):
#   print(nodename)


def calculchemincritique(graph, listecritique):
  for node, layer in sorted(g.nodes.data('layer'), key=lambda layer: layer[1]):
    if graph.nodes[node]['margedate'] == 0:
      graph.nodes[node]['critique'] = True
      listecritique.append(node)
    #   listecritique.append()
  return graph, listecritique


calculchemincritique(g, cheminecritique)

print("noeuds éligibles au chemin critique :", cheminecritique)

# print(list(g.nodes()))

# listedates= [(node, g.nodes[node]['dateauplustard']) for node in g.nodes()]
# print (listedates)


# print("Affichage des dates au plus tot et date au plus tard du graphe", listedate)

###################################PARTIE AFFICHAGE################################

# calcul des positions en utilisant le layer multi
pos = nx.multipartite_layout(g, subset_key="layer")
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

# modification de la position entre deux affichages :
for px in list(newpos):
  # print(px)
  # print(newpos[px])
  # print(px[1][0])
  # print(px[1][1])
  # newpos[px][0] = newpos[px][0] -0.1
  newpos[px][1] = newpos[px][1] + 0.05

# print("liste des positions: ", list(pos.items()))

# print("type de pos:", type(pos))
# print("nouvelle liste de positions: ",list(newpos.items()))

# on dessine les noeuds et les arcs:
nx.draw_networkx_nodes(g, pos=pos)
nx.draw_networkx_edges(g, pos=pos, width=3)
# on dessine les labels :
nx.draw_networkx_labels(g, pos=pos, horizontalalignment='center')


labels = {nod: g.nodes[nod]['weight'] for nod in g.nodes}
nx.draw_networkx_labels(g, pos=newpos, labels=labels)


# nx.draw_networkx_edge_labels(g, pos=pos)

# print(sorted(g.nodes.data('dateauplustot'), key=lambda date: date[1]))
# for nodename, layer in sorted(g.nodes.data('layer'), key=lambda layer: layer[1]):

def make_gantt_chart(graph):
  fig, pltax = plt.subplots()
  noeudsordre = {}
  findetache = {}
  duree = {}
  marges = {}
  demarrage = graph.nodes.data('dateauplustot')
  for node in dict(sorted(graph.nodes.data('dateauplustot'), key=lambda date: date[1])):
    if ('fin' not in node) and ('debut' not in node):
      noeudsordre[node] = graph.nodes[node]['dateauplustot']
  for node in dict(graph.nodes.data()):
    findetache[node] = graph.nodes[node]['weight'] + graph.nodes[node]['dateauplustot']
  for node in dict(graph.nodes.data()):
    duree[node] = graph.nodes[node]['weight']
  for node in dict(graph.nodes.data()):
    marges[node] = graph.nodes[node]['margedate']
  y_chemincritique=1
  y_start = 11
  y_height = 8
  for noeud in noeudsordre:
    if graph.nodes[noeud]['margedate'] == 0:

      pltax.broken_barh([(demarrage[noeud], duree[noeud])], (y_start, y_height), facecolors='lime')

    else:

      pltax.broken_barh([(demarrage[noeud], duree[noeud])], (y_start, y_height), facecolors='cyan')
      pltax.broken_barh([(findetache[noeud], marges[noeud])], (y_start, y_height), facecolors='red')
    # pltax.text(findetache[noeud] + marges[noeud] + 0.5, y_start + y_height, noeud)

    pltax.text(demarrage[noeud]+(duree[noeud])/2, y_start + (y_height/2) + 1 , noeud)
    y_start += 10
  pltax.set_xlim(0, max(findetache.values()) + 5)
  pltax.set_ylim(len(duree) * 10)
  pltax.set_xlabel('Temps')
  pltax.set_ylabel('Taches')
  i = 5
  y_ticks = []
  y_labels = []
  for node in noeudsordre:
    y_labels.append(node[0])
  #while i < len(noeudsordre) * 10:
  #  y_ticks.append(i)
  #  i += 10
  for i in range (len(noeudsordre)+2):
    y_ticks.append(i*10)
    i += 1

  pltax.set_yticks(y_ticks)
  pltax.set_yticklabels([])
  plt.title('Diagramme de Gantt, en escalier ', size=18)
  plt.tick_params(
    axis='both',  # changes apply to the y-axis
    which='both',  # both major and minor ticks are affected
    left='off',  # ticks along the left edge are off
    labelleft='off')  # labels along the left edge are off
  #


# preparation de l'affichage secondaire ordonné:
# ax = plt.subplots()
# ax.margins(0.08)
# plt.axis("on")
# plt.tight_layout()


make_gantt_chart(g)

plt.show()
'''

shrtpath = nx.shortest_path(g, source='debut', target='fin')
# Affichage du chemin le plus court en utilisant la fonction integree a networX
print("le chemin le plus court est : ", shrtpath)

'''
