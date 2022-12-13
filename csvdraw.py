import copy


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def toarray(fichier):  # Fonction qui prends un fichier en paramètre et qui rend un array
  myarray = np.loadtxt(fichier, delimiter=";", dtype=str)
  return myarray


# Demander le nom du fichier csv à l'utilisateur defaut = test.csv
fichier = input('Entrez le fichier à parser : (default test.csv)')
if 'csv' not in fichier:
  fichier = 'test.csv'

arr = toarray(fichier) # appel de la fonction maison toarray


# print(arr)
# print("longueur du premier array : ", len(arr))


def flip_and_chop_array(myarray):  # fonction inutilisée.
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


def construitgraphe(myarray):  # fonction qui retourne un graphe à partir d'un array, ne parse que les 3 premiers champs
  nombreelements = len(myarray)
  print("nombre d'éléments:", nombreelements)
  
  for i in range(nombreelements):
    # print("intégration de : ", (myarray[i][0]), (myarray[i][1]), (myarray[i][2]))
    g.add_node(myarray[i][0], weight=float(myarray[i][1]))  # on ajoute le noeud au graphe
    if bool(myarray[i][2]):  # si on a des voisins en 3eme enregistrement
      liste_voisins = (myarray[i][2]).split(',')  # on decoupe la liste des voisins
      for vois in liste_voisins:  # pour chaque voisin , on crée l arrete voisin - noeud courant.
        # print("traitement de : ", myarray[i][0], ", ajout de : ", vois)
        g.add_edge(vois, myarray[i][0])
  return g


g = nx.DiGraph()# initialisation d'un DiGraphe
g = construitgraphe(arr) # renseignement des attributs du graphe


# prints out the nature of our digraph:
# print(g)

# prints out the complete set of edges with their attributes
# print("listing des edges en format brut:", g.edges.data())

# list the neighbors of each node from our digraph, following directions:
# print("descendants de : ", [(n, ":", list(g.out_edges(n))) for n in g.nodes()])

# print(" impression des liens ascendants de chaque noeud ")
# print("ascendants de : ", [(n, ":", list(g.in_edges(n))) for n in g.nodes()])

# print(list("impression des adjacences", g.adjacency()))

# print("successeurs de : ", [(n, ":", list(g.successors(n))) for n in g.nodes()])
# print("prédécesseurs de: ", [(n, ":", list(g.predecessors(n))) for n in g.nodes()])

# print("re-affichage des nodes", g.nodes.data())

# print("re-affichage des noms et des poids des nodes")
# print(nx.get_node_attributes(g, 'weight'))

# Compute the degree of every node: degrees
# degrees = [len(list(g.neighbors(n))) for n in g.nodes()]

####CALCUL des niveaux de taches sans networkx

def graphelvl(graphe):
  for node in graphe.nodes:
    graphe.nodes[node]['layer'] = -1
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
              traite = False
            else:
              predecessorlvl = int(graphe.nodes[predecessor]['layer'])
              if predecessorlvl > max:
                max = predecessorlvl
                traite = traite & True
          if traite:
            graphe.nodes[node]['layer'] = max + 1
            nodelayer.append(node)
  
  return graphe


# appel de la fonction qui ajoute l'attribut 'layer' aux noeuds du graphe.
graphelvl(g)


### Calcul des niveaux des taches avec networkx, fonction abandonnée ###
def graphelayer(graphe):
  for layer, nodes in enumerate(nx.topological_generations(graphe)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
    for node in nodes:
      graphe.nodes[node]['layer'] = layer
  return graphe


## comparaison des resultats entre la méthode networkx et la méthode maison :
# glayer=copy.deepcopy(g)
# graphelayer(glayer)
# print(list(glayer.nodes.data()),list(g.nodes.data()))


def niveaumax(graph):  # fonction qui calcule le niveau max du graphe initial
  
  listlayer = [(graph.nodes[node]['layer']) for node in graph.nodes()]  # on liste les niveaux présents
  # print("liste des niveaux des noeuds : ", listlayer)
  maxlayer = max(listlayer)  # on prend le max de la liste (numéroté de 0 à N)
  return maxlayer


# ajout des nodes début et fin :
# ajout d'un début et d'une fin ...

print("nombre de niveaux de g : ", niveaumax(g) + 1)
nbreniveauxgraphinitial = niveaumax(g)


def creerdebutfin(graph):  # pour pouvoir calculer le flow d'un bout a l autre, on ajoute debut et fin.
  noeudssanssucc = [nod for nod in g.nodes if not list(graph.successors(nod))]  # liste des noeuds sans successeur
  # print(list(noeudssanssucc))
  graph.add_node('debut', layer=-1, weight=0)  # on ajoute le début avec un layer -1 pour etre avant les premiers noeuds
  graph.add_node('fin', layer=niveaumax(graph) + 1, weight=0)  # la fin est après le niveau le plus haut
  
  for node in graph.nodes():
    if graph.nodes[node]['layer'] == 0:
      graph.add_edge('debut', node)  # on relie les noeuds de niveau 0 au début
  for lastnode in noeudssanssucc:
    graph.add_edge(lastnode, 'fin')  # on relie les noeuds sans successeurs à la fin, Quel que soit le niveau
  return graph


creerdebutfin(g)  # appel de la fonction pour ajouter debut et fin à 'g'


# print("affichage des nodes avec ajout des liens debut fin : ", g.edges())

# print("exploration des poids : ", list(g.nodes.data('weight')))

# print("prédécesseurs de: ", [(nod, ":", list(g.predecessors(nod))) for nod in g.nodes()])

# dictpredecesseurs['D'] = list(g.predecessors('D'))


def calculdateauplustot(graph):  # fonction de calcul des dates au plus tot
  dateauplustot: dict = {}  # un dictionnaire des dates au plus tot
  while len(dateauplustot) < len(graph.nodes):  # tant qu'on a pas autant de dates que de noeuds, on boucle !
    for nod in graph.nodes():
      # print("on récupère le nom du nodes : ", nod)
      if nod not in dateauplustot:  # si on a pas déjà traité le noeud :
        listepredecesseurs = list(graph.predecessors(nod))  # on construit une liste des predecesseur du noeud courant
        # print("on traitre le noeud : ", nod, "type : ", type(nod))
        cheminmax = 0  # on initialise le cheminmax à 0 pour les noeuds du rang 0 qui n'ont pas d'antécédents
        traite = True  # on initialise une variable qui suppose qu'on a tous les antécédents dispos
        for pred in listepredecesseurs:  # on boucle sur la liste des prédécesseurs de nod
          # print("on traite le predecesseur :", pred)
          if pred in dateauplustot:  # si le predecesseurs à bien une date, on traite
            #  print("noeud : ", nod, "predecesseur : ",pred)
            # affichage du predecesseur en cours de traitement.
            cheminmax = max(cheminmax, (dateauplustot[pred] + graph.nodes.data('weight')[pred]))
            # le chemin max, c'est le max entre (la plus grosse valeur connue)
            # et (la date au plus tot du predecesseur qu'on teste + son poids)
          else:  # si on trouve un predecesseur qui n'a pas de date au plus tot, on annule tout !
            traite = False
        if traite:  # on n'inscrit la valeur du nœud dans la table que si on a traité tous ses predecesseurs !
          dateauplustot[nod] = cheminmax  # on inscrit le noeud dans la table des noeuds traités
          graph.nodes[nod]['dateauplustot'] = cheminmax  # on ajoute un attribut au graph pour ce noeud
          # DEBUG #
          # print("dateauplustot de : ", nod, ": ", dateauplustot[nod])
  
  return graph


calculdateauplustot(g)  # appel de la fonction calculdateauplustot pour 'g'


def calculdateauplustard(graph):  # definition de la fonction dateauplustard
  listesuccesseurs = []  # on initialise une liste de successeurs
  dateauplustard = {}  # un dictionnaire des dates au plus tard
  dateauplustard['fin'] = graph.nodes['fin'][
    'dateauplustot']  # par definition, la fin est sur le chemin critique. date au plus tot = date au plus tard.
  graph.nodes['fin']['dateauplustard'] = graph.nodes['fin']['dateauplustot']  # et on l'enregistre dans le graphe
  print("date au plus tot fin = date au plus tard = ", dateauplustard['fin'])
  
  while len(dateauplustard) < len(graph):  # tant qu'on a pas tout traité
    for nod in graph.nodes():  # on boucle sur tous les noeuds
      if nod not in dateauplustard:  # si on n'a pas déjà traité
        listesuccesseurs = list(graph.successors(nod))  # on liste les successeurs
        mindepuisfin = dateauplustard['fin']  # combien au max depuis la fin ?
        traite = True  # on suppose qu'on va pouvoir traiter
        for succ in listesuccesseurs:  # parmis les successseurs
          if succ in dateauplustard:  # si on a déjà sa date au plus tard ...
            mindepuisfin = min(mindepuisfin, (dateauplustard[succ] - graph.nodes.data('weight')[nod]))
            # la date au plus tard, c'est le min entre le meilleur min,
            # et la date au plus tard du successeur qu'on teste moins le poids du noeud courant
          else:  # si on trouve un successeur qui n'a pas de date au plus tard, on annule tout !
            traite = False
        if traite:  # on n'inscrit la valeur du nœud dans la table que si on a traité tous ses predecesseurs !
          dateauplustard[nod] = mindepuisfin  # on enregistre le noeud dans la liste des dates au plus tard
          graph.nodes[nod]['dateauplustard'] = mindepuisfin  # et on inscrit l'info dans le graph pour le noeud courant
          # print (graph.nodes[nod]['dateauplustard'])
          # print("dateauplustard de : ", nod, ": ", dateauplustard[nod])
  return graph


calculdateauplustard(g)  # appel de la fonction dates au plus tard pour 'g'


def calculmarges(graph):  # fonction du calcul des marges
  for nod in g.nodes():  # on parcours le noeuds
    graph.nodes[nod]['margedate'] = (graph.nodes[nod]['dateauplustard'] - graph.nodes[nod]['dateauplustot'])
    # la marge, c'est date au plus tard - date au plus tot, zou, direct dans le graph !
  return graph


calculmarges(g)  # hop la

# DEBUG #
listemarges = [(node, g.nodes[node]['dateauplustot'], g.nodes[node]['dateauplustard'], g.nodes[node]['margedate']) for
               node in g.nodes()]  #
print("liste des marges des noeuds: ", listemarges)

# print("affichage des noeud par layer :", sorted(g.nodes.data('layer'), key=lambda layer: layer[1]))
# print(nx.get_node_attributes(g,'layer'))


## DEBUG ##
# PARCOURS d'un dict g par double index node et layer, trié par le deuxieme champs de l'attribut 'layer'
# for nodename, layer in sorted(g.nodes.data('layer'), key=lambda layer: layer[1]):
#   print(nodename)


cheminecritique = []  # initialisation d'un tableau des chemins critiques, dans l'ordre des layers du graphe


def calculchemincritique(graph, listecritique):
  # fonction qui va renseigner si un noeud est dans le chemin critique, et en rendre la liste
  for node, layer in sorted(g.nodes.data('layer'),
                            key=lambda layer: layer[1]):  # on parcours les noeuds dans l'ordre des layers
    if graph.nodes[node]['margedate'] == 0:  # si la marge est nulle
      graph.nodes[node]['critique'] = True  # on inscrit le chemin critique dans les infos du noeud
      listecritique.append(node)  # et on le rajoute à la liste des candidats
  return graph, listecritique


calculchemincritique(g, cheminecritique)
# on appelle la fonction qui renseigne le graph et le tableau avec le chemin critique

print("noeuds éligibles au chemin critique :", cheminecritique)

###################################PARTIE AFFICHAGE################################

# calcul des positions en utilisant le layer multi qui exploite le champs layer ajouté aux noeuds du graphe
pos = nx.multipartite_layout(g, subset_key="layer")

# print("liste des positions: ")
# print(list(pos.items()))

# Décalage des positions des labels des nodes pour afficher les poids au dessus
# modification de la position entre deux affichages :
newpos = copy.deepcopy(pos)
for px in list(newpos):
  newpos[px][1] = newpos[px][1] + 0.05  # ajout de 0.05 en y à la pos des poids

# print("nouvelle liste de positions: ",list(newpos.items()))

# on dessine les noeuds et les arcs:
nx.draw_networkx_nodes(g, pos=pos)
nx.draw_networkx_edges(g, pos=pos, width=3)
# on dessine les labels :
nx.draw_networkx_labels(g, pos=pos, horizontalalignment='center')

labels = {nod: g.nodes[nod]['weight'] for nod in g.nodes}  # on prepare une liste des labels des poids
nx.draw_networkx_labels(g, pos=newpos, labels=labels)  # et on les affiche avec les pos décalées au dessus du noeud


# preparation de l'affichage du GANTT à partir d'un modèle .
def make_gantt_chart(graph):
  fig, pltax = plt.subplots()  # on cree un affichage séparé
  noeudsordre = {}  # un dict des noeuds dans l'ordre
  findetache = {}  # un dict des fins de taches (debut+duree)
  duree = {}  # un dict des durees
  marges = {}  # un dict des marges
  demarrage = graph.nodes.data('dateauplustot')  # un dict des marges au plus tot
  for node in dict(sorted(graph.nodes.data('dateauplustot'), key=lambda date: date[1])):
    # on boucle sur les noeuds dans l'ordre d'apparition
    if ('fin' not in node) and ('debut' not in node):  # pas d'affichage de debut et fin dans le gantt
      noeudsordre[node] = graph.nodes[node]['dateauplustot']  # la liste des noeuds dans l'ordre des dates au plus tot
  for node in dict(graph.nodes.data()):  # la fin c'est le debut + duree
    findetache[node] = graph.nodes[node]['weight'] + graph.nodes[node]['dateauplustot']
  for node in dict(graph.nodes.data()):  # la duree c'est le champs 'weight'
    duree[node] = graph.nodes[node]['weight']
  for node in dict(graph.nodes.data()):  # la marge c'est la marge
    marges[node] = graph.nodes[node]['margedate']
  y_chemincritique = 1  # si on veut aligner le chemin critique sur la premiere ligne, mais c'est moche
  y_start = 11 # on evite la premiere ligne pour aerer le tableau
  y_height = 8 # chaque barre fera 8 pixels de haut sur un emplacement de 10
  for noeud in noeudsordre: # dans l'ordre des dates au plus tot
    if graph.nodes[noeud]['margedate'] == 0: # si chemin critique
      pltax.broken_barh([(demarrage[noeud], duree[noeud])], (y_start, y_height), facecolors='lime') # vert petant pour le chemin critique
    else:
      pltax.broken_barh([(demarrage[noeud], duree[noeud])], (y_start, y_height), facecolors='cyan') # cyan pour les autres
      pltax.broken_barh([(findetache[noeud], marges[noeud])], (y_start, y_height), facecolors='red') # la marge apparait apres la fin en rouge
    # pltax.text(findetache[noeud] + marges[noeud] + 0.5, y_start + y_height, noeud)
    
    pltax.text(demarrage[noeud] + (duree[noeud]) / 2, y_start + (y_height / 2) + 1, noeud) # on rappelle le nom de la tache : noeud
    y_start += 10 # on decale de 10 pixels pour la tache suivante
  
  pltax.set_xlim(0, max(findetache.values()) + 5) # dimension de 'affichage en fonction de la tache la plus tardive
  pltax.set_ylim(len(duree) * 10)   # en hauteur c'est le nombre de taches , y compris debut en fin pour avoir de la place
  pltax.set_xlabel('Temps') # legende X
  pltax.set_ylabel('Taches') # legende Y
  i = 5
  y_ticks = []
  y_labels = []
 
  for i in range(len(noeudsordre) + 2):
    y_ticks.append(i * 10)  # affichage des séparateurs de l'axe des ordonnées
    i += 1
  
  pltax.set_yticks(y_ticks) # tick tick tick
  pltax.set_yticklabels([]) # pas de nom
  plt.title('Diagramme de Gantt, en escalier ', size=18) # titre du gantt
  plt.tick_params(
    axis='both',  # changes apply to the y-axis
    which='both',  # both major and minor ticks are affected
    left='off',  # ticks along the left edge are off
    labelleft='off')  # labels along the left edge are off
  #


make_gantt_chart(g)

plt.show() # affichage général avec matplot
'''

shrtpath = nx.shortest_path(g, source='debut', target='fin')
# Affichage du chemin le plus court en utilisant la fonction integree a networX
print("le chemin le plus court est : ", shrtpath)

'''
