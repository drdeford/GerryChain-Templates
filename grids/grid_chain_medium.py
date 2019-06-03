import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from networkx.readwrite import json_graph
import math
from functools import partial
import networkx as nx
import numpy as np


from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election,Tally,cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap

import networkx as nx
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random



gn=6
k=5
ns=50
p=.5

graph=nx.grid_graph([k*gn,k*gn])



for n in graph.nodes():
    graph.node[n]["population"]=1

   
    if random.random()<p:
        graph.node[n]["pink"]=1
        graph.node[n]["purple"]=0
    else:
        graph.node[n]["pink"]=0
        graph.node[n]["purple"]=1
    if 0 in n or k*gn-1 in n:
        graph.node[n]["boundary_node"]=True
        graph.node[n]["boundary_perim"]=1

    else:
        graph.node[n]["boundary_node"]=False

#this part adds queen adjacency
#for i in range(k*gn-1):
#    for j in range(k*gn):
#        if j<(k*gn-1):
#            graph.add_edge((i,j),(i+1,j+1))
#            graph[(i,j)][(i+1,j+1)]["shared_perim"]=0
#        if j >0:
#            graph.add_edge((i,j),(i+1,j-1))
#            graph[(i,j)][(i+1,j-1)]["shared_perim"]=0
        
        
        
########## BUILD ASSIGNMENT
cddict = {x: int(x[0]/gn)  for x in graph.nodes()}        
pos = {x:x for x in graph.nodes()}


####CONFIGURE UPDATERS

def step_num(partition):
    parent = partition.parent

    if not parent:
        return 0


    return parent["step_num"] + 1

updaters = {'population': Tally('population'),
                    'cut_edges': cut_edges,
                    'step_num': step_num,
                    #"Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})
                    }                  
                    
                    

                    

#########BUILD FIRST PARTITION

grid_partition = Partition(graph,assignment=cddict,updaters=updaters)

#ADD CONSTRAINTS
popbound=within_percent_of_ideal_population(grid_partition,.1)                      

#########Setup Proposal
ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)

tree_proposal = partial(recom,
                       pop_col="population",
                       pop_target=ideal_population,
                       epsilon=0.05,
                       node_repeats=1
                      )
                      
#######BUILD AND RUN FIRST MARKOV CHAIN


recom_chain=MarkovChain(tree_proposal, Validator([popbound]),accept=always_accept,
initial_state=grid_partition, total_steps=100)


for part in recom_chain:
    pass
    

plt.figure()
nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [part.assignment[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
plt.savefig("./plots/medium/end_of_tree.png")
plt.close()

print("Finished ReCom")
#########BUILD SECOND PARTITION
                   
squiggle_partition = Partition(graph,assignment=part.assignment,updaters=updaters)


#ADD CONSTRAINTS
popbound=within_percent_of_ideal_population(squiggle_partition,.1)                      

                      
#######BUILD AND RUN SECOND MARKOV CHAIN

squiggle_chain=MarkovChain(propose_random_flip, Validator([single_flip_contiguous, popbound]),accept=always_accept, initial_state=squiggle_partition, total_steps=100000)


for part2 in squiggle_chain:
    pass
    
    
plt.figure()
nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [part2.assignment[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
plt.savefig("./plots/medium/end_of_boundary.png")
plt.close()
print("Finished Squiggling")
#########BUILD FINAL PARTITION

final_partition = Partition(graph,assignment=part2.assignment,updaters=updaters)


#ADD CONSTRAINTS
popbound=within_percent_of_ideal_population(final_partition,.3)                      


#########Setup Spectral Proposal

def spectral_cut(G):
    nlist = list(G.nodes())
    n = len(nlist)
    AM = nx.adjacency_matrix(G)
    NLM = (nx.normalized_laplacian_matrix(G)).todense()
    #LM = (nx.laplacian_matrix(G)).todense()
    NLMva, NLMve = LA.eigh(NLM)
    NFv = NLMve[:,1]
    xNFv = [NFv.item(x) for x in range(n)]
    node_color = [xNFv[x] > 0 for x in range(n)]
   
    clusters={nlist[x]:node_color[x] for x in range(n)}

    return clusters

def propose_spectral_merge(partition):
    edge = random.choice(tuple(partition['cut_edges']))
    #print(edge)
    et=[partition.assignment[edge[0]],partition.assignment[edge[1]]]
    #print(et)
    sgn=[]
    for n in partition.graph.nodes():
        if partition.assignment[n] in et:
            sgn.append(n)

    #print(len(sgn))
    sgraph = nx.subgraph(partition.graph,sgn)

    edd={0:et[0],1:et[1]}

    #print(edd)

    clusters = spectral_cut(sgraph)
    #print(len(clusters))
    flips={}
    for val in clusters.keys():
        flips[val]=edd[clusters[val]]

    #print(len(flips))
    #print(partition.assignment)
    #print(flips)
    return partition.flip(flips)





                      
#######BUILD AND RUN FINAL MARKOV CHAIN


final_chain=MarkovChain(propose_spectral_merge, Validator([]),accept=always_accept,
initial_state=final_partition, total_steps=100)


for part3 in final_chain:
    plt.figure()
    nx.draw(graph,pos, node_color = [part3.assignment[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
    plt.savefig(f"./plots/medium/spectral_step{part3['step_num']:02d}.png")
    plt.close() 
 
print("Finished Spectral")   
    


