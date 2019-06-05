# GerryChain-Templates
This repo provides an evolving set of instructional guides for using the <a href="https://github.com/mggg/GerryChain">GerryChain</a> package for generating dual graph partitions. More detailed technical documentation can be found <a href="https://people.csail.mit.edu/ddeford/GerryChain_Guide.pdf">here</a> and my introduction to the mathematics of Markov chains and MCMC for redistrcitng can be found here: (<a href="https://people.csail.mit.edu/ddeford/MCMC_Intro_plus.pdf">pdf</a>) (<a href="https://people.csail.mit.edu/ddeford/mcmc_intro.php">webpage</a>) (<a href="https://github.com/drdeford/MCMC_Intro">GitHub</a>).

<H2> Overview </H2>
These templates assume that you have already installed GerryChain (see <a href="https://github.com/mggg/GerryChain">here</a> or <a href="https://people.csail.mit.edu/ddeford/GerryChain_Guide.pdf">here</a> for directions) and are ready to start building your own ensembles. Since some of the examples use real-world data, you should clone this repo to your computer - this will also make it easy to get access to updates to the templates. Once you have downloaded everything you should be able to run the templates individually by navigating to the appropriate subfolder and entering: 

```python
python grid_chain_simple.py
```

The current templates are:
<ul>
    <li> Grids</li>
    <ul> 
        <li><b>  grid_chain_simple.py </b> </li>
        <li><b>  grid_chain_medium.py </b> </li>
        <li><b>  grid_chain_complicated.py </b> </li>
    </ul>
    <li> Pennsylvania</li>
    <ul>
        <li> <b>PA_Chain.py</b></li>
    </ul>
    <li> Alaska</li>
    <ul>
        <li> <b>AK_Chain.py</b></li>
    </ul>
</ul>

<H2> Grids are fun! </H2>

The best place to start is with the <a href="https://github.com/drdeford/GerryChain-Templates/blob/master/grids/grid_chain_simple.py">simple grids example</a>. This template breaks the code into pieces, building the graph and partition components separately before doing a couple of small runs to compare ReCom to Flip. We start by importing the packages we need:

```python
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
```

Next we build a gn x k by gn x k graph split into k vertical columns: 

```python



#BUILD GRAPH

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

######PLOT GRIDS
        
plt.figure()
nx.draw(graph, pos = {x:x for x in graph.nodes()} ,node_size = ns, node_shape ='s')
plt.show()

cdict = {1:'pink',0:'purple'}

plt.figure()
nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [cdict[graph.node[x]["pink"]] for x in graph.nodes()],node_size = ns, node_shape ='s' )
plt.show()

plt.figure()
nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [cddict[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
plt.show()
```

<table>
  <tr><td> Original Graph </td><td>Random Vote Data</td><td>Initial Partition</td></tr>
  <tr><td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/Original_Grid.png" width = 300/></td><td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/PP_votes.png" width = 300/></td><td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/initial_partition.png" width = 300/></td></tr>
  </table>
  

Next we configure the partitions, updaters, and constraints. 

```python

####CONFIGURE UPDATERS

def step_num(partition):
    parent = partition.parent

    if not parent:
        return 0


    return parent["step_num"] + 1

updaters = {'population': Tally('population'),
                    'cut_edges': cut_edges,
                    'step_num': step_num,
                    "Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})}                  
                    
                    

#########BUILD PARTITION

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
                      
#######BUILD MARKOV CHAINS


recom_chain=MarkovChain(tree_proposal, Validator([popbound]),accept=always_accept,
initial_state=grid_partition, total_steps=100)

boundary_chain=MarkovChain(propose_random_flip, Validator([single_flip_contiguous, popbound]),accept=always_accept,
initial_state=grid_partition, total_steps=10000)

```

Finally, we run two chains, one with each proposal method, and record a variety of statistics. The final state of each of the two chains will be displayed. Notice that even though the ReCom chain only takes 100 steps, it travels much further than the 10,000 step boundary chain. 

```python
#########Run MARKOV CHAINS

rsw = []
rmm = []
reg = []
rce = []

for part in recom_chain:
    rsw.append(part["Pink-Purple"].wins("Pink"))
    rmm.append(mean_median(part["Pink-Purple"]))
    reg.append(efficiency_gap(part["Pink-Purple"]))
    rce.append(len(part["cut_edges"]))
    """
    plt.figure()
    nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
    plt.savefig(f"./Figures/recom_{part['step_num']:02d}.png")
    plt.close()
    """
    
plt.figure()
nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
plt.show()


bsw = []
bmm = []
beg = []
bce = []

for part in boundary_chain:
    bsw.append(part["Pink-Purple"].wins("Pink"))
    bmm.append(mean_median(part["Pink-Purple"]))
    beg.append(efficiency_gap(part["Pink-Purple"]))
    bce.append(len(part["cut_edges"]))
    """
    plt.figure()
    nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
    plt.savefig(f"./Figures/boundary_{part['step_num']:04d}.png")
    plt.close()
    """

plt.figure()
nx.draw(graph, pos = {x:x for x in graph.nodes()}, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='s',cmap = 'tab20')
plt.show()
```

<table>
  <tr><td> ReCom Ensemble</td><td>Boundary Ensemble</td></tr>
  <tr><td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/recom.gif" width = 300/></td><td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/smaller_boundary.gif" width = 300/></td></tr>
  </table>


After the chain runs, we make some plots comparing the behavior of the ensembles. 

```python
##################Partisan Plots

names = ["Cut Edges", "Mean Median", "Pink Seats", "Efficiency Gap"]
lists = [[rce, bce], [rmm, bmm], [rsw, bsw], [reg, beg]]


for z in range(4):
    plt.figure()
    plt.suptitle(f"{names[z]} Comparison")
    
    plt.subplot(2, 2, 1)
    plt.plot(lists[z][0])
    
    

    plt.subplot(2, 2, 3)
    plt.hist(lists[z][0])
    plt.title("ReCom Ensemble")


    plt.subplot(2, 2, 2)
    plt.plot(lists[z][1])

    plt.subplot(2, 2, 4)
    plt.hist(lists[z][1])
    plt.title("Boundary Flip Ensemble")
    fig = plt.gcf()
    fig.set_size_inches((8,7))
    plt.show()
```

<table>
  <tr><td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/ce.png" width = 600/></td><td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/mm.png" width = 600/></td></tr>
   <tr>   <td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/sw.png" width = 600/></td>
        <td><img src="https://raw.githubusercontent.com/drdeford/GerryChain-Templates/master/Figures/eg.png" width = 600/></td></tr>
  </table>

    



