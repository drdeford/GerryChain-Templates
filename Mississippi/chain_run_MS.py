# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:31:27 2019

@author: daryl
"""

import csv
import os
from functools import partial
import json

import geopandas as gpd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.updaters import cut_edges
from gerrychain.tree import recursive_tree_part

state_abbr = "MS"
housen = "HOU"
state_fip = "28"
num_districts = 4



newdir = "./Outputs/"+state_abbr+housen+"_BG/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")



tract_graph = Graph.from_file("./BG"+state_fip+".shp",reproject=False)
tract_graph.to_json("./"+state_abbr+"_BG.json")
print("built graph")


graph_path = "./"+state_abbr+"_BG.json"
plot_path = "./BG"+state_fip+".shp"


df = gpd.read_file(plot_path)


unique_label = "GEOID10"
pop_col = "TOTPOP"


#df["nWHITE"] = df["TOTPOP"] - df["WHITE"]
import pandas as pd


    
    
    

df["TOTPOP"]=pd.to_numeric(df["TOTPOP"])
df["VAP"]=pd.to_numeric(df["VAP"])
df["BVAP"]=pd.to_numeric(df["BVAP"])




df["nBVAP"] = df["VAP"]-df["BVAP"]


graph = Graph.from_json(graph_path)

graph.join(df)


for i in range(1):
    cddict =  recursive_tree_part(graph,range(num_districts),df["TOTPOP"].sum()/num_districts,"TOTPOP", .001,1)
    
    df["initial"]=df.index.map(cddict)
    
    df.plot(column="initial",cmap="jet")
    plt.savefig(newdir+"initial.png")
    plt.close()

with open(newdir+"init.json", 'w') as jf1:
	    json.dump(cddict, jf1)

print("saved Initial")
updater = {
        "population": updaters.Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
    "BVAP":Election("BVAP",{"BVAP":"BVAP","nBVAP":"nBVAP"})
            }


initial_partition = Partition(graph, cddict, updater)

ideal_population = sum(initial_partition["population"].values()) / len(
    initial_partition
)

proposal = partial(
    recom, pop_col="TOTPOP", pop_target=ideal_population, epsilon=0.05, node_repeats=1
)

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, 0.05),
          #constraints.single_flip_contiguous#no_more_discontiguous
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=20
)

cuts=[]
BVAPS=[]

t = 0
for part in chain:
    cuts.append(len(part["cut_edges"]))
    BVAPS.append(sorted(part["BVAP"].percents("BVAP")))
    t+=1
    if t %10 == 0:        
        
        
        df["current"]=df.index.map(dict(part.assignment))
        
        df.plot(column="current",cmap="jet")
        plt.savefig(newdir+"plot"+str(t)+".png")
        plt.close()
        
        with open(newdir+"assignment"+str(t)+".json", 'w') as jf1:
            json.dump(dict(part.assignment), jf1)






df["final"]=df.index.map(dict(part.assignment))

df.plot(column="final",cmap="jet")
plt.savefig(newdir+"final.png")
plt.close()



plt.figure()
plt.hist(cuts)
plt.show()

plt.figure()
plt.boxplot(
            np.array(BVAPS),
            whis=[1, 99],
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="None", color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
)

plt.plot([1,2,3,4],[.226,.272,.333,.635], 'o', color="hotpink", label="Enacted",markersize=10)

plt.axhline(y=.4,color='r',label="40%",linewidth=5)

plt.axhline(y=.45,color='y',label="45%",linewidth=5)

plt.axhline(y=.5,color='g',label="50%",linewidth=5)
plt.plot([],[],color='k',label="ReCom Ensemble")
plt.xlabel("Sorted Districts")
plt.ylabel("BVAP%")

plt.legend()

plt.show()
