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
from gerrychain.tree import recursive_tree_part, bipartition_tree_random


state_abbr = "MA"
housen = "SEN"
state_fip = "25"
num_districts = 40



newdir = "./Outputs/"+state_abbr+housen+"_Precincts/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")





graph = Graph.from_json("./Data/MA2010.json")


df = gpd.read_file("./Data/MA_precincts_12_16.shp")
print("built graph")


unique_label = "GEOID10"
pop_col = "TOTPOP"


#df["nWHITE"] = df["TOTPOP"] - df["WHITE"]
import pandas as pd

num_elections = 6


election_names = [
    "BVAP",
    "SEN12",
    "PRES12",
    "SEN13",
    "SEN14",
    "PRES16"]

election_columns = [
    ["BVAP", "nBVAP"],
    ["SEN12D","SEN12R"],
    ["PRES12D","PRES12R"],
    ["SEN13D","SEN13R"],
    ["SEN14D","SEN14R"],
    ["PRES16D", "PRES16R"]
    ]


totpop = 0 
for n in graph.nodes():
    graph.node[n]["TOTPOP"] = int(graph.node[n]["TOTPOP"])
    graph.node[n]["VAP"] = int(graph.node[n]["VAP"])
    graph.node[n]["BVAP"] = int(graph.node[n]["BVAP"])
    for i in range(1,num_elections):
        for j in election_columns[i]:
            graph.node[n][j] = int(graph.node[n][j].replace(',', ''))


    graph.node[n]["nBVAP"] = graph.node[n]["VAP"] - graph.node[n]["BVAP"]
    totpop+= graph.node[n]["TOTPOP"]



    
    


for i in range(1):
    cddict =  recursive_tree_part(graph,range(num_districts),totpop/num_districts,"TOTPOP", .01,1)
    
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
            }




elections = [
    Election(
        election_names[i],
        {"First": election_columns[i][0], "Second": election_columns[i][1]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}

updater.update(election_updaters)


initial_partition = Partition(graph, cddict, updater)
print("built initial partition")



ideal_population = sum(initial_partition["population"].values()) / len(
    initial_partition
)

proposal = partial(
    recom, pop_col="TOTPOP", pop_target=ideal_population, epsilon=0.05, node_repeats=1, method = bipartition_tree_random
)

#compactness_bound = constraints.UpperBound(
#    lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
#)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, 0.05),
          #constraints.single_flip_contiguous#no_more_discontiguous
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=100000
)

print("initialized chain")


with open(newdir + "Start_Values.txt", "w") as f:
    f.write("Values for Starting Plan: Tree Recursive\n \n ")
    f.write("Initial Cut: " + str(len(initial_partition["cut_edges"])))
    f.write("\n")
    f.write("\n")

    for elect in range(num_elections):
        f.write(
            election_names[elect]
            + "District Percentages"
            + str(
                sorted(initial_partition[election_names[elect]].percents("First"))
            )
        )
        f.write("\n")
        f.write("\n")

        f.write(
            election_names[elect]
            + "Mean-Median :"
            + str(mean_median(initial_partition[election_names[elect]]))
        )

        f.write("\n")
        f.write("\n")

        f.write(
            election_names[elect]
            + "Efficiency Gap :"
            + str(efficiency_gap(initial_partition[election_names[elect]]))
        )

        f.write("\n")
        f.write("\n")

        f.write(
            election_names[elect]
            + "How Many Seats :"
            + str(initial_partition[election_names[elect]].wins("First"))
        )

        f.write("\n")
        f.write("\n")


print("wrote starting values")

pop_vec = []
cut_vec = []
votes = [[], [], [], [],[],[]]
mms = []
egs = []
hmss = []


t = 0
for part in chain:

    pop_vec.append(sorted(list(part["population"].values())))
    cut_vec.append(len(part["cut_edges"]))
    mms.append([])
    egs.append([])
    hmss.append([])

    for elect in range(num_elections):
        votes[elect].append(sorted(part[election_names[elect]].percents("First")))
        mms[-1].append(mean_median(part[election_names[elect]]))
        egs[-1].append(efficiency_gap(part[election_names[elect]]))
        hmss[-1].append(part[election_names[elect]].wins("First"))

    t += 1
    if t%100 ==0:
        print(t)
    if t % 2000 == 0:
        print(t)
        with open(newdir + "mms" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(mms)

        with open(newdir + "egs" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(egs)

        with open(newdir + "hmss" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(hmss)

        with open(newdir + "pop" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(pop_vec)

        with open(newdir + "cuts" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows([cut_vec])

        with open(newdir + "assignment" + str(t) + ".json", "w") as jf1:
            json.dump(dict(part.assignment), jf1)

        for elect in range(num_elections):
            with open(
                newdir + election_names[elect] + "_" + str(t) + ".csv", "w"
            ) as tf1:
                writer = csv.writer(tf1, lineterminator="\n")
                writer.writerows(votes[elect])

        df["plot" + str(t)] = df.index.map(dict(part.assignment))
        df.plot(column="plot" + str(t), cmap="tab20")
        plt.savefig(newdir + "plot" + str(t) + ".png")
        plt.close()

        votes = [[], [], [], [],[],[]]
        mms = []
        egs = []
        hmss = []
        pop_vec = []
        cut_vec = []





