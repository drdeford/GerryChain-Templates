import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import json
import networkx as nx
import time 

from gerrychain import Graph, Partition, Election
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.constraints.validity import within_percent_of_ideal_population

from gerrychain.proposals import recom
from functools import partial

newdir = "./Outputs/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")
    

graph_path =  "./Data/PA_VTDALL.json"#"./Data/PA_BPOP_FINAL/VTD_FINAL.shp"
plot_path = "./Data/VTD_FINAL.shp


df = gp.read_file(plot_path) 


def num_splits(partition):


    df["current"] = df[unique_label].map(partition.assignment)
    

    splits = sum(df.groupby('COUNTYFP10')['current'].nunique() >1)
    


    return splits
    
    
    
unique_label = "GEOID10"
pop_col = "TOT_POP"
district_col = "2011_PLA_1"
county_col =  "COUNTYFP10"

num_elections = 14


election_names =["BPOP","ATG12","GOV14","GOV10","PRES12","SEN10","ATG16","PRES16","SEN16","SEN12","SENW1012","SENW1016","SENW101216","SENW1216"]
election_columns = [["BPOP","nBPOP"],["ATG12D","ATG12R"],["F2014GOVD","F2014GOVR"],["GOV10D","GOV10R"],
["PRES12D","PRES12R"],["SEN10D","SEN10R"],["T16ATGD","T16ATGR"],["T16PRESD","T16PRESR"],
["T16SEND","T16SENR"],["USS12D","USS12R"],["W1012D","W1012R"],["W1016D","W1016R"],["W101216D","W101216R"],["W1216D","W1216R"]]


graph = Graph.from_json(graph_path)


updaters = {"population": updaters.Tally("TOT_POP", alias="population"),
            "cut_edges": cut_edges}
            
            
for i in range(num_elections):
    updaters.update(Election(election_names[i],{"Democratic":election_columns[i][0],"Republican":election_columns[i][1]}))
    

initial_partition = Partition(graph, "2011_PLA_1", updaters)


    
    
    
ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    #print(ideal_population)
    
proposal = partial(recom,
                       pop_col="TOT_POP",
                       pop_target=ideal_population,
                       epsilon=0.05,
                       node_repeats=2
                      )


chain = MarkovChain(
proposal=proposal, 
constraints=[
    constraints.within_percent_of_ideal_population(initial_partition, .02),
    compactness_bound, #single_flip_contiguous#no_more_discontiguous
],
accept=accept.always_accept,
initial_state=initial_partition,
total_steps=10000
    )
    
    
for part in chain:
    pass

