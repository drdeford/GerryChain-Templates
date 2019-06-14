import matplotlib.pyplot as plt

plt.switch_backend("agg")
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import json
import networkx as nx
import time
import os
import csv


from gerrychain import Graph, Partition, Election
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain import (
    GeographicPartition,
    Partition,
    Graph,
    MarkovChain,
    proposals,
    updaters,
    constraints,
    accept,
    Election,
)
from gerrychain.constraints.validity import within_percent_of_ideal_population
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.proposals import recom
from functools import partial

newdir = "./Outputs/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")


graph_path = "./Data/PA_VTDALL.json"  # "./Data/PA_BPOP_FINAL/VTD_FINAL.shp"
plot_path = "./Data/VTD_FINAL.shp"


df = gpd.read_file(plot_path)


def num_splits(partition):

    df["current"] = df[unique_label].map(dict(partition.assignment))

    splits = sum(df.groupby("COUNTYFP10")["current"].nunique() > 1)

    return splits


unique_label = "GEOID10"
pop_col = "TOT_POP"
district_col = "2011_PLA_1"
county_col = "COUNTYFP10"

num_elections = 14


election_names = [
    "BPOP",
    "ATG12",
    "GOV14",
    "GOV10",
    "PRES12",
    "SEN10",
    "ATG16",
    "PRES16",
    "SEN16",
    "SEN12",
    "SENW1012",
    "SENW1016",
    "SENW101216",
    "SENW1216",
]
election_columns = [
    ["BPOP", "nBPOP"],
    ["ATG12D", "ATG12R"],
    ["F2014GOVD", "F2014GOVR"],
    ["GOV10D", "GOV10R"],
    ["PRES12D", "PRES12R"],
    ["SEN10D", "SEN10R"],
    ["T16ATGD", "T16ATGR"],
    ["T16PRESD", "T16PRESR"],
    ["T16SEND", "T16SENR"],
    ["USS12D", "USS12R"],
    ["W1012D", "W1012R"],
    ["W1016D", "W1016R"],
    ["W101216D", "W101216R"],
    ["W1216D", "W1216R"],
]


graph = Graph.from_json(graph_path)


updaters = {
    "population": updaters.Tally("TOT_POP", alias="population"),
    "cut_edges": cut_edges,
}

elections = [
    Election(
        election_names[i],
        {"Democratic": election_columns[i][0], "Republican": election_columns[i][1]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}

updaters.update(election_updaters)


initial_partition = Partition(graph, "2011_PLA_1", updaters)


ideal_population = sum(initial_partition["population"].values()) / len(
    initial_partition
)
# print(ideal_population)

proposal = partial(
    recom, pop_col="TOT_POP", pop_target=ideal_population, epsilon=0.02, node_repeats=2
)

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, 0.02),
        compactness_bound,  # single_flip_contiguous#no_more_discontiguous
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000,
)


with open(newdir + "Start_Values.txt", "w") as f:
    f.write("Values for Starting Plan: 2011 Enacted\n \n ")
    f.write("Initial Cut: " + str(len(initial_partition["cut_edges"])))
    f.write("\n")
    f.write("\n")
    f.write("Initial County Splits: " + str(num_splits(initial_partition)))
    f.write("\n")
    f.write("\n")

    for elect in range(num_elections):
        f.write(
            election_names[elect]
            + "District Percentages"
            + str(
                sorted(initial_partition[election_names[elect]].percents("Democratic"))
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
            + str(initial_partition[election_names[elect]].wins("Democratic"))
        )

        f.write("\n")
        f.write("\n")


pop_vec = []
cut_vec = []
votes = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
mms = []
egs = []
hmss = []

splits = []

t = 0
for part in chain:
    splits.append(num_splits(part))

    pop_vec.append(sorted(list(part["population"].values())))
    cut_vec.append(len(part["cut_edges"]))
    mms.append([])
    egs.append([])
    hmss.append([])

    for elect in range(num_elections):
        votes[elect].append(sorted(part[election_names[elect]].percents("Democratic")))
        mms[-1].append(mean_median(part[election_names[elect]]))
        egs[-1].append(efficiency_gap(part[election_names[elect]]))
        hmss[-1].append(part[election_names[elect]].wins("Democratic"))

    t += 1
    if t % 100 == 0:
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

        with open(newdir + "splits" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows([splits])

        with open(newdir + "assignment" + str(t) + ".json", "w") as jf1:
            json.dump(dict(part.assignment), jf1)

        for elect in range(num_elections):
            with open(
                newdir + election_names[elect] + "_" + str(t) + ".csv", "w"
            ) as tf1:
                writer = csv.writer(tf1, lineterminator="\n")
                writer.writerows(votes[elect])

        df["plot" + str(t)] = df["GEOID10"].map(dict(part.assignment))
        df.plot(column="plot" + str(t), cmap="tab20")
        plt.savefig(newdir + "plot" + str(t) + ".png")
        plt.close()

        votes = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
        mms = []
        egs = []
        hmss = []
        pop_vec = []
        cut_vec = []
        splits = []
