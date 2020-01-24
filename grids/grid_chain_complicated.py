# Import for I/O

import random

# import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import math
from functools import partial
import seaborn as sns

# Imports for GerryChain components
# You can look at the list of available functions in each
# corresponding .py file.


from gerrychain.proposals import recom
from gerrychain import MarkovChain
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.accept import always_accept
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
import networkx as nx


def create_graph():
    graph = nx.grid_graph([k * gn, k * gn])

    for e in graph.edges():
        graph[e[0]][e[1]]["shared_perim"] = 1

    graph.remove_nodes_from(
        [(0, 0), (0, k * gn - 1), (k * gn - 1, 0), (k * gn - 1, k * gn - 1)]
    )

    cddict = {x: int(x[0] / gn) for x in graph.nodes()}

    for n in graph.nodes():
        graph.nodes[n]["population"] = 1
        graph.nodes[n]["part_sum"] = cddict[n]
        graph.nodes[n]["last_flipped"] = 0
        graph.nodes[n]["num_flips"] = 0

        if random.random() < p:
            graph.nodes[n]["pink"] = 1
            graph.nodes[n]["purple"] = 0
        else:
            graph.nodes[n]["pink"] = 0
            graph.nodes[n]["purple"] = 1
        if 0 in n or k * gn - 1 in n:
            graph.nodes[n]["boundary_node"] = True
            graph.nodes[n]["boundary_perim"] = 1

        else:
            graph.nodes[n]["boundary_node"] = False

    # this part adds queen adjacency
    # for i in range(gn-1):
    #    for j in range(gn):
    #        if j<(gn-1):
    #            graph.add_edge((i,j),(i+1,j+1))
    #            graph[(i,j)][(i+1,j+1)]["shared_perim"]=0
    #        if j >0:
    #            graph.add_edge((i,j),(i+1,j-1))
    #            graph[(i,j)][(i+1,j-1)]["shared_perim"]=0

    return graph, cddict


def step_num(partition):
    parent = partition.parent
    if not parent:
        return 0
    return parent["step_num"] + 1


def slow_reversible_propose(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting a boundary node at random and uniformly picking one of its
    neighboring parts.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    b_nodes = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )

    flip = random.choice(list(b_nodes))
    neighbor_assignments = list(
        set(
            [
                partition.assignment[neighbor]
                for neighbor in partition.graph.neighbors(flip)
            ]
        )
    )
    neighbor_assignments.remove(partition.assignment[flip])

    flips = {flip: random.choice(neighbor_assignments)}

    return partition.flip(flips)


def reversible_propose(partition):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )

    flip = random.choice(list(boundaries1))
    return partition.flip({flip: -partition.assignment[flip]})


def cut_accept(partition):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )
    boundaries2 = {x[0] for x in partition.parent["cut_edges"]}.union(
        {x[1] for x in partition.parent["cut_edges"]}
    )

    bound = 1
    if partition.parent is not None:
        bound = (
            base ** (-len(partition["cut_edges"]) + len(partition.parent["cut_edges"]))
        ) * (len(boundaries1) / len(boundaries2))

    return random.random() < bound


def annealing_cut_accept(partition, t):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )
    boundaries2 = {x[0] for x in partition.parent["cut_edges"]}.union(
        {x[1] for x in partition.parent["cut_edges"]}
    )

    if t < 50000:
        beta = 0
    elif t < 200000:
        beta = (t - 50000) / 50000
    else:
        beta = 3
    bound = 1
    if partition.parent is not None:
        bound = (
            base
            ** (
                beta
                * (-len(partition["cut_edges"]) + len(partition.parent["cut_edges"]))
            )
        ) * (len(boundaries1) / len(boundaries2))
        # bound = min(1, (how_many_seats_value(partition, col1="G17RATG",
        # col2="G17DATG")/how_many_seats_value(partition.parent, col1="G17RATG",
        # col2="G17DATG"))**2  )
        # for some states/elections probably want to add 1 to denominator so
        # you don't divide by zero

    return random.random() < bound


def annealing_cut_accept2(partition):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )
    boundaries2 = {x[0] for x in partition.parent["cut_edges"]}.union(
        {x[1] for x in partition.parent["cut_edges"]}
    )

    t = partition["step_num"]

    if t < 100000:
        beta = 0
    elif t < 400000:
        beta = (t - 100000) / 100000  # was 50000)/50000
    else:
        beta = 3

    bound = 1
    if partition.parent is not None:
        exponent = beta * (
            -len(partition["cut_edges"]) + len(partition.parent["cut_edges"])
        )
        bound = (base ** (exponent)) * (len(boundaries1) / len(boundaries2))

        # maybe_bound = (
        #     how_many_seats_value(partition, col1="G17RATG", col2="G17DATG")
        #     / how_many_seats_value(partition.parent, col1="G17RATG", col2="G17DATG")
        # ) ** 2
        # bound = min(1, maybe_bound)
        # for some states/elections probably want to add 1 to denominator
        # so you don't divide by zero

    return random.random() < bound


gn = 10
k = 5
ns = 200
p = 0.4

for exp_num in [40, 20, 1]:  # range(22,31):

    for pop_bal in [5, 10, 50]:  # [10,15,20,25,30,35,40,45,50]
        base = exp_num / 10  # 1/math.pi
        graph, cddict = create_graph()

        # Necessary updaters go here
        updaters = {
            "population": Tally("population"),
            "cut_edges": cut_edges,
            "step_num": step_num,
        }
        # updaters={}

        # election_updaters=dict()
        # election_updaters["Pink-Purple"]=Election("Pink-Purple",{"Pink":"pink","Purple":"purple"},alias="Pink-Purple")

        gp3 = Partition(graph, assignment=cddict, updaters=updaters)

        ideal_population = sum(gp3["population"].values()) / len(gp3)

        proposal = partial(
            recom,
            pop_col="population",
            pop_target=ideal_population,
            epsilon=0.02,
            node_repeats=1,
        )

        popbound = within_percent_of_ideal_population(gp3, 0.05)

        g3chain = MarkovChain(
            proposal,  # propose_chunk_flip,
            Validator([popbound]),
            accept=always_accept,
            initial_state=gp3,
            total_steps=100,
        )

        t = 0
        for part3 in g3chain:
            t += 1

        print("finished tree walk")

        pos_dict = {n: n for n in graph.nodes()}
        pos = pos_dict

        plt.figure()
        plt.title("Starting Point")
        nx.draw(
            graph,
            pos,
            node_color=[part3.assignment[x] for x in graph.nodes()],
            node_size=ns,
            node_shape="s",
            cmap="tab20",
        )
        plt.title("Starting Point")
        plt.savefig(
            "./plots/complicated/start_exp_"
            + str(exp_num)
            + "_"
            + str(pop_bal)
            + "pop.png"
        )
        plt.close()

        gp = Partition(graph, assignment=dict(part3.assignment), updaters=updaters)

        popbound = within_percent_of_ideal_population(gp, pop_bal / 100)

        gchain = MarkovChain(
            slow_reversible_propose,  # propose_random_flip,#propose_chunk_flip, # ,
            Validator([single_flip_contiguous, popbound]),
            accept=annealing_cut_accept2,  # aca,#cut_accept,#always_accept,#
            initial_state=gp,
            total_steps=500000,
        )

        pos_dict = {n: n for n in graph.nodes()}
        pos = pos_dict

        ce_hist = []
        t = 0
        cuts = []
        for part in gchain:

            cuts.append(len(part["cut_edges"]))
            if part.flips is not None:
                f = list(part.flips.keys())[0]
                # graph.node[f]["part_sum"]=graph.node[f]["part_sum"]-part.assignment[f]*(t-graph.node[f]["last_flipped"])
                # graph.node[f]["last_flipped"]=t
                graph.node[f]["num_flips"] = graph.node[f]["num_flips"] + 1

            # mm_hist.append(mean_median(part["Pink-Purple"]))
            # ce_hist.append(len(part["cut_edges"]))
            # print(len(part["cut_edges"]))
            # if t % 200 == 0:
            #     plt.figure()
            #     nx.draw(
            #         graph,
            #         pos,
            #         node_color=[part.assignment[x] for x in graph.nodes()],
            #         node_size=ns,
            #     )  # ,cmap="tab20")
            #     # plt.savefig("./plots/GRIDn_"+str(int(t/1000))+".png")
            #     plt.show()
            t += 1
            if t % 50000 == 0:
                print(t)
                plt.figure()
                plt.title(str(t) + "Steps")
                nx.draw(
                    graph,
                    pos,
                    node_color=[part.assignment[x] for x in graph.nodes()],
                    node_size=ns,
                    node_shape="s",
                    cmap="tab20",
                )
                plt.title("Ending Point")
                plt.savefig(
                    "./plots/complicated/middle"
                    + str(t)
                    + "2_"
                    + str(exp_num)
                    + "_"
                    + str(pop_bal)
                    + "pop.png"
                )
                plt.close()
            # print(t)

        for n in graph.nodes():
            if graph.node[n]["last_flipped"] == 0:
                graph.node[n]["part_sum"] = t * part.assignment[n]
            graph.node[n]["num_flips"] = math.log(graph.node[n]["num_flips"] + 1)
        print("finished flip")

        # plt.figure()
        # plt.title("Starting Point")
        # nx.draw(
        #     graph,
        #     pos,
        #     node_color=[cddict[x] for x in graph.nodes()],
        #     node_size=ns,
        #     cmap="tab20",
        # )
        # plt.title("Starting Point")
        # plt.show()

        plt.figure()
        plt.title("Ending Point")
        nx.draw(
            graph,
            pos,
            node_color=[part.assignment[x] for x in graph.nodes()],
            node_size=ns,
            node_shape="s",
            cmap="tab20",
        )
        plt.title("Ending Point")
        plt.savefig(
            "./plots/complicated/newend2_"
            + str(exp_num)
            + "_"
            + str(pop_bal)
            + "pop.png"
        )
        # plt.show()

        # plt.figure()
        # plt.title("Weighted Community Assignment")
        # nx.draw(
        #     graph,
        #     pos,
        #     node_color=[graph.nodes[x]["part_sum"] for x in graph.nodes()],
        #     node_size=ns,
        #     node_shape="s",
        #     cmap="jet",
        # )
        # plt.title("Weighted Community Assignment")
        # plt.savefig(".//plotswca2_" + str(exp_num) + "_" + str(pop_bal) + "pop.png")
        # # plt.show()

        plt.figure()
        plt.title("Flips")
        nx.draw(
            graph,
            pos,
            node_color=[graph.nodes[x]["num_flips"] for x in graph.nodes()],
            node_size=ns,
            node_shape="s",
            cmap="jet",
        )
        plt.title("Flips")

        plt.savefig(
            "./plots/complicated/flips_" + str(exp_num) + "_" + str(pop_bal) + "pop.png"
        )
        plt.close()

        plt.figure()
        plt.title("Cut Lengths")
        plt.plot(cuts)

        plt.savefig(
            "./plots/complicated/cuts_" + str(exp_num) + "_" + str(pop_bal) + "pop.png"
        )
        plt.close()

        plt.figure()
        plt.title("Cut Lengths")
        sns.distplot(cuts, bins=100, kde=False)

        plt.savefig(
            "./plots/complicated/cuthist_"
            + str(exp_num)
            + "_"
            + str(pop_bal)
            + "pop.png"
        )
        plt.close()
