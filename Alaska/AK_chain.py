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

from FKT import FKT



whole_start = time.time()

#Initialization
df = gpd.read_file("./AK_precincts_ns/AK_precincts_ns/AK_precincts_ns.shp")
df["nAMIN"] = df["TOTPOP"]-df["AMIN"]
elections = [
    Election("GOV18x", {"Democratic": "GOV18D_x", "Republican": "GOV18R_x"}),
    Election("USH18x", {"Democratic": "USH18D_x", "Republican": "USH18R_x"}),
    Election("GOV18ns", {"Democratic": "GOV18D_NS", "Republican": "GOV18R_NS"}),
    Election("USH18ns", {"Democratic": "USH18D_NS", "Republican": "USH18R_NS"}),
    Election("Native_percent", {"Native":"AMIN", "nonNative":"nAMIN"})
]

my_updaters = {"population": updaters.Tally("POPULATION", alias="population"),
"cut_edges":cut_edges}

election_updaters = {election.name: election for election in elections}

my_updaters.update(election_updaters)



#Construct Dual Graphs



#Tightest Alaska
print("Building Tight Graph")

G_tight = Graph.from_file("./AK_precincts_ns/AK_precincts_ns/AK_precincts_ns.shp")
G_tight.join(df,columns= ["nAMIN"])

idict={}

for index, row in df.iterrows():

	idict[int(row["ID"])] = index

#Connect Islands
to_add = [(426,444), (437,438),(437,442),(411,420),
(411,414), (411,358),(411,407),(399,400),(399,349),(381,384),(240,210)]
	

for i in range(len(to_add)):
	G_tight.add_edge(idict[to_add[i][0]],idict[to_add[i][1]])
	G_tight[idict[to_add[i][0]]][idict[to_add[i][1]]]["shared_perim"]=1

#Separate Anchorage
to_remove = [(210,195),(210,203),(210,202),(210,193),(210,235),(210,234),(169,78),(169,77),(169,70),
(169,68),(169,32),(169,23),(169,179),(234,78),(235,78),(235,89),(235,106),(235,102),(102,190),
(190,105),(190,145),(145,233),(233,133),(234,169),(234,151),(77,74),(77,70)]

for i in range(len(to_remove)):
    G_tight.remove_edge(idict[to_remove[i][0]],idict[to_remove[i][1]])
    

print("Starting Ensemble Analysis")

ensemble_time = time.time()




GOV18ns= sorted((0.5187406297776442, 0.3916434540240218, 0.23378661089760477,
          0.561916167590201, 0.5277943813115976, 0.3967808623758262, 
          0.24347460005648527, 0.2040194040100815, 0.29961685822777134, 
          0.2658227848088675, 0.30213024956718243, 0.2738569188011496, 
          0.3331949346122295, 0.3753434474711785, 0.5018867924316115,
          0.5426127015341067, 0.5913152254553772, 0.6266881029630763,
          0.6404409922858867, 0.6744921745562409, 0.5829798514395329, 
          0.46457747422269396, 0.49254507629009764, 0.45721212122003807, 
          0.5081005584437817, 0.44071841251982957, 0.5078786300489014, 
          0.4823874755528126, 0.3030897498782328, 0.28720868644817754, 
          0.45392418577347565, 0.548494983265805, 0.6961661695141408, 
          0.5386310904517134, 0.5566274613453184, 0.4174338319909295, 
          0.6555631089965965, 0.7312614259593524, 0.7193151212004562, 
          0.61932265686532))
GOV18x=sorted((0.4974431818181818, 0.37424547283702214, 0.20789779326364694, 
        0.5263911254249418, 0.50988230068843, 0.38398798025327324, 
        0.23922875505831945, 0.19625226677412855, 0.28292410714285715, 
        0.2544677544677545, 0.28685985055585933, 0.2606104388658118, 
        0.32402732402732404, 0.3625, 0.47884788478847884, 0.5222623345367028,
        0.5623608017817372, 0.6011283497884344, 0.6125967628430683,
        0.6509209744503862, 0.5640572886011379, 0.45389537071885583, 
        0.4786472475931869, 0.447171453437772, 0.48655110579796773,
        0.42658509454949944, 0.49777777777777776, 0.46891624867001064, 
        0.28942840757025423, 0.27886435331230286, 0.40277539832105536, 
        0.525532969757065, 0.6567947910102919, 0.4908722109533469, 
        0.5396059509449136, 0.4065186962607478, 0.6617647058823529,
        0.7358445297504799, 0.7188718183902775, 0.6151213441194773))
USH18x=sorted((0.5058695058695059, 0.3900736719658782, 0.2543404735062007,
        0.5312662393902651, 0.5138918802498385, 0.37274049449407853, 
        0.2843413033286451, 0.2460960664162878, 0.3040479215828644, 
        0.3051849027830728, 0.32579668862382055, 0.2848743987172635,
        0.35922610453364134, 0.38727149627623564, 0.5108885017421603, 
        0.5274647887323943, 0.5599028602266595, 0.5989611809732094, 
        0.6076079506511309, 0.6491633006347375, 0.5707167497125335,
        0.46821448313985625, 0.4998794309139137, 0.43823479298006474, 
        0.49396417445482865, 0.4271950554444646, 0.49546608632571637, 
        0.467220409374073, 0.33054471091280907, 0.3152729503169086, 
        0.4366804979253112, 0.46370683579985905, 0.6810362274843149,
        0.5284996724175585, 0.5495891458054654, 0.44719662058371734,
        0.4837432852700028, 0.584951998213887, 0.5047358450852452,
        0.4665718349928876))
USH18ns=sorted((0.526629425384675, 0.41081640244677586, 0.2774311777926235, 
         0.5665135137823519, 0.5324396994583951, 0.39077610338199437,
         0.29063095741083556, 0.2520081528281857, 0.3242478870458404,
         0.3145534181033724, 0.3380180359170024, 0.2979472351935256,
         0.36280653131210194, 0.3964223985086792, 0.5198083711638748, 
         0.5483611707902181, 0.5889840870354133, 0.6223695336477948, 
         0.632669752041204, 0.6672778804198621, 0.5931440818852669,
         0.4755756282959946, 0.5105624482537662, 0.4545425262952549,
         0.5164939040837906, 0.4425673222996577, 0.5097079657012924,
         0.4870557891023724, 0.33723477574740685, 0.3260691986801825, 
         0.47925932440512, 0.4839707790207739, 0.7077419247658372, 
         0.5693768081345366, 0.5608357157934066, 0.45216772423949964,
         0.48482656936912155, 0.5846342120988194, 0.5154696428000252,
         0.47246020775337016))
Native=sorted((0.14453345368385423, 0.04791972037433758, 0.04153228088043909,
        0.04711570898459463, 0.09071032124236138, 0.19222777559386758,
        0.0498220640569395, 0.06539540100953449, 0.05580923389142567,
        0.0532899534414091, 0.0553172273650937, 0.04923320694923886, 
        0.03773051250141419, 0.034796273431361546, 0.07209144409234948,
        0.09912389082331799, 0.10816429735348654, 0.1110181311018131, 
        0.1408546235586706, 0.11965233096286262, 0.07601179004648, 
        0.07231765699802872, 0.09871413330338592, 0.05795955259292735, 
        0.08296139254630663, 0.0650539761487594, 0.07099219368706867,
        0.03138710766115423, 0.08620880949739265, 0.06442483768936241,
        0.04401535807690168, 0.16086740056425292, 0.13535582648142896,
        0.09395517319447588, 0.20246844319775595, 0.21248741188318226, 
        0.4281292984869326, 0.8349481363273681, 0.8371329976806019,
        0.6643768400392541))

types = ["tight"]
c='k'

for z in range(1):
    
    
    initial_partition = Partition(G_tight,assignment=df["HDIST"],updaters=my_updaters)
    
    
    
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    #print(ideal_population)
    
    proposal = partial(recom,
                       pop_col="POPULATION",
                       pop_target=ideal_population,
                       epsilon=0.05,
                       node_repeats=2
                      )
    
    
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]),
        2*len(initial_partition["cut_edges"])
    )
    
    chain = MarkovChain(
    proposal=proposal, 
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, .05),
        compactness_bound, #single_flip_contiguous#no_more_discontiguous
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=10000
        )

    print("Started chain")

    
    percents1 = []
    wins1 = []
    percents2 = []
    wins2 = []
    percents3 = []
    wins3 = []
    percents4 = []
    wins4 = []
    
    num_edges = []
    num_matchings = []
    
    wins5 = []
    percents5 = []
    
    maxe = 10
    mine = 11111110
    maxm = 10
    minm = 11111110

    maxea = {}
    minea = {}
    maxma = {}
    minma = {}
    maxeA = []
    mineA = []
    maxmA = []
    minmA = []    
    maxen = 0
    minen = 0
    maxmn = 0
    minmn = 0
    zerom = 0
    zeroA = np.matrix([[0,0],[0,0]])
    zerot = 0
    zeroe = 0

    t=0
    for c_part in chain:
        
        wins1.append(c_part["GOV18x"].wins("Democratic"))
        percents1.append(sorted(c_part["GOV18x"].percents("Democratic")))
        wins2.append(c_part["GOV18ns"].wins("Democratic"))
        percents2.append(sorted(c_part["GOV18ns"].percents("Democratic")))
        wins3.append(c_part["USH18x"].wins("Democratic"))
        percents3.append(sorted(c_part["USH18x"].percents("Democratic")))
        wins4.append(c_part["USH18ns"].wins("Democratic"))
        percents4.append(sorted(c_part["USH18ns"].percents("Democratic")))
        wins5.append(c_part["Native_percent"].wins("Native"))
        percents5.append(sorted(c_part["Native_percent"].percents("Native")))
        
        new_dg = nx.Graph()
        new_dg.add_edges_from(list({(c_part.assignment[x[0]],c_part.assignment[x[1]]) for x in c_part["cut_edges"]}) )
        A = nx.adjacency_matrix(new_dg).todense()
        
        num_edges.append(A.sum()/2)
        ans = FKT(A)
        if ans is not None:
            num_matchings.append(round(ans))
        else:
            num_matchings.append(0)

        if ans is None:
            ans = 0


        if A.sum()/2 > maxe:
            maxea = dict(c_part.assignment)
            maxeA = A[:]
            maxen = t
            maxe = A.sum()/2

        if (A.sum()/2) < mine:
            minea = dict(c_part.assignment)
            mineA = A[:]
            minen = t
            mine = A.sum()/2

        if round(ans) > maxm:
            maxma = dict(c_part.assignment)
            maxmA = A[:]
            maxmn = t
            maxm = round(ans)

            
        if round(ans) < minm  and round(ans) !=0:
            minma = dict(c_part.assignment)
            minmA = A[:]
            minmn = t
            minm = round(ans)

        if round(ans) == 0:
            zerom = dict(c_part.assignment)
            zeroA = A[:]
            zerot = t
            zeroe = A.sum()/2


        
        if t%1000 == 0:
            print(types[z],"chain ",t," steps")
        t+=1
        
    print("Finished ", types[z], " Ensemble")

    with open("./Outputs/Ensemble_"+types[z]+"_extremes.json",'w') as wf:
        json.dump({0:maxea,1:maxeA.tolist(),2:maxen,3:maxe,4:minea,5:mineA.tolist(),6:minen,7:mine,
8:maxma,9:maxmA.tolist(),10:maxmn,11:maxm,12:minma,13:minmA.tolist(),14:minmn,15:minm,16:zerom,
17:zeroA.tolist(),18:zerot,19:zeroe}, wf)

    
    
    partisan_w = [wins1,wins2,wins3,wins4]
    partisan_p = [percents1,percents2,percents3,percents4]
    p_types=["GOV18N", "GOV18A", "USH18N", "USH18A"]
    p_vecs=[GOV18x, GOV18ns, USH18x, USH18ns]
    
    for y in range(4):
        
    
        plt.figure()
        plt.boxplot(np.array(partisan_p[y]),whis=[1,99],showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor="None", color=c),
                        capprops=dict(color=c),
                        whiskerprops=dict(color=c),
                        flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color=c),
                        )
        
        plt.plot(range(1,41),p_vecs[y],'o',color='red',label='Current Plan')
        plt.plot([.5,41],[.5,.5],color='green',label="50%")
        plt.xlabel("Sorted Districts")
        plt.ylabel("Dem %")
        plt.xticks([1,20,40],['1','20','40'])
        plt.legend()
        
        fig = plt.gcf()
        fig.set_size_inches((20,10), forward=False)
        fig.savefig("./Outputs/plots/Ensemble_Box_"+ types[z] + p_types[y] + ".png")
        plt.close()
    
    
    
    plt.figure()
    plt.boxplot(np.array(percents5),whis=[1,99],showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="None", color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color=c),
                    )
    
    plt.plot(range(1,41),Native,'o',color='red',label='Current Plan')
    plt.plot([.5,41],[.5,.5],color='green',label="50%")
    plt.xlabel("Sorted Districts")
    plt.ylabel("Native %")
    plt.xticks([1,20,40],['1','20','40'])
    plt.legend()
    
    fig = plt.gcf()
    fig.set_size_inches((20,10), forward=False)
    fig.savefig("./Outputs/plots/Ensemble_Box_"+ types[z] +"Native.png")
    plt.close()
    
    print("Finished ",types[z]," Box plots")
    
    for y in range(4):
        plt.figure()
        sns.distplot(partisan_w[y],kde=False,color='slateblue',bins=[x for x in range(10,25)],
                                                        hist_kws={"rwidth":1,"align":"left"})
        plt.axvline(x=sum([val>.5 for val in p_vecs[y]]),color='r',label="Current Plan",linewidth=5)
        plt.axvline(x=np.mean(partisan_w[y]),color='g',label="Ensemble Mean",linewidth=5)
        plt.legend()
        print(p_types[y],"wins: ", np.mean(partisan_w[y]))
        plt.savefig("./Outputs/plots/Ensemble_Hist_"+ types[z] + p_types[y] + ".png")
        plt.close()
        
        
    plt.figure()
    sns.distplot(wins5,kde=False,color='slateblue',bins=[x for x in range(5)],
                                                    hist_kws={"rwidth":1,"align":"left"})
    plt.axvline(x=2,color='r',label="Current Plan",linewidth=5)
    plt.axvline(x=np.mean(wins5),color='g',label="Ensemble Mean",linewidth=5)
    plt.legend()
    print("Native wins: ", np.mean(wins5))
    plt.savefig("./Outputs/plots/Ensemble_Hist_"+ types[z] + "Native.png")
    plt.close()
    
    
    print("Finished ",types[z]," Seats plots")
    
    
    
    with open("./Outputs/values/Ensemble_" + types[z] + ".txt", "w") as f:
        f.write("Ensemble Values for Graph: "+types[z]+" \n\n")

        for y in range(4):
        
        
            f.write("Enacted Wins : "+ p_types[y] + ": "+ str(sum([val>.5 for val in p_vecs[y]])))
            f.write("\n")
            f.write("Ensemble Average Wins : "+ p_types[y] + ": "+ str(np.mean(partisan_w[y])))
            f.write("\n")
            f.write("\n")

    
    
    
    cis=[]
    ces=[]
    for y in range(4):
        votes = partisan_p[y]
        comp = []
        
        for i in range(len(votes)):
            temp = 0
            for j in votes[i]:
                if .4 < j < .6:
                    temp+=1
            comp.append(temp)
        
        c_init = 0
        
        for x in p_vecs[y]:
            if .4 < x <.6:
                c_init += 1
                
        cis.append(c_init)
                
        
        sns.distplot(comp,kde=False,color='slateblue',bins=[x for x in range(15,35)],
                                                        hist_kws={"rwidth":1,"align":"left"})
        plt.axvline(x=c_init,color='r',label="Current Plan",linewidth=5)
        plt.axvline(x=np.mean(comp),color='g',label="Ensemble Mean",linewidth=5)
        print(p_types[y],"competitive: ",np.mean(comp))
        plt.legend()
        plt.savefig("./Outputs/plots/Ensemble_Comp_"+ types[z] + p_types[y] + ".png")
        plt.close()
        
        ces.append(np.mean(comp))


    with open("./Outputs/values/Ensemble_Comp" + types[z] + ".txt", "w") as f:
        f.write("Ensemble Values for Graph: "+types[z]+" \n\n")

        for y in range(4):
        
        
            f.write("Enacted Comp : "+ p_types[y] + ": "+ str(cis[y]))
            f.write("\n")
            f.write("Ensemble Average Comp : "+ p_types[y] + ": "+ str(ces[y]))
            f.write("\n")
            f.write("\n")    
    votes = percents5
    comp = []
    
    for i in range(len(votes)):
        temp = 0
        for j in votes[i]:
            if .4 < j < .6:
                temp+=1
        comp.append(temp)
    
    sns.distplot(comp,kde=False,color='slateblue',bins=[x for x in range(6)],
                                                    hist_kws={"rwidth":1,"align":"left"})
    plt.axvline(x=0,color='r',label="Current Plan",linewidth=5)
    plt.axvline(x=np.mean(comp),color='g',label="Ensemble Mean",linewidth=5)
    print("Native competitive: ", np.mean(comp))
    plt.legend()
    plt.savefig("./Outputs/plots/Ensemble_Comp_"+ types[z] + "Native.png")
    plt.close()
    
    print("Finished ",types[z]," Competitive plots")
    
    plt.figure()
    plt.plot(num_edges,num_matchings,'o',markersize=1)
    plt.savefig("./Outputs/plots/Ensemble_Comp_"+ types[z] + "edges.png")
    plt.close()
    
    
    
    with open("./Outputs/Ensemble_"+types[z]+"_stats.json",'w') as wf:
        json.dump({0:wins1,1:percents1,2:wins2,3:percents2,4:wins3,5:percents3,
                   6:wins4,7:percents4,8:wins5,9:percents5,10:num_edges,11:num_matchings}, wf)

    
    

