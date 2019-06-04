import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
#sns.set_style('darkgrid')
sns.set_style("darkgrid", {"axes.facecolor": ".97"})
#sns.set_style('white')
import json
import os


num_elections = 14
election_names =["BPOP","ATG12","GOV14","GOV10","PRES12","SEN10","ATG16","PRES16","SEN16","SEN12","SENW1012","SENW1016","SENW101216","SENW1216"]
election_columns = [["BPOP","nBPOP"],["ATG12D","ATG12R"],["F2014GOVD","F2014GOVR"],["GOV10D","GOV10R"],
["PRES12D","PRES12R"],["SEN10D","SEN10R"],["T16ATGD","T16ATGR"],["T16PRESD","T16PRESR"],
["T16SEND","T16SENR"],["USS12D","USS12R"],["W1012D","W1012R"],["W1016D","W1016R"],["W101216D","W101216R"],["W1216D","W1216R"]]


plan_name = "2011 Enacted"

newdir = "./Plots/"

num_districts = 18


datadir = "./Outputs/" 


os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")
    

max_steps = 1000
step_size = 100 

ts = [x*step_size for x in range(1,int(max_steps/step_size)+1)]



###for burn in###

ts.pop(0)


cuts = np.zeros([1,max_steps-step_size])
splits = np.zeros([1,max_steps-step_size])

for t in ts:
    temp = np.loadtxt(datadir+"cuts"+str(t)+".csv", delimiter=',')
    cuts[0,t-step_size-step_size:t-step_size]=temp
    temp = np.loadtxt(datadir+"splits"+str(t)+".csv", delimiter=',')
    splits[0,t-step_size-step_size:t-step_size]=temp

    
    

plt.plot(cuts[0,:])
#plt.plot([0,max_steps],[cuts[0,0],cuts[0,0]],color='orange',label="Initial Value")
plt.plot([0,max_steps],[np.mean(cuts[0,:]),np.mean(cuts[0,:])],color='g',label="Ensemble Mean")
#plt.title(plan_name + "Plan")
plt.axhline(y=1402,color='orange',label="Remedial")
plt.axhline(y=2361,color='purple',label="2011")
plt.ylabel("# Cut Edges")
plt.xlabel("Step")
#plt.plot([],[],color='green',label='Ensemble Mean')
#plt.plot([],[],color='red',label='Initial Value')
plt.legend()
plt.savefig(newdir+"cut_trace.png")
plt.close()


#plt.hist(cuts[0,:],bins=1000)
sns.distplot(cuts[0,:], kde=False,color='gray')

#plt.title(plan_name + "Plan")
plt.axvline(x=1402,color='orange',label='Remedial')
plt.axvline(x=2361,color='purple',label='2011')
plt.axvline(x=np.mean(cuts[0,:]),color='g',label='Ensemble Mean')
plt.legend()
#plt.xlim([400,800])
plt.ylabel("Frequency")
plt.xlabel("# Cut Edges")
plt.savefig(newdir+"cut_hist.png")
plt.close()


plt.plot(splits[0,:])
plt.plot([0,max_steps],[np.mean(splits[0,:]),np.mean(splits[0,:])],color='g',label="Ensemble Mean")
#plt.title(plan_name + "Plan")
plt.ylabel("# Counties Split")
plt.axhline(y=14,color='orange',label="Remedial")
plt.axhline(y=29,color='purple',label="2011")
plt.xlabel("Step")
#plt.plot([],[],color='green',label='Ensemble Mean')
#plt.plot([],[],color='red',label='Initial Value')
plt.legend()
plt.savefig(newdir+"split_trace.png")
plt.close()


sns.distplot(splits[0,:], kde=False,color='gray')

#plt.title(plan_name + "Plan")
plt.axvline(x=14,color='orange',label='Remedial')
plt.axvline(x=29,color='purple',label='2011')
plt.axvline(x=np.mean(splits[0,:]),color='g',label='Ensemble Mean')
plt.legend()
#plt.xlim([0,70])
plt.ylabel("Frequency")
plt.xlabel("# Counties Split")
plt.savefig(newdir+"split_hist.png")
plt.close()


egs = np.zeros([14,max_steps-step_size])
hmss = np.zeros([14,max_steps-step_size])
mms = np.zeros([14,max_steps-step_size])
dlocs = np.zeros([14,max_steps-step_size])
#pops = np.zeros([7,max_steps])

for t in ts:
    temp = np.loadtxt(datadir+"egs"+str(t)+".csv", delimiter=',')
    egs[:,t-step_size-step_size:t-step_size]=temp.T
    temp = np.loadtxt(datadir+"hmss"+str(t)+".csv", delimiter=',')
    hmss[:,t-step_size-step_size:t-step_size]=temp.T
    temp = np.loadtxt(datadir+"mms"+str(t)+".csv", delimiter=',')
    mms[:,t-step_size-step_size:t-step_size]=temp.T

    	    #temp = np.loadtxt(datadir+"pop"+str(t)+".csv", delimiter=',')
    #pops[:,t-step_size:t]=temp    


rmms = [0.05523296978114108,0.037755720134190485,0.022865694023377126,0.02045994014559044,
0.027780520455162827, 0.0203570360532751, 0.002851480104821502, 0.0197426645358596,
0.008348358442480441, 0.020906187147143207, 0.021019699370123046, 0.014695362060662354,
0.016901042588589488, 0.01472298172097497]
regs = [-0.33848762574863445,-0.04108421813228032,0.030339286105926337,0.12745685566177842, 0.032846877782017045, 0.14565735958500176, -0.04775419977387685, 0.026328638043364935, 0.19224026731762833, 0.009367255236277123, 0.08777331514254728, 0.14436828519047082,
0.1268419060455173,0.019537723548940532]

rhmss = [1,12, 12, 5, 9, 6, 10, 8, 5, 10, 8, 6, 7, 9]



emms = [0.061302087051234334,0.06040256110829312,0.05197222760489917,0.06615168261534632,
0.04554182457794509,0.06234727304937371, 0.04990132386630031, 0.03802944348266313,
0.043177937983041204, 0.05288069247284355, 0.06068591542234092, 0.06281202018775361,
0.06025126999574559, 0.04832454701108302]
eegs = [-0.3384824227238481, -0.0953240274115086, 0.1360385146810817, 0.1961113018562784, 
0.2738257238543389, 0.21595555460446897, 0.19381191844363213, 0.14296721837629872, 
0.25941828046688337, 0.07176053351222789, 0.26526082164253695, 0.21362333856059532,
0.2472709219339319, 0.262928605598663]
ehmss = [1,13,8,4,5,5,6,6,4,9,5,5,5,5]



for j in range(14):



    plt.plot(egs[j,:])
    plt.plot([0,max_steps],[eegs[j],eegs[j]],color='purple',label="2011")
    plt.plot([0,max_steps],[regs[j],regs[j]],color='orange',label="Remedial")
    plt.plot([0,max_steps],[np.mean(egs[j,:]),np.mean(egs[j,:])],color='g',label="Ensemble Mean")
    plt.title(plan_name + "Plan" + election_names[j])
    #plt.axhline(y=.104,color='orange',label="SM-Cong")
    plt.ylabel("Efficiency Gap")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig(newdir+election_names[j]+"eg_trace.png")
    plt.close()


    sns.distplot(np.negative(egs[j,:]),bins=1000, kde=False,color='gray')
    plt.title(plan_name + "Plan" + election_names[j])
    plt.axvline(x=-eegs[j],color='purple',label='2011')
    plt.axvline(x=-regs[j],color='orange',label='Remedial')
    plt.axvline(x=-np.mean(egs[j,:]),color='g',label='Ensemble Mean')
    plt.ylabel("Frequency")
    plt.xlabel("Efficiency Gap")
    #plt.xlim([-.15,.25])
    #plt.plot([],[],color='green',label='Ensemble Mean')
    #plt.plot([],[],color='red',label='Initial Value')
    plt.legend()
    plt.savefig(newdir+election_names[j]+"eg_hist.png")

    plt.close()



    plt.plot(mms[j,:])
    plt.plot([0,max_steps],[emms[j],emms[j]],color='purple',label="2011")
    plt.plot([0,max_steps],[rmms[j],rmms[j]],color='orange',label="Remedial")
    plt.plot([0,max_steps],[np.mean(mms[j,:]),np.mean(mms[j,:])],color='g',label="Ensemble Mean")
    #plt.title(plan_name + "Plan" + election_names[j])
    #plt.axhline(y=.032 ,color='orange',label="SM-Cong")
    plt.ylabel("Mean-Median")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig(newdir+election_names[j]+"mm_trace.png")
    plt.close()


    sns.distplot(np.negative(mms[j,:]),bins=400, kde=False,color='gray')
    plt.title(plan_name + "Plan" + election_names[j])
    plt.axvline(x=-emms[j],color='purple',label='2011')
    plt.axvline(x=-rmms[j],color='orange',label='Remedial')
    plt.axvline(x=-np.mean(mms[j,:]),color='g',label='Ensemble Mean')
    plt.ylabel("Frequency")
    plt.xlabel("Mean-Median")
    #plt.xlim([-.06,.06])
    #plt.plot([],[],color='green',label='Ensemble Mean')
    #plt.plot([],[],color='red',label='Initial Value')
    plt.legend()
    plt.savefig(newdir+election_names[j]+"mm_hist.png")
    plt.close()
    
    plt.plot(hmss[j,:])
    plt.plot([0,max_steps],[ehmss[j],ehmss[j]],color='purple',label="2011")
    plt.plot([0,max_steps],[rhmss[j],rhmss[j]],color='orange',label="Remedial")
    plt.plot([0,max_steps],[np.mean(hmss[j,:]),np.mean(hmss[j,:])],color='g',label="Ensemble Mean")
    #plt.axhline(y=hmss[j,0],color='orange',label="SM-Cong")
    #plt.title(plan_name + "Plan" + election_names[j])
    plt.ylabel("# Dem Seats")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig(newdir+election_names[j]+"seats_trace.png")
    plt.close()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)    
    #sns.distplot(hmss[j,:],kde=False,color='gray',bins=list(range(4,10)),hist_kws={"rwidth":.8,"align":"left"})
    sns.distplot(hmss[j,:],kde=False,color='gray')
    plt.title(plan_name + "Plan" + election_names[j])
    plt.axvline(x=ehmss[j],color='purple',label='2011')
    plt.axvline(x=rhmss[j],color='orange',label='Remedial')
    plt.axvline(x=np.mean(hmss[j,:]),color='g',label='Ensemble Mean')
    #plt.plot([],[],color='green',label='Ensemble Mean')
    #plt.plot([],[],color='red',label='Initial Value')
    #plt.xlim([4,10]) 
    #plt.xticks(list(range(4,11)))
    #ax1.set_xticklabels([str(x) for x in range(4,11)])
    plt.ylabel("Frequency")
    plt.xlabel("# Dem Seats")
    plt.legend()
    plt.savefig(newdir+election_names[j]+"seats_hist.png")
    plt.close()
    
    


with open(newdir + "Average_Values.txt", "w") as f:
    f.write("Values for Starting Plan: "+plan_name+" \n\n")

    f.write("\n")
    f.write("\n")
    for elect in range(num_elections):
        print(elect)
        

        f.write(election_names[elect] + " Initial Mean-Median: "+ str(mms[elect,0]))
        
        f.write("\n")
        f.write("\n")
        f.write(election_names[elect] + " Average Mean-Median: "+ str(np.mean(mms[elect,:])))
        
        f.write("\n")
        f.write("\n")
        f.write(election_names[elect] + " Number Mean-Median Higher: "+ str((mms[elect,:]>mms[elect,0]).sum()))
        
        f.write("\n")
        f.write("\n")
         
        f.write("\n")
        f.write("\n")
        
        
        f.write(election_names[elect] + " Initial Efficiency Gap: "+ str(egs[elect,0]))
        
        f.write("\n")
        f.write("\n")
        f.write(election_names[elect] + " Average Efficiency Gap: "+ str(np.mean(egs[elect,:])))
        
        f.write("\n")
        f.write("\n")
        f.write(election_names[elect] + " Number Efficiency Gap Higher: "+ str((egs[elect,:]>egs[elect,0]).sum()))
        
        f.write("\n")
        f.write("\n")
         
        f.write("\n")
        f.write("\n")
        
        
        f.write(election_names[elect] + " Initial Dem Seats: "+ str(hmss[elect,0]))
        
        f.write("\n")
        f.write("\n")
        f.write(election_names[elect] + " Average EDem Seats: "+ str(np.mean(hmss[elect,:])))
        
        f.write("\n")
        f.write("\n")
        f.write(election_names[elect] + " Number Dem Seats Higher: "+ str((hmss[elect,:]>hmss[elect,0]).sum()))
        
        f.write("\n")
        f.write("\n")
         
        f.write("\n")
        f.write("\n")
        
a=[] #np.zeros([max_steps,40])

for t in ts:#
     temp=np.loadtxt(datadir+"pop"+str(t)+".csv", delimiter=',')
     for s in range(step_size):
            a.append(temp[s,:])

a=np.array(a)
mpop=np.mean(a[0,:])

mean_dev=[]
l1pop=[]
maxpop=[]
minpop=[]


for i in range(max_steps-step_size):
    mean_dev.append(sum([abs(mpop-x)/mpop for x in a[i,:]])/40)
    l1pop.append(sum([abs(mpop-x) for x in a[i,:]]))
    maxpop.append(max([abs(mpop-x)/mpop for x in a[i,:]]))
    minpop.append(min([abs(mpop-x)/mpop for x in a[i,:]]))



sns.distplot(mean_dev,bins=100, kde=False,color='gray')
#plt.title(plan_name + "Plan" + election_names[j])
#plt.axvline(x=.0064,color='orange',label='SM-Con')
plt.axvline(x=np.mean(mean_dev),color='g',label='Ensemble Mean')
plt.ylabel("Frequency")
plt.xlabel("Average Percentage Population Deviation")

#plt.title("Population Deviation")
plt.savefig(newdir+"meanpop.png")

plt.close()


sns.distplot(l1pop,bins=100, kde=False,color='gray')
#plt.title(plan_name + "Plan" + election_names[j])
#plt.axvline(x=51432,color='orange',label='SM-Con')
plt.axvline(x=np.mean(l1pop),color='g',label='Ensemble Mean')
plt.ylabel("Frequency")
plt.xlabel("Total Population Deviation")

#plt.title("Population Deviation")
plt.savefig(newdir+"l1pop.png")

plt.close()
sns.distplot(maxpop,bins=100, kde=False,color='gray')
#plt.title(plan_name + "Plan" + election_names[j])
plt.axvline(x=.026,color='orange',label='SM-Con')
#plt.axvline(x=np.mean(maxpop),color='g',label='Ensemble Mean')
plt.ylabel("Frequency")
plt.xlabel("max Population Deviation")

#plt.title("Population Deviation")
plt.savefig(newdir+"maxpop.png")

plt.close()
sns.distplot(minpop,bins=100, kde=False,color='gray')
#plt.title(plan_name + "Plan" + election_names[j])
#plt.axvline(x=.00000025,color='orange',label='SM-Con')
plt.axvline(x=np.mean(minpop),color='g',label='Ensemble Mean')
plt.ylabel("Frequency")
plt.xlabel("Min Population Deviation")

#plt.title("Population Deviation")
plt.savefig(newdir+"minpop.png")

plt.close()

evec=[[0.020978076408537227, 0.023740272325305626, 0.02918576512455516, 0.030986984447037236, 0.03334549754452667, 0.03478019354134754, 0.043058621500690016, 0.04474671496947767, 0.046149949126374615, 0.04823043049715295, 0.04912956883374996, 0.05584345181818052, 0.06609182668460183, 0.07794164119438433, 0.1779248483748163, 0.21719823224090962, 0.355071231370148, 0.5984576775321707],
[0.4640374505396519, 0.4650916275590472, 0.46755879663633776, 0.4968090109822204, 0.49820231634728696, 0.501358377019075, 0.5034003971398697, 0.5060284556783152, 0.5070672357408126, 0.5153058654458716, 0.5155426324792557, 0.5213360009989226, 0.5263756776546703, 0.6280333115884805, 0.6858123387161138, 0.7380270790602862, 0.8381130295995357, 0.9105044074436827],
[0.40839389612327104, 0.4202627988798736, 0.443183561562483, 0.44498748293126994, 0.45874979305125235, 0.45896532120523026, 0.47102222711267605, 0.4746733302097975, 0.4966506279382877, 0.4970290574481011, 0.5127529582774989, 0.5152901301639682, 0.5169665291304969, 0.6084005526361832, 0.6957809109469701, 0.6980867480215798, 0.8397645439575294, 0.9176567957692146],
[0.3138790794107531, 0.3157976703600727, 0.3240797676360164, 0.3542616060095303, 0.3587052569403426, 0.3681327878360861, 0.37397868561278863, 0.37664822724003516, 0.38366471719802914, 0.39588113980306155, 0.41723948537855965, 0.42579611532255607, 0.43912662319872886, 0.4805349978626866, 0.6031809470175216, 0.6041414484543728, 0.7771922241432815, 0.8944022206616276],
[0.360563113997135, 0.3945190020784081, 0.4138281486173291, 0.4146944790442273, 0.42114565837271983, 0.4224864787420753, 0.43664622590382224, 0.448314913890733, 0.4686870801709849, 0.48477310168246807, 0.4865538276484234, 0.4912404185608727, 0.4996448016083608, 0.5601932911026877, 0.6677799513335559, 0.6897774323253762, 0.830268596798612, 0.909777957206298],
[0.3442819246026917, 0.35064077580870723, 0.36814086402764246, 0.38242408793678373, 0.40152795297141636, 0.4130630116313127, 0.4165386539011146, 0.4185086608679291, 0.4256352584175982, 0.4296190424264139, 0.4490426214749548, 0.45739932477644624, 0.46076003833440904, 0.5189416560641861, 0.6250723978651604, 0.6610192680424125, 0.7925499663767349, 0.9043741169589198],
[0.35259948403849195, 0.3585620460899631, 0.39122780755067216, 0.3937416023615583, 0.4061230716424172, 0.4212669757924787, 0.43914994720168954, 0.4399690613616328, 0.44921150202933025, 0.47529057728941776, 0.47632185470566063, 0.4812432841588194, 0.5006185983040318, 0.5176307856746155, 0.6901134057835705, 0.7052173302436626, 0.8154883032435485, 0.9049669059905766],
[0.28003476439933894, 0.3173068885850031, 0.35145093013585643, 0.3650241321760199, 0.373425884282653, 0.3878860501364614, 0.3922670210520912, 0.39908905863800564, 0.4464757792544871, 0.459727385377943, 0.463387206356325, 0.4987630710220176, 0.5026886831269881, 0.5135626700056128, 0.6728498685020424, 0.6799742604280526, 0.8139346686469956, 0.922510142253914],
[0.3209603739509189, 0.34375237660658603, 0.3691496443346217, 0.37882322021091364, 0.3855132655591371, 0.3998609037167345, 0.4053810669176514, 0.40946654077321776, 0.445355730915498, 0.4462495139274331, 0.4589948042046221, 0.4617352820437841, 0.4662456921577296, 0.49039293316152005, 0.6390778911097517, 0.6893306510386865, 0.798127055092298, 0.8932331415600172],
[0.3940272478762035, 0.4110122216690228, 0.4279560770662464, 0.43846015840952585, 0.4482932599136606, 0.4556307880225737, 0.4619459828653729, 0.4722427536231884, 0.4742473314315538, 0.5041793749873199, 0.5050016209632903, 0.5123706031812132, 0.5170009545606146, 0.5757252377935134, 0.6890585711319753, 0.7163989738872202, 0.8408877673169627, 0.9132538975815913],
[0.37701871384872093, 0.3809631879597022, 0.3943810434983396, 0.420419700116617, 0.42937534275862893, 0.43209974483747265, 0.43587827162139, 0.43668979502118455, 0.4453079260212046, 0.46550227629333846, 0.4773822966185846, 0.484488180414754, 0.4885708949878486, 0.5474422245986853, 0.6577595683569395, 0.6895446143458875, 0.817897087895825, 0.9089174292379002],
[0.34402170741098553, 0.34487168576650135, 0.368207553406184, 0.3913576368149586, 0.3983377125703936, 0.4007122941996278, 0.4141204939130095, 0.4154755041022255, 0.4180056025944282, 0.4357869730710169, 0.45544287603988887, 0.45818842656306996, 0.46351353008116536, 0.504719403054512, 0.6322536643179404, 0.6754462725572511, 0.7955151877780504, 0.898773020127361],
[0.3612005689812185, 0.366046177987092, 0.39138790720572164, 0.4034536345033885, 0.4149274301456343, 0.4239905658964086, 0.4278470298877962, 0.4324784120542098, 0.435183337919664, 0.4585889237050939, 0.47215257559039064, 0.47603182960987733, 0.4810359688268876, 0.5285667556059821, 0.6514111088730807, 0.6894729266810828, 0.8109649753400437, 0.9037330857326699],
[0.35764989447385526, 0.3773277125106721, 0.39853464055878507, 0.4116863862308429, 0.414076566925805, 0.4327547980071001, 0.4340133457922736, 0.43579657008599737, 0.4600657632086842, 0.4743435003374175, 0.4834631266372515, 0.48558639790814284, 0.4912452612122702, 0.533378300590928, 0.6639732647222122, 0.7030138680066774, 0.8191941044416559, 0.9034217164638372]]

rvec=[[0.018361218970278193, 0.02147162463571346, 0.022240160881449065, 0.024719667666142142, 0.03213433066599333, 0.032636710909495825, 0.0367610496357045, 0.047084281876020366, 0.05179467999501321, 0.05472018587054389, 0.06031593860483849, 0.06555162878857741, 0.08982449081367805, 0.10716057032172614, 0.19019412741879702, 0.22916474598003242, 0.2677141594274349, 0.6009776763891153],
[0.42173330984675006, 0.4339057925106587, 0.4455111185816502, 0.4795271697604387, 0.4928513693497348, 0.4960692905177968, 0.5161244807851801, 0.5225882483048446, 0.5304415489037778, 0.5350898948009506, 0.5459604532068301, 0.5690787841850511, 0.5790089417296809, 0.6386951892507948, 0.6459056800796522, 0.6939198701089418, 0.7980426870081468, 0.9249321268271055],
[0.394927651286611, 0.39999482093378563, 0.4099255104126451, 0.42929961280203893, 0.4492475378997455, 0.47095614253198936, 0.47334750231539724, 0.47822213687113585, 0.524122421571146, 0.5253175216469516, 0.5302254698917588, 0.55134318492989, 0.5944195825198979, 0.5945572313284352, 0.6454131900788661, 0.6510254007911722, 0.797803421398927, 0.9363936421732718],
[0.27467158331412767, 0.2805574064953501, 0.2981600320866339, 0.3360419986695955, 0.3439751920975683, 0.345835493754419, 0.3727294582754568, 0.40996910530017716, 0.4234159137054867, 0.4439846446432345, 0.4473588830407514, 0.4559843007854731, 0.48155455217469484, 0.5103597240537336, 0.5617292467962538, 0.5684004444242516, 0.705093713854825, 0.9150622542870853],
[0.3234053489129131, 0.36436446548141305, 0.37508047634639224, 0.3865207008755949, 0.40966556291390727, 0.41935483870967744, 0.4644502303224514, 0.47545691478027113, 0.4769525429759601, 0.5082502675001408, 0.5148622588699001, 0.5360593612099825, 0.560067319133427, 0.5655357733436225, 0.6412774059476058, 0.6422366020919439, 0.7795743068275122, 0.9237602862351225],
[0.3092596883508081, 0.3151015651015651, 0.3307014007860526, 0.3830184952328646, 0.38542907894095707, 0.3885690458252188, 0.4314678186365861, 0.4497187831386218, 0.4663643322293895, 0.4693701534741394, 0.47131488666440596, 0.4774762271909535, 0.519573115120855, 0.5343882734417323, 0.5972333757510615, 0.6134850334658112, 0.7216274634456452, 0.9239382834940447],
[0.3139156624594165, 0.3461457084420622, 0.35672989181126064, 0.36210171437645433, 0.3828728881631793, 0.42680342221819917, 0.42901631336050083, 0.4426724806432756, 0.5034954366619383, 0.5116290468131481, 0.5119817583190376, 0.5251836189273401, 0.5274389518582085, 0.5852861006494626, 0.6195192962364718, 0.6600412951239988, 0.7696166115135796, 0.9129967955850282],
[0.26369456422930976, 0.2765213057903144, 0.3114685929907281, 0.32302048222291385, 0.3499951397403053, 0.3643561971237206, 0.39667164650321085, 0.45099544653837204, 0.4510201650226517, 0.4871719127799477, 0.5058064597854761, 0.5102456367162249, 0.5482076381778691, 0.6002339117663058, 0.6390681044332399, 0.6468456372020516, 0.7453170028818443, 0.9284568179643828],
[0.2891087731386397, 0.3194355742893232, 0.33139528726502515, 0.35728827013285613, 0.3606357544321534, 0.38828387330973774, 0.4029599841782583, 0.4363855928714538, 0.4767058509217205, 0.48017171795353775, 0.4924339352194784, 0.49790921421593776, 0.4983204237066085, 0.5465448116676964, 0.6007697292872329, 0.6432607128740496, 0.7368981052615915, 0.9036609611166722],
[0.3554503848554192, 0.3839455871722416, 0.3888763977575921, 0.3960255717446496, 0.44126599841431646, 0.45641646489104115, 0.4825482966521905, 0.48373323281688674, 0.5142488515850961, 0.5250048978368848, 0.52890466218178, 0.5477766808651648, 0.5839281239529573, 0.5894302700568195, 0.6569017939952037, 0.6710286995168622, 0.7977580753793981, 0.9263511237719003],
[0.33485509652929285, 0.35309863633276506, 0.3592981194712007, 0.38346986686205975, 0.41441505626643277, 0.4340888100365253, 0.4436709106143128, 0.4658118425361767, 0.4900078243625665, 0.49694697629531265, 0.5004676785071288, 0.5138262576144156, 0.5543415778465052, 0.5591889461659201, 0.6268653298450682, 0.6428996683324332, 0.7625209858173969, 0.9251742111476164],
[0.3023838673797118, 0.3310447354322231, 0.3356268254761286, 0.35235093483239455, 0.3733274367990014, 0.4104219323977377, 0.41086538030365266, 0.4271693249088808, 0.47305892474813444, 0.47310838230818764, 0.4849937491693802, 0.48531084025891985, 0.5088289271138536, 0.5404880732021846, 0.5989963634733798, 0.6285357217901487, 0.7298691544339792, 0.9136417065709233],
[0.3198316446408992, 0.35002240217817626, 0.3556823819940942, 0.36275790252881013, 0.39577827203820176, 0.42556477340840676, 0.4348578142258199, 0.4455758828872854, 0.4868027010155249, 0.49010958996858234, 0.49974543390125115, 0.506405135230562, 0.535676366656494, 0.5549578026852651, 0.618161651648761, 0.6430199589849628, 0.7535071292373413, 0.917972542221139],
[0.3222989022439412, 0.35198401121819334, 0.3599498174299282, 0.37804878617841203, 0.3995471161824052, 0.4224711035829063, 0.4433873385998417, 0.45986125818087825, 0.4973917227173435, 0.5005088032616148, 0.5136031842883227, 0.5197329367556034, 0.5438307392222127, 0.5651934597893665, 0.6287278564921182, 0.6573030639867274, 0.7671714438817797, 0.9151068607765802]]



c="k"

a=[]
for elect in range(14):
    a=[]
    for t in ts:
        tempvotes=np.loadtxt(datadir+election_names[elect]+"_"+str(t)+".csv", delimiter=',')
        for s in range(step_size):
            a.append(tempvotes[s,:])
            
    a=np.array(a)        
    medianprops = dict(color='black')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    #ax1.add_patch(patches.Rectangle((0, .37), 35, .18,color='honeydew'))
    #plt.plot([0,34], [.55, .55], 'lightgreen')
    #plt.plot([0,34], [.37, .37], 'lightgreen')
    if elect==0:
        plt.boxplot(a,whis=[1,99],showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="None", color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            )
        plt.plot(range(1,19),evec[0],'o',color='purple',label='2011') 
        plt.plot(range(1,19),rvec[0],'o',color='orange',label='Remedial') 
        plt.plot([.5,num_districts+1],[.4,.4],color='yellow',label='40%')
        plt.ylabel("BVAP %")
        plt.ylim([0,.7])
 
    else:
        plt.boxplot(a,whis=[1,99],showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="None", color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            )
        plt.plot(range(1,19),evec[elect],'o',color='purple',label='2011') 
        plt.plot(range(1,19),rvec[elect],'o',color='orange',label='Remedial') 
        plt.ylabel("Dem %")
        plt.ylim([.25,.95])
    
    plt.plot([.5,num_districts+1],[.5,.5],color='green',label='50%')    
    #plt.plot([],[],color=c,label="ReCom Ensemble")
    
    #fig, ax = plt.subplots()
    #draw_plot(a, 1, "black", "white")
    #plt.xticks(range(1,num_districts+1))
    #plt.plot(range(1,num_districts+1),a[0,:],'o',color='r',label='Initial Plan', markersize=3)
    #plt.plot([1,num_districts+1],[np.mean(a[0,:]),np.mean(a[0,:])],color='blue',label='Initial Mean')
    #plt.plot([1,num_districts+1],[np.median(a[0,:]),np.median(a[0,:])],color='yellow',label='Initial Median')

        
    
    plt.xlabel("Sorted Districts")
    plt.legend()
    plt.savefig(newdir+election_names[elect]+"_box.png")
    fig = plt.gcf()
    fig.set_size_inches((12,6), forward=False)
    fig.savefig(newdir+election_names[elect]+"_box2.png", dpi=600)


    plt.close()        
