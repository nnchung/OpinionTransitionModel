'''This code fit dynamical data of the adoption of an opinion (we use the adoption of Friendster as an example here) with the opinion transition model.

Parameter:
(1) N: size of social network (BA scale-free network)
(2) k: average degree of social network
(3) threshold: a threshold to define the majority, e.g. 0.5
(4) Nens: Number of realization
(5) T: number of iteration for each value of attractiveness, if the opinion attractiveness changes rapidly, set smaller T value (<1N), else allow enough number of iteration for the system to reach equilibrium
(6) pm_fit: range of pm, probability to update agents' state by peer influence (i.e. majority rule)
(7) za_fit: range of za, fraction of zealots whose opinion are always 1
(8) zb_fit: range of zb, fraction of zealots whose opinion are always 0
Parameters for the model can be modified in the "Set parameters" section.

Output:
(1) Values of pm, za and zb for the best fit
(2) a figure displaying the empirical opinion dynamics with the best fit

Reference:
Ning Ning Chung, Lock Yue Chew, Pei-Chun Ko, Choy Heng Lai and Stefan Thurner. Understanding social transitions as an interplay of homophily and
utility (2025). See README for the full citation.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

################################### Load data #####################################
filename = 'Friendster_Worldwide.csv'
cols = ['Month', 'Proxy of attractiveness', 'Google index']
dtypes={'Month': str, 'Proxy of attractiveness': float, 'Google index': float}
data = pd.read_csv(filename,usecols=cols,dtype=dtypes,skiprows=0)
month = data['Month']
month = np.array(month)
proxy = data['Proxy of attractiveness']
proxy = np.array(proxy)
GI = data['Google index']
GI = np.array(GI)

############################# Visualizing relationship ############################
fig1 = plt.figure(num=1, figsize=(5,4))
ax1 = fig1.add_subplot(111)
scatter_min = 4
scatter_max = 45
plt.scatter(proxy,GI,s=np.arange(scatter_min,scatter_max,(scatter_max-scatter_min)/len(GI)),c=np.arange(0,len(GI)),cmap='RdYlGn_r')
plt.xlabel('Proxy of attractiveness', fontsize=10)
plt.ylabel('Google trends', fontsize=10)
plt.xlim([-0.08,0.88])
plt.ylim([-10,110])
plt.text(0.05, 1.04, 'Friendster worldwide', transform=ax1.transAxes, fontsize=10)
plt.subplots_adjust(top=0.88,bottom=0.14,left=0.14,right=0.95,hspace=0.42,wspace=0.35)

forward_path = np.argmax(GI)                               # estimate number of data in the forward path (after data visualization)
########################### Opinion Tansition Model #############################
################################ Set Parameters #################################
N = 4000                                                   # population size
k = 4                                                      # average degree of social network
threshold = 0.5                                            # threshold for majority rule
Nens = 10                                                  # number of realizations
T = 1*N                                                    # number of iterations, depending on the time interval (e.g. a month, four years) between successive data
pm_fit = np.arange(0.55,1.0,0.1)               # probability to follow the majority opinion
za_fit = np.arange(0.0,0.15,0.05)              # zealot
zb_fit = np.arange(0.0,0.15,0.05)              # zealot
h = 0.01                                                   # change in attractiveness
A1 = np.arange(0,0.8,h)                        # values of attractiveness in the forward cycle
A2 = np.arange(0.8,0-h,-h)                     # values of attractiveness in the backward cycle
A = np.concatenate((A1,A2))
GlobalO = np.zeros((len(pm_fit),len(za_fit),len(zb_fit),len(A)))    # store the mean adoption
L2Error = 100.0                                            # initialize the L2 error
bestfit = []                                               # to record the index for the best fit
alpha = 0.05                                                # mutation

################## Simulate the model for varying values of pm, za and zb ##################
r1 = 0
for pm in pm_fit:
    r2 = 0
    for za in za_fit:
        r3 = 0
        for zb in zb_fit:
            ens = 0
            OpinionEns = np.zeros((Nens,len(A)))
            nza = int(round(N * za))
            nzb = int(round(N * zb))
            nf = N - nza - nzb
            print('Running pm=' + str(pm) + ', za=' + str(za) + ', zb=' + str(zb))
            while ens < Nens:
                Opinion = np.zeros((N,1))
                zealots = np.random.choice(N, nza + nzb, replace=False)
                zealota = zealots[:nza]
                zealotb = zealots[nzb:]
                flexi = [i for i in range(N)]
                for zz in zealots:
                    flexi.remove(zz)
                Opinion[zealota] = 1
                if ens % 5 == 0:
                    print('Running realization: ' + str(ens+1))
                ################################################# Generate network #####################################################
                G = [ [] for _ in range(N)]                            # Graph, G[i] record neighbors of node i
                degree = np.zeros(N,dtype=int)                         # degree, degree[i] record degree of node i
                Nseed_BA = 4                                           # seed size to generate BA network
                for jj in range(Nseed_BA):                             # full seed, i.e., all connect to all, make sure the seed forms a connected cluster
                    G[jj] = [x for x in range(Nseed_BA) if x != jj]
                    degree[jj] = len(G[jj])

                mlink = k/2                                            # number of link to be added to each new node
                pos = Nseed_BA                                         # index of new node
                sumdeg = np.sum(degree[:Nseed_BA])
                while pos < N:                                         # add a new node to the network
                    linkage = 0
                    while linkage < mlink:                             # until "mlink" number of link is added
                        nodelist = np.random.choice(pos-1,1)      # choose an existing node randomly
                        node = copy.copy(nodelist[0])
                        if not np.in1d(pos,G[node]):                   # the two are not linked before
                            if node != pos and degree[node]/(sumdeg+0.0) >= np.random.random(1):          # determine whether the node is selected to draw a link by comparing its "degree/total degree" with a random number
                                G[pos].append(node)                    # add "node" to the list of neighbors of "pos"
                                G[node].append(pos)
                                degree[pos] += 1
                                degree[node] += 1
                                linkage += 1
                                sumdeg += 1
                    pos += 1

                ####################################################################################################################
                for qq in range(0,len(A)):
                    if len(A) / 2 > qq > len(A) / 2 - 10:
                        T = (1 - 0.05 * (qq - len(A) / 2 + 10)) * N
                    elif qq >= len(A) / 2:
                        T = 0.5 * N
                    else:
                        T = 1.0 * N
                    attractiveness = A[qq]
                    Mutation = np.random.choice(nf, int(round(alpha * nf)), replace=False)
                    for mu in Mutation:
                        Opinion[flexi[mu]] = 1 - Opinion[flexi[mu]]
                    Equil = 0
                    while Equil < T:
                        updatef = np.random.choice(nf, 1)[0]
                        update = flexi[updatef]
                        if np.random.rand(1) < pm:
                            O1Neigh = np.sum(Opinion[G[update]])/float(degree[update])
                            if Opinion[update] == 0 and O1Neigh > threshold:
                                Opinion[update] = 1
                            elif Opinion[update] == 1 and O1Neigh < threshold:
                                Opinion[update] = 0
                        else:
                            if np.random.rand(1) < attractiveness:
                                Opinion[update] = 1
                            else:
                                Opinion[update] = 0
                        Equil += 1
                    OpinionEns[ens,qq] = np.mean(Opinion)
                ens += 1
            Adoption = -6 + 125 * np.mean(OpinionEns, 0)     # match the scale to Google Index
            GlobalO[r1, r2, r3, :] = Adoption
            ############################# L2 Error #############################
            L2 = 0
            for fit in range(len(proxy)):
                if fit <= forward_path:
                    id = np.argmin(abs(A[:len(A1)] - proxy[fit]))
                    L2 += (0.01*Adoption[id] - 0.01*GI[fit])**2
                else:
                    id = np.argmin(abs(A[len(A1):] - proxy[fit]))
                    L2 += (0.01*Adoption[len(A1)+id] - 0.01*GI[fit])**2
            L2 = np.sqrt(L2)
            if L2 < L2Error:
                bestfit = [r1, r2, r3, L2]
                L2Error = copy.copy(L2)
            r3 += 1
        r2 += 1
    r1 += 1

print('pm: ', pm_fit[bestfit[0]])
print('za: ', za_fit[bestfit[1]])
print('zb: ', zb_fit[bestfit[2]])
print('L2Error: ', bestfit[3])
plt.plot(A[::3],GlobalO[bestfit[0], bestfit[1], bestfit[2], ::3],color='grey',linewidth=1.0)
fig1.savefig('Figure2.png', format='png', dpi=300)
plt.show()