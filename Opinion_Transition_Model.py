'''This code simulate the opinion transition model for a change of opinion attractiveness from 0 to 1 and back to 0.

Parameter:
(1) N: size of social network (BA scale-free network)
(2) k: average degree of social network
(3) threshold: a threshold to define the majority, e.g. 0.5
(4) Nens: Number of realization
(5) T: number of iteration for each value of attractiveness
(6) pm: probability to update agents' state by peer influence (i.e. majority rule)
(7) za: fraction of zealots whose opinion are always 1
(8) zb: fraction of zealots whose opinion are always 0
Parameters for the model can be modified in the "Set parameters" section.

Output:
(1) npz files which store the mean adoption rate
(2) a figure displaying the opinion dynamics

Reference:
Ning Ning Chung, Lock Yue Chew, Pei-Chun Ko, Choy Heng Lai and Stefan Thurner. Understanding social transitions as an interplay of homophily and
utility (2025).
'''

import numpy as np
import matplotlib.pyplot as plt
import copy

############################################# Set Parameters #####################################################
N = 4000                                                   # population size
k = 4                                                      # average degree of social network
threshold = 0.5                                            # threshold for majority rule
Nens = 50                                                  # number of realizations
T = 10*N                                                   # number of iteration
pm = 0.8                                                   # probability to follow the opinion of the majority
za = 0.0                                                   # fraction of zealots stick to opinion A
zb = 0.0                                                   # fraction of zealots stick to opinion B
nza = int(round(N * za))
nzb = int(round(N * zb))
nf = N - nza - nzb
h = 0.01                                                   # change in attractiveness
A1 = np.arange(0,1,h)                          # values of attractiveness in the forward cycle
A2 = np.arange(1,0-h,-h)                       # values of attractiveness in the backward cycle
A = np.concatenate((A1,A2))
GlobalO = np.zeros((Nens,len(A)))                          # store the mean adoption
alpha = 0.05                                               # mutation
ens = 0

while ens < Nens:
    Opinion = np.zeros((N,1))                              # initialization of opinion
    zealots = np.random.choice(N, nza + nzb, replace=False)
    zealota = zealots[:nza]
    zealotb = zealots[nzb:]
    flexi = [i for i in range(N)]
    for zz in zealots:
        flexi.remove(zz)
    Opinion[zealota] = 1
    if ens % 10 == 0:                                      # display progress
        print('Running realization: ' + str(ens+1))
    ################################################# Generate network #####################################################
    G = [ [] for _ in range(N)]                            # Graph, G[i] record neighbors of node i
    degree = np.zeros(N,dtype=int)                         # degree, degree[i] record degree of node i

    Nseed_BA = copy.copy(k)                                # seed size to generate BA network
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

    ########################################### Iteration ########################################################
    for qq in range(0,len(A)):                             # Attractiveness of the opinion changes from 0 to 1 and back to 0
        attractiveness = A[qq]
        Mutation = np.random.choice(nf, int(round(alpha * nf)), replace=False)
        for mu in Mutation:                                # randomly choose alpha of nodes to change opinion
            Opinion[flexi[mu]] = 1 - Opinion[flexi[mu]]
        Opinion[Mutation] = 1 - Opinion[Mutation]
        Equil = 0
        while Equil < T:                                   # run the update until T
            updatef = np.random.choice(nf, 1)[0]      # randomly choose an agent from those whose opinion is flexible to change to update
            update = flexi[updatef]
            if np.random.rand(1) <= pm:                    # update by majority rule
                O1Neigh = np.sum(Opinion[G[update]])/float(degree[update])   # fraction of neighbors with opinion 1
                if Opinion[update] == 0 and O1Neigh > threshold:             # if opinion is 0 and fraction of neighbors with opinion 1 greater than threshold
                    Opinion[update] = 1                                      # switch opinion
                elif Opinion[update] == 1 and O1Neigh < threshold:           # if opinion is 1 and fraction of neighbors with opinion 1 less than threshold
                    Opinion[update] = 0                                      # switch opinion
            else:                                          # update according to attractiveness
                if np.random.rand(1) <= attractiveness:
                    Opinion[update] = 1
                else:
                    Opinion[update] = 0
            Equil += 1

        GlobalO[ens,qq] = np.mean(Opinion)                 # store the value of mean opinion
    ens += 1

################################### Save the results in npz format #####################################
filename = 'AttractivenessModel_N' + str(N) + '_k' + str(k) + '_Nens' + str(Nens) + '_pm' + str(pm) + '.npz'
np.savez(filename, GlobalO=GlobalO, A=A, N=N, k=k, T=T, Nens=Nens)

#################################### Plot the opinion dynamics #########################################
fig1 = plt.figure(num=1, figsize=(6.0,4.5))
plt.plot(A,np.mean(GlobalO,0),color='dodgerblue')
plt.xlabel('Attractiveness', fontsize=12)
plt.ylabel('Fraction of adoption', fontsize=12)
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.95, hspace=0.3, wspace=0.56)
fig1.savefig('Figure1.png', format='png', dpi=300)

plt.show()