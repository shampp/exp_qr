# we provide two selection strategies, one is as per our proposed algorithm and the second one is random selection
import heapq
import numpy as np
from joblib import Parallel,delayed
import multiprocessing
from operator import itemgetter
from numpy.random import Generator, PCG64
from math import log10
from itertools import chain
import logging

logger = logging.getLogger(__name__)

def scheme_selection(scheme,A,S,k,epsilon): #A : set of contexts (by index), S : similarity vector wrt anchor index, m : number of partitions, k : number of output arms
    return submodular_select(A,S,k,epsilon) if scheme == 'submodular' else random_select(A,S,k,epsilon)

def random_select(A,S,k,epsilon):
    logger.info("Running random selection scheme")
    rg = Generator(PCG64(54321))
    return rg.choice(A,k,replace=False)

def submodular_select(A,S,k,epsilon): #A : set of contexts (by index), S : similarity vector wrt anchor index, m : number of partitions, k : number of output arms
    #parts = partition(np.delete(A,qi),m)
    logger.info("Running submodular selection scheme")
    num_cores = multiprocessing.cpu_count()
    parts = partition(A,num_cores)

    sm = S.sum() # S is a one dimensional numpy array
    sm_g = S[S >= epsilon].sum() + 1e-3
    sm_l = S[S < epsilon].sum() + 1e-3
    pi = sm_g/sm
    br_pi = sm_l/sm
    logger.info("sm_g: %f" %(sm_g))
    logger.info("sm_l: %f" %(sm_l))
    U = Parallel(n_jobs=num_cores)(delayed(lazy_greedy)(A,sm_g, sm_l, pi, br_pi, k, S, P) for P in parts)    #in total we have k \times m elementss
    #U = list()
    #for P in parts:
    #    U.append(lazy_greedy(A,sm_g, sm_l, pi, br_pi, k, S, P))    #in total we have k \times m elementss
    P_m, sol1 = max(U,key=itemgetter(1))
    B = chain(*[u[0] for u in U])
    P_B, sol2 = lazy_greedy(A,sm_g, sm_l, pi, br_pi, k, S, B)
    logger.info("finished with submodular selection scheme")

    return P_m if sol1 > sol2 else P_B

def lazy_greedy(A,sm_g, sm_l, pi, br_pi, k, S, P):
    hq = list()
    for j in P:
        s = compute_score(A,sm_g,sm_l,pi,br_pi,[j],S)
        heapq.heappush(hq, (-s,j))
    s,j = heapq.heappop(hq) #this s is negative of the original
    solution = [j]
    #score = -s
    for _ in range(k-1):    # we already selected one item, now have to select k-1 items
        matched = False
        while not matched:
            _, c_j = heapq.heappop(hq)
            s_g = compute_score(A,sm_g, sm_l, pi, br_pi, solution + [c_j], S) + s  # we need to subtract s but since python uses min heap, we store negative value. so we add instead of subtract
            heapq.heappush(hq, (-s_g,c_j))
            matched = hq[0][1] == c_j
        s_g, j = heapq.heappop(hq)
        s += s_g
        solution.append(j)
        #scores.append(-s)
    return solution,-s
        

def compute_score(A,sm_g, sm_l, pi, br_pi, sols, S):
    return log10(sum([ pi*pi*S[np.where(A==j)[0][0]]/sm_g + br_pi*br_pi*S[np.where(A == j)[0][0]]/sm_l for j in sols ]))
    

def partition(A,m):
    return np.array_split(A,m)

