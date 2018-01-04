'''
Python Programs from Markus Ferdinand Dablander (CoMPLEX Project 2)
'''

import numpy as np
from CoMPLEX_Project_1 import tsn
import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF as empdis



def ccorr(ts1,ts2,tau):

	'''
	Cross-Correlation between two time series ts1 and ts2 and timeshift tau.
	ts1 and ts2 can be lists or 1-dimensional numpy arrays.
	The function returns the Pearson Correlaton Coefficient between ts1(t) and ts2(t+tau).
	This means that ts1 "happens before" ts2 and we say that ts1 leads ts2.
	'''

    if tau == 0:
        c = np.corrcoef(ts1,ts2)[0,1]
    if tau > 0:
        c = np.corrcoef(ts1[:-tau],ts2[tau:])[0,1]
    if tau < 0:
        c = np.corrcoef(ts1[-tau:],ts2[:tau])[0,1]
    return c
    
	
def cc3(t1,t2):
    return ccorr(t1,t2,3)

def cc0(t1,t2):
    return ccorr(t1,t2,0)


def adjacency_matrix(T, weightfunction = cc9, self_loops = True):

	'''
	T = List of Lists (or 2-dimensional numpy array) that contains a time series in every row.
	weightfunction = function that assigns a single number to any given pair of time series.
	adjacency matrix returns a 2-d-numpy array that can be seen as the adjacency matrix of a graph. In this graph, every time series in T is a node and there is a directed edge between any two nodes with a weight given by weightfunction.

	'''

    #Initial checks and definitions
    T = np.array(T)
    n = T.shape[0]
    
    #Construction of weight-adjacency matrix
    Gw = np.zeros((n,n))
    for i in list(range(n)):
        for j in list((range(n))):
            if self_loops == True:
                weight = weightfunction(T[i],T[j])
                Gw[i,j] = weight
            else:
                if i != j:
                    weight = weightfunction(T[i],T[j])
                    Gw[i,j] = weight
    return Gw


def emp_dis_tscorr_null(T, n_small, weightfunction = cc0 , blocksize = 100, rep = 1):

	'''
	Returns an empirical cumulative null distribution function F for the absolute correlation values between pairs of uncorrelated time series of length n_small.
	This is done via block-bootstrapping with blocksize "blocksize". The higher "rep" is, the smoother F becomes.
	'''
    
    m = len(T)
    n = len(T[0])
    
    if n % blocksize > 0:
        big_blocks_extracted = math.floor(n/blocksize)
    elif n % blocksize == 0:
        big_blocks_extracted = math.floor(n/blocksize)-1
    
    T_blocks_big = []
    T_blocks_small = []
    
    for ts in T:
        for b in list(range(big_blocks_extracted)):
            T_blocks_big.append(ts[b*blocksize:(b+1)*blocksize])
            
        T_blocks_small.append(ts[big_blocks_extracted*blocksize:])
    
    
    null_values = []
    
    
    for kk in range(rep):
        T_blocks_big = np.random.permutation(np.array(T_blocks_big))
        T_blocks_small = np.random.permutation(np.array(T_blocks_small))

        T_mixed = []

        for k in list(range(m)):

            ts_mixed = []

            for j in list(range(k*big_blocks_extracted,(k+1)*big_blocks_extracted)):
                ts_mixed.append(list(T_blocks_big[j]))
            ts_mixed.append(list(T_blocks_small[k]))

            ts_mixed = np.random.permutation(ts_mixed)
            ts_mixed = list(ts_mixed)
            ts_mixed = [val for sublist in ts_mixed for val in sublist][0:n_small]

            T_mixed.append(ts_mixed)

        T_mixed = np.array(T_mixed)

        for i in list(range(m)):
            for j in list((range(i+1,m))):
                    weight = weightfunction(T_mixed[i],T_mixed[j])
                    null_values.append(weight)    
    
    
    null_values = np.array(null_values)
    
    F_null = empdis(np.absolute(null_values))
    
    return [F_null, null_values]


def adjacency_matrix_benjamini_hochberg(T, weightfunction, F_emp_null, false_discovery_rate = 0.1, pos_reg_dep = "off", self_loops = True):

	'''
	Returns a adjacency matrix just like the function adjacency_matrix, but uses an empirical null distribution and the Benjamini-Hochberg-Yekutieli
	procedure to test all edgeweights for significance. Only significant edges appear in the final adjacency matrix.
	'''
    
    #Initial checks and definitions 
    T = np.array(T)
    n = T.shape[0]
    t = len(T[0])

    if self_loops == True:
        m = n*n
    else:
        m = n*(n-1)
  
    #Definition of empirical distribution function under null hypothesis that correlation(t1,t2) -> 0 (block-bootstrapping)
    F_null = F_emp_null
    

    #Construction of weight-adjacency matrix Gw of T
    Gw = adjacency_matrix(T, weightfunction = weightfunction, self_loops = self_loops)
    
    #Finding of Benjamin-Hochberg significant p-values and associated pairs of time series
    pvals = []
    for i in range(len(Gw)):
        for j in range(len(Gw)):
            if self_loops == True:
                p = 1-F_null(abs(Gw[i,j]))
                pvals.append([i,j,p])
            else:
                if i != j:
                    p = 1-F_null(abs(Gw[i,j]))
                    pvals.append([i,j,p])

    def getKey(item):
        return item[2]
    
    pvals_sorted = sorted(pvals, key = getKey)
    
    if pos_reg_dep == "off":
        harmonic = np.sum(np.array([1/j for j in list(range(1,m+1))]))
    else:
        harmonic = 1
    # assumption of positive regression dependency of set of all statistics on null hypothesis statistics
    # this is reasonable because high correlation between ind. ts by chance is not expected to decrease correlation
    # between any other pair of ts
    
    def th(k, fdr, mm):
        return k*fdr/(mm*harmonic)
    
    pvals_logic = []
    
    for k in list(range(1,len(pvals_sorted)+1)):
        if pvals_sorted[k-1][2] <= th(k,false_discovery_rate,m):
            pvals_logic.append(1)
        else:
            pvals_logic.append(0)
    
    kmax = -1
    
    for k in list(range(len(pvals_logic)-1)):
        if pvals_logic[k] == 1 and sum(pvals_logic[k+1:]) == 0:
            kmax = k
            break
    
    if pvals_logic[-1] == 1:
        kmax = len(pvals_logic)-1
    
    pvals_significant = []
    Gw_bh = np.zeros((n,n))
    
    if kmax != -1:    
        pvals_significant = pvals_sorted[:kmax+1]

        #Construction of adjacency matrix with only significant weights
       
        for v in pvals_significant:
            i = v[0]
            j = v[1]
            Gw_bh[i,j] = Gw[i,j]


        print("Results:", len(pvals_significant),"discoveries at a false-discovery-rate of", false_discovery_rate)
    
    if kmax == -1:
        print("No significant results")
    
    return [Gw, Gw_bh, pvals_significant]


def matrix_to_gml_file(Gw, L, name):

	'''
	Takes and adjacency matrix Gw with a list of labels (one label per row) and saves the corresponding labeled graph as a gml file with name "name".
	'''
    
    Gw = np.array(Gw)
    n = len(Gw)
    
    G = nx.from_numpy_matrix(Gw, create_using=nx.MultiDiGraph())
    
    ll = {}
    for k in list(range(n)):
        ll[k] = str(k) +": " + L[k]
    G = nx.relabel_nodes(G, ll)
    
    nx.write_gml(G, name + ".gml")