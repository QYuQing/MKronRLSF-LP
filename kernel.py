# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:43:21 2022

@author: qianyuqing
"""
import numpy as np
from sklearn import metrics
import numpy.matlib
from jax import random
from neural_tangents import stax

def kernel_gussian(x , gamma):
    #相似性矩阵计算
    n = len(x)
    ga = np.dot(x,x.T)
    ga = gamma * ga/np.diagonal(ga).mean()
    
    di = np.diagonal(ga)
    d = np.matlib.repmat(di,len(di),1) + np.matlib.repmat(di,len(di),1).T - 2*ga   
            
    return np.exp(-d)

def kernel_cosine(x , mu , sigma):
    #Calculates the link indicator kernel from a graph adjacency by cosine similiarity 
    n = len(x)
    m = len(x[0])
    #Add Gaussian random noise matrix
    x = x + np.random.normal(mu, sigma , (n,m))
    kernel = np.zeros([n,n])
    for i in range(n):
        for j in range(i,n):
            kernel[i,j] = np.dot(x[i,:],x[j,:].T)/(np.linalg.norm(x[i,:])*np.linalg.norm(x[j,:]))
            kernel[j,i] = kernel[i,j]
    return kernel

def kernel_corr(x , mu , sigma):
    #Calculates the link indicator kernel from a graph adjacency by pairwise linear correlation coefficient
    n = len(x)
    m = len(x[0])
    #Add Gaussian random noise matrix
    x = x + np.random.normal(mu, sigma , (n,m))
    return np.corrcoef(x)

def kernel_MI(x):
    n = len(x)
    m = len(x[0])
    kernel = np.zeros([n,n])
    for i in range(n):
        for j in range(i,n):
            kernel[i,j] = metrics.normalized_mutual_info_score(x[i,:], x[j,:])
            kernel[j,i] = kernel[i,j]
    return kernel

def kernel_normalized(k):
    #理想核矩阵的归一化
    n = len(k)
    
    k = np.abs(k)
    index_nozeros = k.nonzero()
    min_value = min(k[index_nozeros])
    k[np.where(k==0)] = min_value
    
    diag = np.resize(np.diagonal(k), [n,1])**0.5
    k_nor = k/(np.dot(diag,diag.T))
    return k_nor

def kernel_ntk(y): 
             
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(2048), stax.Relu(),
        stax.Dense(2), stax.Relu(),
        stax.Dense(1)
    )

    key1, key2 = random.split(random.PRNGKey(2022))

    ntk = kernel_fn(y , y , 'ntk')
    ntk = np.array(ntk)
    # return kernel_normalized(ntk)
    return ntk

