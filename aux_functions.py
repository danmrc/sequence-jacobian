#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:38:52 2022

@author: danielc
"""

import scipy as sp
import numpy as np

def chebyschev_grid(amax,n,amin): #see Judd(1998, p. 223)
    grid = np.linspace(1,n,num = n)
    cheby_node = -np.cos((2*grid-1)/(2*n)*np.pi)
    adj_node = (cheby_node+1)*(amax - amin)/2 + amin
    return adj_node

def kernel_smoothing(vec,bandwidth):
    n = np.size(vec)
    result = np.zeros(n)
    for i in range(n):
        kernel = sp.stats.norm(vec[i],bandwidth)
        weights = kernel.pdf(vec)
        weights = weights/np.sum(weights)
        result[i] = weights@vec
    return result