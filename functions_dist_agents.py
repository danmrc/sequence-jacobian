#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:13:42 2022

@author: danielcoutinho
"""

import numpy as np
import random

def return_asset(a,e,a_grid,e_grid,decision_rule):
    a_index = np.nonzero(a == a_grid)
    e_index = np.nonzero(e == e_grid)
    return decision_rule[e_index,a_index]

def return_shock(e,e_grid,Pi):
    nE = e_grid.size
    index = np.nonzero(e == e_grid)
    probs = Pi[index[0][0],:]
    return int(random.choices(range(nE),weights  = probs, k = 1)[0])

def start_from_steady(nP,D_ss,e_grid,a_grid):
    nE,nA = D_ss.shape
    D_vec = np.reshape(D_ss,(nE*nA,1))
    start_number = random.choices(range(nE*nA),weights = D_vec, k = nP)
    
    state = np.zeros((2,nP))
    
    for i in range(nP):
        state[0,i] = e_grid[int(start_number[i]/nA)]
        state[1,i] = a_grid[start_number[i] % nA]
        
    return state

def update_economy(state_now,a_grid,e_grid,a_policy,Pi):
    state = np.copy(state_now)
    _,nP = state_now.shape
    for i in range(nP):
        state[0,i] = return_shock(state_now[0,i],e_grid,Pi)
        state[1,i] = return_asset(state_now[1,i], state_now[0,i], a_grid, e_grid, a_policy)
    return state

def compute_distribution(state,a_grid,nE):
    
    a_on_grid = np.digitize(state[1,:],a_grid)
    nA = a_grid.size
    _,nP = state.shape
    
    distribution = np.zeros((nE,nA))
    
    for i in range(nP):
        column = a_on_grid[i]-1
        line = int(state[0,i])
        distribution[line,column] +=1

    distribution = distribution/nP
    
    return distribution
        
        
        
        
        