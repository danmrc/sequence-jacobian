#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:12:47 2022

@author: danielcoutinho
"""
import random
import numpy as np
import monetary_policy_full_sim as rstar
import matplotlib.pyplot as plt

def return_asset(a,e,a_grid,e_grid,decision_rule):
    a_index = np.nonzero(a == a_grid)
    e_index = np.nonzero(e == e_grid)
    return decision_rule[e_index,a_index]

def return_shock(e,e_grid,Pi):
    nE = e_grid.size
    index = np.nonzero(e == e_grid)
    probs = Pi[index[0][0],:]
    return random.choices(range(nE),weights  = probs, k = 1)

a = rstar.a_all_rstar[:,:,0] #rule for asset choice in the first period

nP = 1_000_000 #size of the population
D_vec = np.reshape(rstar.D_ss_rstar,(rstar.nE*rstar.nA,1))
start_number = random.choices(range(rstar.nE*rstar.nA),weights = D_vec, k = nP)

e_shock = np.zeros(nP)

for i in range(nP):

    e_shock[i] = rstar.e_grid_rstar[int(start_number[i]/rstar.nA)]


start_asset = np.zeros(nP)

for i in range(nP):

    start_asset[i] = rstar.a_grid_rstar[start_number[i] % rstar.nA]
    
assets_next = np.zeros(nP) #will receive the amount of assets choosen by the agent in the next period
shock_next = np.zeros(nP) #will receive the position of the shock in the next period

for i in range(nP):

    assets_next[i] = return_asset(start_asset[i],e_shock[i],rstar.a_grid_rstar,rstar.e_grid_rstar,a)
    shock_next[i] = return_shock(e_shock[i], rstar.e_grid_rstar, rstar.Pi_rstar)[0]    
    #shock next is a position on the grid, while asset next is a value!!! very different animals
    
pos = np.digitize(assets_next,rstar.a_grid_rstar) #computes the point of the grid that is closest to the choice of agent   
D_marg = np.sum(rstar.D_ss_rstar,axis=0) 

# intermediate check: marginalizing wrt the productivity shock
hist = np.zeros(rstar.nA)
for i in range(rstar.nA):
    hist[i] = np.size(np.nonzero(pos == i + 1))
    
plt.plot(np.abs(D_marg - hist/nP))

# full distribution 

shock_next = np.intc(shock_next) #makes the shock position become a integer instead of a float

full_dist = np.zeros((rstar.nE,rstar.nA)) 

for i in range(nP):
    full_dist[shock_next[i],pos[i]] += 1

full_dist = full_dist/nP #to make it sure that the distribution sums to 1




