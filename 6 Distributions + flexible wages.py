#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:03:50 2022

@author: danielc
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from sequence_jacobian import het, simple, create_model              # functions
from sequence_jacobian import interpolate, grids, misc, estimation   # modules

def bissection_onestep(f,a,b):
    if not np.all(f(a)*f(b) <= 0):
        raise ValueError("No sign change")
    else:
        mid_point = (a + b)/2
        mid_value = f(mid_point)
        new_a = a
        new_b = b
        indxs_a = np.nonzero(mid_value*f(b) <= 0)
        indxs_b = np.nonzero(mid_value*f(a) <= 0)
        if indxs_a[0].size != 0:
            new_a[indxs_a] = mid_point[indxs_a]
        if indxs_b[0].size != 0:
            new_b[indxs_b] = mid_point[indxs_b]
        return new_a,new_b

def vec_bissection(f,a,b,iter_max = 100,tol = 1E-11):
    i = 1
    err = 1
    while i < iter_max and err > tol:
        a,b = bissection_onestep(f,a,b)
        err = np.max(np.abs(a - b))
        i += 1
    if i >= iter_max:
        raise ValueError("No convergence")
    return a

# Household heterogeneous block
def consumption(c, we, rest, gamma, nu, phi, tauc, taun):
    return (1 + tauc) * c - (1 - taun) * we * ((1 - taun) * we / ((1 + tauc) * phi * c ** gamma)) ** (1/nu) - rest

def household_guess(a_grid, e_grid, r, w, gamma, T, tauc, taun):
    wel = (1 + r) * a_grid[np.newaxis,:] + (1 - taun) * w * e_grid[:,np.newaxis] + T[:,np.newaxis]
    V_prime = (1 + r) / (1 + tauc) * (wel * 0.1) ** (-gamma) # check
    return V_prime

@het(exogenous = 'Pi',policy = 'a', backward = 'V_prime', backward_init = household_guess)
def household(V_prime_p, a_grid, e_grid, r, w, T, beta, gamma, nu, phi, tauc, taun):

    we = w * e_grid
    c_prime = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma) # c_prime is the new guess for c_t
    n_prime = ((1 - taun) * we[:,np.newaxis] / ((1 + tauc) * phi * c_prime ** gamma)) ** (1/nu)
    new_grid = ((1 + tauc) * c_prime + a_grid[np.newaxis,:] - (1 - taun) * we[:,np.newaxis] * n_prime 
                - T[:,np.newaxis])
    wel = (1 + r) * a_grid

    c = interpolate.interpolate_y(new_grid,wel,c_prime)
    n = interpolate.interpolate_y(new_grid,wel,n_prime)

    a = wel + (1 - taun) * we[:,np.newaxis] * n + T[:,np.newaxis] - (1 + tauc) * c
    V_prime = (1 + r) / (1 + tauc) * c ** (-gamma) # check

    # Check for violation of the asset constraint and fix it
    indexes_asset = np.nonzero(a < a_grid[0]) # first dimension: labor grid, second dimension: asset grid
    a[indexes_asset] = a_grid[0]

    if indexes_asset[0].size != 0 and indexes_asset[1].size !=0:
        aa = np.zeros((indexes_asset[0].size)) + 1E-10 # was 1E-5
        rest = wel[indexes_asset[1]] - a_grid[0] + T[indexes_asset[0]]
        bb = c[indexes_asset] + 0.5 + 100
        c[indexes_asset] = vec_bissection(lambda c : consumption(c,we[indexes_asset[0]],
                                                                 rest,gamma,nu,phi,tauc,taun),aa,bb)
        n[indexes_asset] = ((1 - taun) * we[indexes_asset[0]] 
                            / ((1 + tauc) * phi * c[indexes_asset] ** gamma)) ** (1/nu)
        V_prime[indexes_asset] = (1 + r) / (1 + tauc) * (c[indexes_asset]) ** (-gamma) # check
    return V_prime, a, c, n

def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho = rho_e, sigma = sd_e, N = nE)
    a_grid = grids.agrid(amin = amin, amax = amax, n = nA)
    return e_grid, Pi, a_grid, pi_e

def transfers(pi_e, Div, Transfer, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = np.ones(e_grid.size), e_grid #np.ones(e_grid.size)
    div = Div / np.sum(pi_e * div_rule) * div_rule
    transfer =  Transfer / np.sum(pi_e * tax_rule) * tax_rule 
    T = div + transfer
    return T

household_inp = household.add_hetinputs([make_grid,transfers])

def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne

hh_ext = household_inp.add_hetoutputs([labor_supply])

@simple
def firm(Y, w, Z, pi, mu, kappa, tauc):
    L = Y / Z
    Div = Y - w * L - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    cpi = (1 + pi) * (1 + tauc) - 1
    return L, Div, cpi

@simple
def monetary(pi, rstar, phi_pi):
    r = (1 + rstar(-1) + phi_pi * pi(-1)) / (1 + pi) - 1
    i = rstar
    return r, i

@simple
def fiscal(r, Transfer, B, C, L, tauc, taun, w):
    govt_res = Transfer + (1 + r) * B(-1) - tauc * C - taun * w * L - B
    Deficit = tauc * C + taun * w * L - Transfer # primary surplus
    Trans = Transfer
    return govt_res, Deficit, Trans

@simple
def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    return asset_mkt, labor_mkt, goods_mkt

@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w

@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


#########################################################################
##########################################################################

def household_d(V_prime_p, a_grid, e_grid, r, w, T, beta, gamma, nu, phi, tauc, taun):

    we = w * e_grid
    c_prime = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma) # c_prime is the new guess for c_t
    n_prime = ((1 - taun) * we[:,np.newaxis] / ((1 + tauc) * phi * c_prime ** gamma)) ** (1/nu)
    new_grid = ((1 + tauc) * c_prime + a_grid[np.newaxis,:] - (1 - taun) * we[:,np.newaxis] * n_prime 
                - T[:,np.newaxis])
    wel = (1 + r) * a_grid

    c = interpolate.interpolate_y(new_grid,wel,c_prime)
    n = interpolate.interpolate_y(new_grid,wel,n_prime)

    a = wel + (1 - taun) * we[:,np.newaxis] * n + T[:,np.newaxis] - (1 + tauc) * c
    V_prime = (1 + r) / (1 + tauc) * c ** (-gamma) # check

    # Check for violation of the asset constraint and fix it
    indexes_asset = np.nonzero(a < a_grid[0]) # first dimension: labor grid, second dimension: asset grid
    a[indexes_asset] = a_grid[0]

    if indexes_asset[0].size != 0 and indexes_asset[1].size !=0:
        aa = np.zeros((indexes_asset[0].size)) + 1E-10 # was 1E-5
        rest = wel[indexes_asset[1]] - a_grid[0] + T[indexes_asset[0]]
        bb = c[indexes_asset] + 0.5 + 100
        c[indexes_asset] = vec_bissection(lambda c : consumption(c,we[indexes_asset[0]],
                                                                 rest,gamma,nu,phi,tauc,taun),aa,bb)
        n[indexes_asset] = ((1 - taun) * we[indexes_asset[0]] 
                            / ((1 + tauc) * phi * c[indexes_asset] ** gamma)) ** (1/nu)
        V_prime[indexes_asset] = (1 + r) / (1 + tauc) * (c[indexes_asset]) ** (-gamma) # check
    return V_prime, a, c, n

def iterate_household(foo,V_prime_start,Pi,a_grid,w,taun,pi_e,e_grid,r,Div,Transfer,beta,gamma,nu,tauc,maxit = 1000,tol = 1E-8):
    
    V_prime_p = Pi@V_prime_start
    V_prime_old = V_prime_start
    #_,_,c,_ = foo(V_prime_p,a_grid,z_grid,e_grid,r,T,beta,gamma,nu,tauc)
    
    ite = 0
    err = 1
    
    T = transfers(pi_e, Div, Transfer, e_grid)
    
    while ite < maxit and err > tol:
        
        #c_old = np.copy(c)
        V_prime_temp,a,c,n = foo(V_prime_p,a_grid,e_grid,e_grid,r,T,beta,gamma,nu,tauc,taun)
        V_prime_p = Pi@V_prime_temp
        
        ite += 1
        err = np.max(np.abs(V_prime_old - V_prime_temp))
        V_prime_old = V_prime_temp
        
    #print("Iteration ", ite, " out of ", maxit, "\n Difference in policy (sup norm):", err)
    
    return V_prime_temp,a,c,n

##############################################################################
############################### Transfer Policy ###############################
###############################################################################

blocks_ss = [hh_ext, firm, monetary,fiscal, nkpc_ss, mkt_clearing]

hank_ss = create_model(blocks_ss, name = "One-Asset HANK SS")


calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.92, 'nE': 8,
               'amin': 0, 'amax': 200, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0, 
               'tauc': 0.1, 'taun': 0.036}

unknowns_ss = {'beta': 0.986, 'phi': 0.8, 'Transfer': 0.05}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0, 'govt_res': 0}

ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver = "hybr")

blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc]
hank = create_model(blocks, name = "One-Asset HANK")
#print(*hank.blocks, sep='\n')
ss = hank.steady_state(ss0)

#for k in ss0.keys():
#    assert np.all(np.isclose(ss[k], ss0[k]))
    
T = 300
exogenous = ['rstar', 'Transfer', 'Z', 'tauc']
unknowns = ['pi', 'w', 'Y', 'B']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt', 'govt_res']

# general equilibrium jacobians
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

shock = np.zeros(T)
discount = (1 / (1 + ss['r']))
#discount = 1
#A, B, C, D, E = 1, 0.5, 0.19499, 5, 3
A, B, C, D, E = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (A - B * (x - E)) * np.exp(-C * (x - E) - D) 
    
# z_grid = ss0.internals['household']['z_grid']
e_grid = ss0.internals['household']['e_grid']
a_grid = ss0.internals['household']['a_grid']
D = ss0.internals['household']['Dbeg']
pi_e =  ss0.internals['household']['pi_e']
Pi = ss0.internals['household']['Pi']

nu = ss0['nu']
beta = ss0['beta']
gamma = ss0['gamma']
tauc = ss0['tauc']
taun = ss0['taun']
nE = ss0['nE']
nA = ss0['nA']

c_steady = ss0.internals['household']['c']
r_steady = ss0['r']
Transfer_steady = ss0['Transfer']
Div_steady = ss0['Div']
T_steady = transfers(pi_e,Div_steady,Transfer_steady,e_grid)
w_steady = ss0['w']
N_steady = ss0['N']

rhos = 0.9
dtstar = shock

path_w = w_steady + G['w']['Transfer']@dtstar
path_r = r_steady + G['r']['Transfer']@dtstar
path_div = Div_steady + G['Div']['Transfer']@dtstar
path_n = N_steady + G['N']['Transfer']@dtstar
path_transfer = Transfer_steady + dtstar

V_prime_p = (1+r_steady)*c_steady**(-gamma)
all_c = np.zeros((nE,nA,T))

for t in range(299,-1,-1):
    #print(t)
    V_prime_p,_,c,_ = iterate_household(household_d,V_prime_p,Pi,a_grid,path_w[t],taun,pi_e,
                            e_grid,path_r[t],path_div[t],path_transfer[t],beta,gamma,nu,tauc)
    all_c[:,:,t] = c
    
all_c_devi = np.copy(all_c)

for l in range(T):

    all_c_devi[:,:,l] = all_c[:,:,l] - c_steady    

c_full_dist = all_c_devi[:,:,0]#np.sum(all_c_devi,2) #
c_asset_dist = np.zeros(nA)

for i in range(nA):
    c_asset_dist[i] = D[:,i]@c_full_dist[:,i]

c_asset_dist_t = np.copy(c_asset_dist)
    
##############################################################################
############################ Interest Rate Policy ############################
###############################################################################

blocks_ss = [hh_ext, firm, monetary,fiscal, nkpc_ss, mkt_clearing]

hank_ss = create_model(blocks_ss, name = "One-Asset HANK SS")


calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.92, 'nE': 8,
               'amin': 0, 'amax': 200, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 1.5, 'B': 6.0, 
               'tauc': 0.1, 'taun': 0.036}

unknowns_ss = {'beta': 0.986, 'phi': 0.8, 'Transfer': 0.05}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0, 'govt_res': 0}

ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver = "hybr")

blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc]
hank = create_model(blocks, name = "One-Asset HANK")
#print(*hank.blocks, sep='\n')
ss = hank.steady_state(ss0)

#for k in ss0.keys():
#    assert np.all(np.isclose(ss[k], ss0[k]))
    
T = 300
exogenous = ['rstar', 'Z', 'tauc']
unknowns = ['pi', 'w', 'Y', 'Transfer']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt', 'govt_res']

# general equilibrium jacobians
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

shock = np.zeros(T)
discount = (1 / (1 + ss['r']))
#discount = 1
#A, B, C, D, E = 1, 0.5, 0.19499, 5, 3
A, B, C, D, E = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (A - B * (x - E)) * np.exp(-C * (x - E) - D) 
    
z_grid = ss0.internals['household']['z_grid']
e_grid = ss0.internals['household']['e_grid']
a_grid = ss0.internals['household']['a_grid']
pi_e =  ss0.internals['household']['pi_e']
Pi = ss0.internals['household']['Pi']

nu = ss0['nu']
beta = ss0['beta']
gamma = ss0['gamma']
tauc = ss0['tauc']
taun = ss0['taun']
nE = ss0['nE']
nA = ss0['nA']

c_steady = ss0.internals['household']['c']
r_steady = ss0['r']
Transfer_steady = ss0['Transfer']
Div_steady = ss0['Div']
T_steady = transfers(pi_e,Div_steady,Transfer_steady,e_grid)
w_steady = ss0['w']
N_steady = ss0['N']

rhos = 0.9
dtstar = shock

path_w = w_steady + G['w']['Transfer']@dtstar
path_r = r_steady + G['r']['Transfer']@dtstar
path_div = Div_steady + G['Div']['Transfer']@dtstar
path_n = N_steady + G['N']['Transfer']@dtstar
path_transfer = Transfer_steady + dtstar

V_prime_p = (1+r_steady)*c_steady**(-gamma)
all_c = np.zeros((nE,nA,T))

for t in range(299,-1,-1):
    #print(t)
    V_prime_p,_,c,_ = iterate_household(household_d,V_prime_p,Pi,a_grid,path_w[t],taun,pi_e,
                            e_grid,path_r[t],path_div[t],path_transfer[t],beta,gamma,nu,tauc)
    all_c[:,:,t] = c
    
all_c_devi = np.copy(all_c)

for l in range(T):

    all_c_devi[:,:,l] = all_c[:,:,l] - c_steady    
    
c_asset_dist = np.zeros(nA)

for i in range(nA):
    c_asset_dist[i] = D[:,i]@c_full_dist[:,i]
    
wealth_perc = grids.agrid(amin = 0,amax=1,n=nA)

plt.plot(wealth_perc,c_asset_dist, label = "Interest Rate Shock")
plt.plot(wealth_perc,c_asset_dist_t, label = "Transfer Shock")
plt.legend()
plt.show()