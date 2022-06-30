#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:29:50 2022

@author: danielcoutinho
"""

import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import het, simple, create_model    # functions
from sequence_jacobian import interpolate, grids           # modules
import functions_dist_agents as dist

def chebyschev_grid(amin, amax, n): # see Judd(1998, p. 223)
    grid = np.linspace(1, n, num=n)
    cheby_node = -np.cos((2 * grid - 1) / (2 * n) * np.pi)
    adj_node = (cheby_node + 1) * (amax - amin) / 2 + amin
    return adj_node

# Household heterogeneous block
def household_guess(a_grid, z_grid, gamma, r, T, tauc):
    new_z = np.ones((z_grid.shape[0],1))
    wel = (1 + r) * a_grid[np.newaxis,:] + new_z + T[:,np.newaxis]
    V_prime = (1 + r) / (1 + tauc) * (wel * 0.1) ** (-gamma)
    return V_prime

@het(exogenous='Pi', policy ='a', backward='V_prime', backward_init=household_guess)
def household(V_prime_p, a_grid, e_grid, z_grid, beta, gamma, r, T, tauc):
    c_prime = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma) # c_prime is the new guess for c_t
    new_grid = (1 + tauc) * c_prime + a_grid[np.newaxis, :] - z_grid[:, np.newaxis] - T[:, np.newaxis]
    wel = (1 + r) * a_grid
    c = interpolate.interpolate_y(new_grid, wel, c_prime)
    a = wel + z_grid[:, np.newaxis] + T[:, np.newaxis] - (1 + tauc) * c
    V_prime = (1 + r) / (1 + tauc) * c ** (-gamma)

    # Check for violation of the asset constraint and adjust policy rules
    indexes_asset = np.nonzero(a < a_grid[0])
    if indexes_asset[0].size != 0 and indexes_asset[1].size !=0:  
        a[indexes_asset] = a_grid[0]
        c[indexes_asset] = (wel[indexes_asset[1]] + z_grid[indexes_asset[0]] + T[indexes_asset[0]] - a[indexes_asset]) / (1 + tauc)
        V_prime[indexes_asset] = (1 + r) / (1 + tauc) * (c[indexes_asset]) ** (-gamma)
    uce = e_grid[:, np.newaxis] * c ** (-gamma)
    return V_prime, a, c, uce

def income(e_grid, N, taun, w):
    z_grid = (1 - taun) * w * N * e_grid
    return z_grid

def make_grid(amin, amax, nA, nE, rho_e, sd_e):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA) # original grid used by Auclert et al
    # a_grid = chebyschev_grid(amin=amin, amax=amax, n=nA) # Chebyshev grid
    return a_grid, e_grid, Pi, pi_e

def transfers(pi_e, Div, Tau, e_grid):
    tau_rule, div_rule = np.ones(e_grid.size), e_grid # uniform transfer, dividend proportional to productivity
    # tau_rule, div_rule = np.ones(e_grid.size), np.array((0, 0, 0, 0, 1, 1, 1, 1)) # all for the rich
    # tau_rule, div_rule = np.array((1/pi_e[7], 0, 0, 0, 0, 0, 0, 0)), np.array((1/pi_e[7], 0, 0, 0, 0, 0, 0, 0)) # all for the poor
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tau =  Tau / np.sum(pi_e * tau_rule) * tau_rule 
    T = div + tau
    return T

hh_inp = household.add_hetinputs([make_grid, transfers, income])

# Simple blocks
@simple
def firm(kappa, mu, pi, w, Y, Z):
    N = Y / Z
    Div = Y - w * N - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    return Div, N

@simple
def monetary(phi_pi, pi, rstar):
    r = (1 + rstar(-1) + phi_pi * pi(-1)) / (1 + pi) - 1
    i = rstar
    return r, i

@simple
def monetary2(phi_pi, pi, r_ss, rstar):
    r = (1 + rstar(-1) + phi_pi * pi(-1)) / (1 + pi) - 1
    i = rstar
    r_resid = r - r_ss
    return r, i, r_resid

@simple
def fiscal(B, C, N, r, Tau, tauc, taun, w):
    govt_res = Tau + (1 + r) * B(-1) - tauc * C - taun * w * N - B
    Deficit = Tau - tauc * C - taun * w * N # primary deficit
    Trans = Tau
    return govt_res, Deficit, Trans

@simple
def fiscal2(B, B_ss, N, r, r_ss, rhot, Tau, tauc, taun, w):
    # Tau = taun * w * N + B - (1 + r) * B(-1) # immediate adjustment of transfers, no tauc
    govt_res = Tau - (taun * w * N + B - (1 + r) * B(-1)) 
    # govt_res = Tau - rhot * Tau(-1) - (1 - rhot) * (taun * w * N + B - (1 + r) * B(-1)) # delayed adjustment of transfers
    Deficit = Tau - taun * w * N + (1 + r) * B(-1) - B # primary deficit, no tauc
    fiscal_rule = (B - B_ss) - (B(-1) - B_ss) - rhot * (r - r_ss)  # delayed adjustment of transfers
    Trans = Tau
    return Deficit, Trans, govt_res, fiscal_rule

@simple
def mkt_clearing(mu, kappa, A, B, C, pi, Y):
    asset_mkt = A - B
    goods_mkt = Y - C - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    return asset_mkt, goods_mkt

@simple
def nkpc_ss(mu, Z):
    w = Z / mu
    return w

@simple 
def union_ss(kappaw, muw, nu, N, UCE, tauc, taun, w):
    phi = ((1 - taun) * w * N ** (-nu) * UCE) / ((1 + tauc) * muw)
    wnkpc = kappaw * (phi * N ** (1 + nu) - (1 - taun) * w * N * UCE / ((1 + tauc) * muw))
    return wnkpc, phi

@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw

@simple
def union(beta, kappaw, muw, nu, phi, piw, N, UCE, tauc, taun, w):
    wnkpc = (kappaw * (phi * N ** (1+nu) - (1 - taun) * w * N * UCE / ((1 + tauc) * muw)) 
             + beta * (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log))
    return wnkpc

@simple
def nkpc(kappa, mu, pi, r, w, Y, Z):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc_res

def household_d(V_prime_p, a_grid, e_grid, z_grid, beta, gamma, r, T, tauc):   
    c_prime = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma) # c_prime is the new guess for c_t
    new_grid = (1 + tauc) * c_prime + a_grid[np.newaxis, :] - z_grid[:, np.newaxis] - T[:, np.newaxis]
    wel = (1 + r) * a_grid
    c = interpolate.interpolate_y(new_grid, wel, c_prime)
    a = wel + z_grid[:, np.newaxis] + T[:, np.newaxis] - (1 + tauc) * c
    V_prime = (1 + r) / (1 + tauc) * c ** (-gamma)

    # Check for violation of the asset constraint and adjust policy rules
    indexes_asset = np.nonzero(a < a_grid[0])
    if indexes_asset[0].size != 0 and indexes_asset[1].size !=0:  
        a[indexes_asset] = a_grid[0]
        c[indexes_asset] = (wel[indexes_asset[1]] + z_grid[indexes_asset[0]] 
                            + T[indexes_asset[0]] - a[indexes_asset]) / (1 + tauc)
        V_prime[indexes_asset] = (1 + r) / (1 + tauc) * (c[indexes_asset]) ** (-gamma) # check
    uce = e_grid[:,np.newaxis] * c ** (-gamma)
    return V_prime, a, c, uce

def iterate_h(foo, V_prime_start, a_grid, e_grid, Pi, pi_e, beta, gamma,
              Div, N, r, Tau, tauc, taun, w, maxit=1000, tol=1E-8):
    ite = 0
    err = 1
    V_prime_p = Pi @ V_prime_start # Pi is the markov chain transition matrix
    V_prime_old = V_prime_start # Initialize, V_prime_start will be set to ss V_prime_p
    T = transfers(pi_e, Div, Tau, e_grid)
    z_grid = income(e_grid, N, taun, w)
    
    while ite < maxit and err > tol:
        V_prime_temp, a, c, uce = foo(V_prime_p, a_grid, e_grid, z_grid, beta, gamma, r, T, tauc) # foo is a placeholder
        V_prime_p = Pi @ V_prime_temp
        ite += 1
        err = np.max(np.abs(V_prime_old - V_prime_temp))
        V_prime_old = V_prime_temp
    return V_prime_temp, a, c, uce 



# Steady state
blocks_ss_tau = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc_ss, union_ss]
hank_ss_tau = create_model(blocks_ss_tau, name="One-Asset HANK SS")
calib_tau = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 8,
               'amin': 0, 'amax': 150, 'nA': 50, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.05, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0,
               'kappaw': 0.06, 'muw': 1.2, 'N': 1.0, 'tauc': 0.1, 'taun': 0.05}
unknowns_ss_tau = {'beta': 0.986, 'Tau': 0.02}
targets_ss_tau = {'asset_mkt': 0, 'govt_res': 0}

ss0_tau = hank_ss_tau.solve_steady_state(calib_tau, unknowns_ss_tau, targets_ss_tau, backward_tol=1E-22, solver="hybr")


# Dynamic model
blocks_tau = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc, wage, union]
hank_tau = create_model(blocks_tau, name="One-Asset HANK")
ss_tau = hank_tau.steady_state(ss0_tau)
T = 300
exogenous_tau = ['rstar','Tau', 'Z', 'tauc']
unknowns_tau = ['pi', 'w', 'Y', 'B']
targets_tau = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']

G_tau = hank_tau.solve_jacobian(ss_tau, unknowns_tau, targets_tau, exogenous_tau, T=T)

B_ss_tau = ss_tau['B']
C_ss_tau = ss_tau['C']
D_ss_tau = ss_tau['Deficit']
Div_ss_tau = ss_tau['Div']
N_ss_tau = ss_tau['N']
r_ss_tau = ss_tau['r']
T_ss_tau = ss_tau['Trans']
Tau_ss_tau = ss_tau['Tau']
w_ss_tau = ss_tau['w']
Y_ss_tau = ss_tau['Y']

discount = (1 / (1 + r_ss_tau))

# Zero net present value sock
shock = np.zeros(T)
s1, s2, s3, s4, s5 = 1, 0.5, 0.1723464735, 5, 3 # r_ss discount factor
#s1, s2, s3, s4, s5 = 1, 0.5, 0.1608615, 5, 3 # beta discount factor
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
dtau = shock

beta = ss_tau['beta']
tauc = ss_tau['tauc']
taun = ss_tau['taun']
gamma = ss_tau['gamma']
nE = ss_tau['nE']
nA = ss_tau['nA']

# Steady-state variables
a_grid_tau = ss_tau.internals['household']['a_grid']
e_grid_tau = ss_tau.internals['household']['e_grid']
c_ss_tau = ss_tau.internals['household']['c']
Pi_tau = ss_tau.internals['household']['Pi']
pi_e_tau = ss_tau.internals['household']['pi_e']
# D_ss_tau = ss_tau.internals['household']['Dbeg']
D_ss_tau = ss_tau.internals['household']['Dbeg']

# Aggregate transition dynamics
path_div_tau = Div_ss_tau + G_tau['Div']['Tau'] @ dtau
path_n_tau = N_ss_tau + G_tau['N']['Tau'] @ dtau
path_r_tau = r_ss_tau + G_tau['r']['Tau'] @ dtau
path_tau_tau = Tau_ss_tau + dtau
path_w_tau = w_ss_tau + G_tau['w']['Tau'] @ dtau

# Aggregate dynamics, multiplicative shock
# path_div_tau = Div_ss_tau * (1 + G_tau['Div']['Tau'] @ dtau)
# path_n_tau = N_ss_tau * (1 + G_tau['N']['Tau'] @ dtau)
# path_r_tau = r_ss_tau * (1 + G_tau['r']['Tau'] @ dtau)
# path_tau_tau = Tau_ss_tau * (1 + dtau)
# path_w_tau = w_ss_tau * (1 + G_tau['w']['Tau'] @ dtau)

# Compute all individual consumption paths
print("Computing individual paths...", end=" ")
V_prime_p_tau = (1 + r_ss_tau) / (1 + tauc) * c_ss_tau ** (-gamma)
c_all_tau = np.zeros((nE, nA, T))
a_all_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, a, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                        path_div_tau[t], path_n_tau[t], path_r_tau[t], path_tau_tau[t], tauc, taun, path_w_tau[t])
    c_all_tau[:, :, t] = c  
    a_all_tau[:,:,t] = a
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
V_prime_p_tau = (1 + r_ss_tau) / (1 + tauc) * c_ss_tau ** (-gamma)
c_direct_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                        Div_ss_tau, N_ss_tau, r_ss_tau, path_tau_tau[t], tauc, taun, w_ss_tau)
    # V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                        # Div_ss_tau, N_ss_tau, r_ss_tau, Tau_ss_tau, tauc, taun, w_ss_tau)
    c_direct_tau[:, :, t] = c
print("Done")


econ_ss = dist.start_from_steady(1_000_000, D_ss_tau, e_grid_tau, a_grid_tau)
period_0_tau = dist.update_economy(econ_ss, a_grid_tau, e_grid_tau, a_all_tau[:,:,0], Pi_tau)
dist_0_tau = dist.compute_distribution(period_0_tau, a_grid_tau, nE)

print(np.sum(c_all_tau[:,:,0]*dist_0_tau))
print(G_tau['C']['Tau'][0][0] * dtau[0] + ss0_tau['C'])

c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau
c_first_dev_tau_direct = (c_direct_tau[:, :, 0] - c_ss_tau) / c_ss_tau

c_first_tau, c_first_tau_direct = np.zeros(nA), np.zeros(nA)
for i in range(nA):
    c_first_tau[i] = (c_first_dev_tau[:, i] @ dist_0_tau[:, i]) / np.sum(dist_0_tau[:,i])
    c_first_tau_direct[i] = (c_first_dev_tau_direct[:, i] @ dist_0_tau[:, i]) / np.sum(dist_0_tau[:,i])
    
c_first_bin_tau = c_first_tau.reshape(-1, nA, order='F').mean(axis=0)  
c_first_bin_tau_direct = c_first_tau_direct.reshape(-1, nA, order='F').mean(axis=0) 
c_first_bin_tau_indirect = c_first_bin_tau - c_first_bin_tau_direct

D_quant_tau = 100 * np.cumsum(np.sum(dist_0_tau, axis=0))
