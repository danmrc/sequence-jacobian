#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:24:39 2022

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

blocks_ss_rstar = [hh_inp, firm, monetary2, fiscal2, mkt_clearing, nkpc_ss, union_ss]
hank_ss_rstar = create_model(blocks_ss_rstar, name = "One-Asset HANK SS")
calib_rstar = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 8,
               'amin': 0, 'amax': 150, 'nA': 50, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.05, 'rstar': 0.005, 'phi_pi': 1.5, 'B': 6.0,
               'kappaw': 0.06, 'muw': 1.2, 'N': 1.0, 'tauc': 0.0, 'taun': 0.05, 'rhot': 0.0}
# unknowns_ss_rstar = {'beta': 0.986, 'Tau': -0.03}
# targets_ss_rstar = {'asset_mkt': 0, 'govt_res': 0}
unknowns_ss_rstar = {'beta': 0.986, 'Tau': 0.02, 'B_ss': 6.0, 'r_ss': 0.005}
targets_ss_rstar = {'asset_mkt': 0, 'govt_res': 0, 'fiscal_rule': 0, 'r_resid': 0}
ss0_rstar = hank_ss_rstar.solve_steady_state(calib_rstar, unknowns_ss_rstar, targets_ss_rstar, backward_tol=1E-22, solver="hybr")

# Dynamic model
blocks_rstar = [hh_inp, firm, monetary2, fiscal2, mkt_clearing, nkpc, wage, union]
hank_rstar = create_model(blocks_rstar, name = "One-Asset HANK")
ss_rstar = hank_rstar.steady_state(ss0_rstar)
T = 300
exogenous_rstar = ['rstar', 'Z']
# unknowns_rstar = ['pi', 'w', 'Y', 'Tau']
# targets_rstar = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']
unknowns_rstar = ['pi', 'w', 'Y', 'B', 'Tau']
targets_rstar = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res', 'fiscal_rule']

G_rstar = hank_rstar.solve_jacobian(ss_rstar, unknowns_rstar, targets_rstar, exogenous_rstar, T=T)

B_ss_rstar = ss_rstar['B']
C_ss_rstar = ss_rstar['C']
D_ss_rstar = ss_rstar['Deficit']
Div_ss_rstar = ss_rstar['Div']
N_ss_rstar = ss_rstar['N']
r_ss_rstar = ss_rstar['r']
T_ss_rstar = ss_rstar['Trans']
Tau_ss_rstar = ss_rstar['Tau']
w_ss_rstar = ss_rstar['w']
Y_ss_rstar = ss_rstar['Y']

rhos = 0.64
# dtau = 0.01 * rhos ** np.arange(T)
drstar = -0.0019 * rhos ** np.arange(T)

beta = ss_rstar['beta']
tauc = ss_rstar['tauc']
taun = ss_rstar['taun']
gamma = ss_rstar['gamma']
nE = ss_rstar['nE']
nA = ss_rstar['nA']

# Steady-state variables
a_grid_rstar = ss_rstar.internals['household']['a_grid']
e_grid_rstar = ss_rstar.internals['household']['e_grid']
c_ss_rstar = ss_rstar.internals['household']['c']
Pi_rstar = ss_rstar.internals['household']['Pi']
pi_e_rstar = ss_rstar.internals['household']['pi_e']
# D_ss_rstar = ss_rstar.internals['household']['Dbeg']
D_ss_rstar = ss_rstar.internals['household']['Dbeg']

# Aggregate transition dynamics
path_div_rstar = Div_ss_rstar + G_rstar['Div']['rstar'] @ drstar
path_n_rstar = N_ss_rstar + G_rstar['N']['rstar'] @ drstar
path_r_rstar = r_ss_rstar + G_rstar['r']['rstar'] @ drstar
path_tau_rstar = Tau_ss_rstar + G_rstar['Tau']['rstar'] @ drstar
path_w_rstar = w_ss_rstar + G_rstar['w']['rstar'] @ drstar

# Aggregate dynamics, multiplicative shock
# path_div_rstar = Div_ss_rstar * (1 + G_rstar['Div']['rstar'] @ drstar)
# path_n_rstar = N_ss_rstar * (1 + G_rstar['N']['rstar'] @ drstar)
# path_r_rstar = r_ss_rstar * (1 + G_rstar['r']['rstar'] @ drstar)
# path_tau_rstar = Tau_ss_rstar * (1 + G_rstar['Tau']['rstar'] @ drstar)
# path_w_rstar = w_ss_rstar * (1 + G_rstar['w']['rstar'] @ drstar)

# Compute all individual consumption paths
V_prime_p_rstar = (1 + r_ss_rstar) / (1 + tauc) * c_ss_rstar ** (-gamma)
c_all_rstar = np.zeros((nE, nA, T))
a_all_rstar = np.zeros((nE,nA,T))
for t in range(T-1, -1, -1):
    V_prime_p_rstar, a, c, _ = iterate_h(household_d, V_prime_p_rstar, a_grid_rstar, e_grid_rstar, Pi_rstar, pi_e_rstar, beta, gamma,
                                          path_div_rstar[t], path_n_rstar[t], path_r_rstar[t], path_tau_rstar[t], tauc, taun, path_w_rstar[t])
    c_all_rstar[:, :, t] = c
    a_all_rstar[:, :, t] = a

# Direct effect of policy
V_prime_p_rstar = (1 + r_ss_rstar) / (1 + tauc) * c_ss_rstar ** (-gamma)
c_direct_rstar = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_rstar, _, c, _ = iterate_h(household_d, V_prime_p_rstar, a_grid_rstar, e_grid_rstar, Pi_rstar, pi_e_rstar, beta, gamma,
                                          Div_ss_rstar, N_ss_rstar, path_r_rstar[t], Tau_ss_rstar, tauc, taun, w_ss_rstar)
    c_direct_rstar[:, :, t] = c


econ_ss = dist.start_from_steady(1_000_000, D_ss_rstar, e_grid_rstar, a_grid_rstar)
period_0_rstar = dist.update_economy(econ_ss, a_grid_rstar, e_grid_rstar, a_all_rstar[:,:,0], Pi_rstar)
dist_0_rstar = dist.compute_distribution(period_0_rstar, a_grid_rstar, nE)

print(np.sum(c_all_rstar[:,:,0]*dist_0_rstar))
print(G_rstar['C']['rstar'][0][0] * drstar[0] + ss0_rstar['C'])

c_first_dev_rstar = (c_all_rstar[:, :, 0] - c_ss_rstar) / c_ss_rstar
c_first_dev_rstar_direct = (c_direct_rstar[:, :, 0] - c_ss_rstar) / c_ss_rstar

c_first_rstar, c_first_rstar_direct = np.zeros(nA), np.zeros(nA)
for i in range(nA):
    c_first_rstar[i] = (c_first_dev_rstar[:, i] @ dist_0_rstar[:, i]) / np.sum(dist_0_rstar[:,i])
    c_first_rstar_direct[i] = (c_first_dev_rstar_direct[:, i] @ dist_0_rstar[:, i]) / np.sum(dist_0_rstar[:,i])
    
c_first_bin_rstar = c_first_rstar.reshape(-1, nA, order='F').mean(axis=0)  
c_first_bin_rstar_direct = c_first_rstar_direct.reshape(-1, nA, order='F').mean(axis=0) 
c_first_bin_rstar_indirect = c_first_bin_rstar - c_first_bin_rstar_direct

D_quant = 100 * np.cumsum(np.sum(dist_0_rstar, axis=0))

color_map = ["#FFFFFF", "#D95319"] # myb: "#0072BD"
fig, ax = plt.subplots(1,2)
ax[0].set_title(r'Interest rate policy')
ax[0].plot(D_quant, 100 * c_first_bin_rstar_direct, label="Direct effect", linewidth=3)  
ax[0].stackplot(D_quant, 100 * c_first_bin_rstar_direct, 100 * c_first_bin_rstar_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)  
ax[0].legend(loc='upper left', frameon=False)
ax[0].set_xlabel("Wealth percentile"), ax[0].set_ylabel("Percent deviation from steady state")






