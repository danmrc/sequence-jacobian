"""Distributional responses to shocks in HANK"""

# =============================================================================
# Initialize
# =============================================================================

print("BASELINE MODEL")
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import het, simple, create_model             # functions
from sequence_jacobian import interpolate, grids, misc, estimation  # modules

# Bisection
def bisection_onestep(f,a,b):
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

def vec_bisection(f, a, b, iter_max=100, tol=1E-11):
    i = 1
    err = 1
    while i < iter_max and err > tol:
        a,b = bisection_onestep(f,a,b)
        err = np.max(np.abs(a - b))
        i += 1
    if i >= iter_max:
        raise ValueError("No convergence")
    return a


# =============================================================================
# Household heterogeneous block
# =============================================================================

def consumption(c, we, rest, gamma, nu, phi, tauc, taun):
    return (1 + tauc) * c - (1 - taun) * we * ((1 - taun) * we / ((1 + tauc) * phi * c ** gamma)) ** (1/nu) - rest

def household_guess(a_grid, e_grid, r, w, gamma, T, tauc, taun):
    wel = (1 + r) * a_grid[np.newaxis,:] + (1 - taun) * w * e_grid[:,np.newaxis] + T[:,np.newaxis]
    V_prime = (1 + r) / (1 + tauc) * (wel * 0.1) ** (-gamma) # check
    return V_prime

@het(exogenous='Pi', policy='a', backward='V_prime', backward_init=household_guess)
def household(V_prime_p, a_grid, e_grid, r, w, T, beta, gamma, nu, phi, tauc, taun):
    we = w * e_grid
    c_prime = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma) # c_prime is the new guess for c_t
    n_prime = ((1 - taun) * we[:,np.newaxis] / ((1 + tauc) * phi * c_prime ** gamma)) ** (1/nu)
    new_grid = ((1 + tauc) * c_prime + a_grid[np.newaxis,:] - (1 - taun) * we[:,np.newaxis] * n_prime 
                - T[:,np.newaxis])
    wel = (1 + r) * a_grid
    c = interpolate.interpolate_y(new_grid, wel, c_prime)
    n = interpolate.interpolate_y(new_grid, wel, n_prime)
    a = wel + (1 - taun) * we[:,np.newaxis] * n + T[:,np.newaxis] - (1 + tauc) * c
    V_prime = (1 + r) / (1 + tauc) * c ** (-gamma)

    # Check for violation of the asset constraint and adjust policy rules
    indexes_asset = np.nonzero(a < a_grid[0]) # first dimension: labor grid, second dimension: asset grid
    a[indexes_asset] = a_grid[0]

    if indexes_asset[0].size != 0 and indexes_asset[1].size !=0:
        aa = np.zeros((indexes_asset[0].size)) + 1E-5
        rest = wel[indexes_asset[1]] - a_grid[0] + T[indexes_asset[0]]
        bb = c[indexes_asset] + 0.5
        c[indexes_asset] = vec_bisection(lambda c : consumption(c, we[indexes_asset[0]], rest,
                                                                gamma, nu, phi, tauc, taun), aa, bb)
        n[indexes_asset] = ((1 - taun) * we[indexes_asset[0]] 
                            / ((1 + tauc) * phi * c[indexes_asset] ** gamma)) ** (1/nu)
        V_prime[indexes_asset] = (1 + r) / (1 + tauc) * (c[indexes_asset]) ** (-gamma)
    return V_prime, a, c, n

def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi, a_grid, pi_e

def transfers(pi_e, Div, Tau, e_grid):
    tax_rule, div_rule = np.ones(e_grid.size), e_grid #np.ones(e_grid.size)
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tau =  Tau / np.sum(pi_e * tax_rule) * tax_rule 
    T = div + tau
    return T

household_inp = household.add_hetinputs([make_grid, transfers])

def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne

hh_ext = household_inp.add_hetoutputs([labor_supply])


# =============================================================================
# Simple blocks
# =============================================================================

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
def fiscal(r, Tau, B, C, L, tauc, taun, w):
    govt_res = Tau + (1 + r) * B(-1) - tauc * C - taun * w * L - B
    Deficit = tauc * C + taun * w * L - Tau # primary surplus
    Trans = Tau
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


# =============================================================================
# Steady state
# =============================================================================

blocks_ss = [hh_ext, firm, monetary,fiscal, nkpc_ss, mkt_clearing]
hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.50, 'nE': 8,
               'amin': 0, 'amax': 180, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0, 
               'tauc': 0.1, 'taun': 0.036}

unknowns_ss = {'beta': 0.986, 'phi': 0.8, 'Tau': 0.05}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0, 'govt_res': 0}
print("Computing steady state...")
ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")
print("Done")


# =============================================================================
# Dynamic model and Jacobian
# =============================================================================

blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc]
hank = create_model(blocks, name="One-Asset HANK")
ss = hank.steady_state(ss0)

T = 300
exogenous = ['rstar', 'Tau', 'Z', 'tauc']
unknowns = ['pi', 'w', 'Y', 'B']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt', 'govt_res']

print("Computing Jacobian...")
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)
print("Done")


# =============================================================================
# Household iteration policy rule
# =============================================================================

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
    V_prime = (1 + r) / (1 + tauc) * c ** (-gamma)

    # Check for violation of the asset constraint and adjust policy rules
    indexes_asset = np.nonzero(a < a_grid[0]) # first dimension: labor grid, second dimension: asset grid
    a[indexes_asset] = a_grid[0]

    if indexes_asset[0].size != 0 and indexes_asset[1].size !=0:
        aa = np.zeros((indexes_asset[0].size)) + 1E-5
        rest = wel[indexes_asset[1]] - a_grid[0] + T[indexes_asset[0]]
        bb = c[indexes_asset] + 0.5
        c[indexes_asset] = vec_bisection(lambda c : consumption(c,we[indexes_asset[0]],
                                                                 rest,gamma,nu,phi,tauc,taun),aa,bb)
        n[indexes_asset] = ((1 - taun) * we[indexes_asset[0]] 
                            / ((1 + tauc) * phi * c[indexes_asset] ** gamma)) ** (1/nu)
        V_prime[indexes_asset] = (1 + r) / (1 + tauc) * (c[indexes_asset]) ** (-gamma)
    return V_prime, a, c, n

def iterate_household(foo, V_prime_start, Pi, a_grid, w, taun, pi_e, e_grid, r, Div, 
                      Tau, beta, gamma, nu, phi, tauc, maxit=1000, tol=1E-8):
    V_prime_p = Pi @ V_prime_start
    V_prime_old = V_prime_start    
    ite = 0
    err = 1
    T = transfers(pi_e, Div, Tau, e_grid)
    
    while ite < maxit and err > tol:
        #c_old = np.copy(c)
        V_prime_temp, a, c, n = foo(V_prime_p, a_grid, e_grid, r, w, T, beta, gamma, nu, phi, tauc, taun)
        V_prime_p = Pi @ V_prime_temp
        ite += 1
        err = np.max(np.abs(V_prime_old - V_prime_temp))
        V_prime_old = V_prime_temp 
    #print("Iteration ", ite, " out of ", maxit, "\n Difference in policy (sup norm):", err)
    return V_prime_temp, a, c, n


# =============================================================================
# Common to both policies
# =============================================================================

# Parameters
beta = ss['beta']
gamma = ss['gamma']
nE = ss['nE']
nA = ss['nA']
nu = ss['nu']
phi = ss['phi']
tauc = ss['tauc']
taun = ss['taun']

# Aggregate steady-state variables
Div_ss = ss['Div']
N_ss = ss['N']
r_ss = ss['r']
Tau_ss = ss['Tau']
w_ss = ss['w']

# Distributional steady-state variables
e_grid = ss.internals['household']['e_grid']
a_grid = ss.internals['household']['a_grid']
D_ss = ss.internals['household']['Dbeg']
pi_e =  ss.internals['household']['pi_e']
Pi = ss.internals['household']['Pi']
a_ss = ss.internals['household']['a']
c_ss = ss.internals['household']['c']
n_ss = ss.internals['household']['n']
T_ss = transfers(pi_e, Div_ss, Tau_ss, e_grid)

# Sort assets into bins
nbin = np.zeros(100)
nbin[0], nbin[1], nbin[2], nbin[3] = 0, 0.0000001, 0.5, 1
nbin[3:] = np.arange(1, ss['amax'], ss['amax'] / 97)
nbin = a_grid
# nbin = np.arange(0, ss['amax'], ss['amax'] / 100)
a_bin = np.digitize(a_ss, nbin)

# Zero net present value shock
shock = np.zeros(T)
discount = (1 / (1 + r_ss))
s1, s2, s3, s4, s5 = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 


# =============================================================================
# Simulating policies
# =============================================================================

# Consumption tax policy
print("TAX POLICY")
# dtauc = - shock 
rhos = 0.9
dtauc = - 0.03 * rhos ** (np.arange(T)[:, np.newaxis])
    
# Aggregate transition dynamics
path_n = N_ss + G['N']['tauc'] @ dtauc
path_r = r_ss + G['r']['tauc'] @ dtauc
path_w = w_ss + G['w']['tauc'] @ dtauc
path_div = Div_ss + G['Div']['tauc'] @ dtauc
path_tauc = tauc + dtauc

# Initialize individual consumption paths
V_prime_p = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
c_all_tauc, n_all_tauc = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))

# Compute all individual consumption paths
print("Computing individual paths...")
for t in range(T-1, -1, -1):
    V_prime_p, _, c, n = iterate_household(household_d, V_prime_p, Pi, a_grid, path_w[t], taun, pi_e,
                            e_grid, path_r[t], path_div[t], Tau_ss, beta, gamma, nu, phi, path_tauc[t])
    c_all_tauc[:, :, t] = c  
    n_all_tauc[:, :, t] = n
print("Done")

# Transfer policy
print("TRANSFER POLICY")
# dtau = shock
dtau = - dtauc
    
# Aggregate transition dynamics
path_n = N_ss + G['N']['Tau'] @ dtau
path_r = r_ss + G['r']['Tau'] @ dtau
path_w = w_ss + G['w']['Tau'] @ dtau
path_div = Div_ss + G['Div']['Tau'] @ dtau
path_tau = Tau_ss + dtau

# Initialize individual consumption paths
V_prime_p = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
c_all_tau, n_all_tau = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))

# Compute all individual consumption paths
print("Computing individual paths...")
for t in range(T-1, -1, -1):
    V_prime_p, _, c, n = iterate_household(household_d, V_prime_p, Pi, a_grid, path_w[t], taun, pi_e,
                            e_grid, path_r[t], path_div[t], path_tau[t], beta, gamma, nu, phi, tauc)
    c_all_tau[:,:,t] = c
    n_all_tau[:, :, t] = n
print("Done")

# # Interest rate policy
# print("INTEREST RATE POLICY")
# drstar = -0.02 * rhos ** (np.arange(T)[:, np.newaxis])

# # Aggregate transition dynamics
# path_n = N_ss + G['N']['rstar'] @ drstar
# path_r = r_ss + G['r']['rstar'] @ drstar
# path_w = w_ss + G['w']['rstar'] @ drstar
# path_div = Div_ss + G['Div']['rstar'] @ drstar
# path_tau = Tau_ss + G['Tau']['rstar'] @ drstar

# # Initialize individual consumption paths
# V_prime_p = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
# c_all_rstar, n_all_rstar = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))

# # Compute all individual consumption paths
# print("Computing individual paths...")
# for t in range(T-1, -1, -1):
#     V_prime_p, _, c, n = iterate_household(household_d, V_prime_p, Pi, a_grid, path_w[t], taun, pi_e,
#                             e_grid, path_r[t], path_div[t], path_tau[t], beta, gamma, nu, phi, tauc)
#     c_all_rstar[:, :, t] = c  
#     n_all_rstar[:, :, t] = n
# print("Done")


# =============================================================================
# Impact response by wealth percentile
# =============================================================================
    
# Select first period only and express as deviation from steady state
c_first_dev_tauc = (c_all_tauc[:, :, 0] - c_ss) / c_ss
n_first_dev_tauc = (n_all_tauc[:, :, 0] - n_ss) / n_ss
c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss) / c_ss
n_first_dev_tau = (n_all_tau[:, :, 0] - n_ss) / n_ss

# Method 1: Weigh response by mass of agents
c_first_tauc, n_first_tauc, c_first_tau, n_first_tau = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
for i in range(nA):
    c_first_tauc[i] = c_first_dev_tauc[:, i] @ D_ss[:, i]
    n_first_tauc[i] = n_first_dev_tauc[:, i] @ D_ss[:, i]
    c_first_tau[i] = c_first_dev_tau[:, i] @ D_ss[:, i]
    n_first_tau[i] = n_first_dev_tau[:, i] @ D_ss[:, i]
       
# Pool into percentile bins
c_first_bin_tauc = c_first_tauc.reshape(-1, 100, order='F').sum(axis=0)
n_first_bin_tauc = n_first_tauc.reshape(-1, 100, order='F').sum(axis=0)
c_first_bin_tau = c_first_tau.reshape(-1, 100, order='F').sum(axis=0)
n_first_bin_tau = n_first_tau.reshape(-1, 100, order='F').sum(axis=0)

# # Method 2: Weigh response by population/wealth quartiles CHECK IF CORRECT
# D_dist = np.sum(D_ss, axis=0)
# D_pct = np.percentile(D_dist, (25, 50, 75, 100))
# D_pos = np.searchsorted(D_dist, D_pct)

# # Asset quartiles
# a_dist = np.sum(np.multiply(a_ss, D_ss), axis=0)
# a_pct = np.percentile(a_dist, (25, 50, 75, 100))
# a_pos = np.searchsorted(a_dist, a_pct)

# c_first_tauc_q = np.zeros(len(D_pos))
# for i in range(len(D_pos)):
#     if i == 0:
#         c_first_tauc_q[i] = np.sum(np.multiply(c_first_dev_tauc[:, a_pos[i]], D_ss[:, a_pos[i]]))
#     else:
#         c_first_tauc_q[i] = np.sum(np.multiply(c_first_dev_tauc[:, a_pos[i-1]+1:D_pos[i]], D_ss[:, a_pos[i-1]+1:D_pos[i]]))
  
# plt.plot(c_first_tauc_q)
# plt.show()    
  
# Method 3: Weigh response by mass of agents (gives same results as method 1)
a_flag, D_dist = np.zeros((nE, nA, len(nbin))), np.zeros((nE, nA, len(nbin)))
c_dist_tauc, n_dist_tauc, c_dist_tau, n_dist_tau = (np.zeros((nE, nA, len(nbin))), np.zeros((nE, nA, len(nbin))), 
                                                    np.zeros((nE, nA, len(nbin))), np.zeros((nE, nA, len(nbin))))
D_tot, c_tauc1, n_tauc1, c_tau1, n_tau1, D_tot  = (np.zeros(len(nbin)), np.zeros(len(nbin)), np.zeros(len(nbin)),
                                                    np.zeros(len(nbin)), np.zeros(len(nbin)), np.zeros(len(nbin)))

for i in range(0, len(nbin)):  
    a_flag[:, :, i] = np.where(a_bin == i+1, 1, 0) # returns 1 if true, 0 otherwise
    D_dist[:, :, i] = np.multiply(D_ss, a_flag[:, :, i])
    
    c_dist_tauc[:, :, i] = np.multiply(c_first_dev_tauc, D_dist[:, :, i])
    n_dist_tauc[:, :, i] = np.multiply(n_first_dev_tauc, D_dist[:, :, i])
    c_tauc1[i] = np.sum(c_dist_tauc[:, :, i])
    n_tauc1[i] = np.sum(n_dist_tauc[:, :, i])
    
    c_dist_tau[:, :, i] = np.multiply(c_first_dev_tau, D_dist[:, :, i])
    n_dist_tau[:, :, i] = np.multiply(n_first_dev_tau, D_dist[:, :, i])
    c_tau1[i] = np.sum(c_dist_tau[:, :, i])
    n_tau1[i] = np.sum(n_dist_tau[:, :, i])
    
# Pool into bins
c_first_bin_tauc = c_tauc1.reshape(-1, 100, order='F').sum(axis=0)
n_first_bin_tauc = n_tauc1.reshape(-1, 100, order='F').sum(axis=0)
c_first_bin_tau = c_tau1.reshape(-1, 100, order='F').sum(axis=0)
n_first_bin_tau = n_tau1.reshape(-1, 100, order='F').sum(axis=0) 

# Plot results
plt.rcParams["figure.figsize"] = (18,7)
fig, ax = plt.subplots(1,2)
# fig.suptitle(r'Impact change in consumption and labor supply, consumption tax policy $\tau_c$ versus transfer policy $\tau$', size=16)

ax[0].set_title(r'Consumption $c$')
ax[0].plot(c_first_bin_tauc * 100, label="Consumption tax policy")
# ax[0].plot(c_first_bin_rstar[0:] * 100, label="Interest rate policy")
ax[0].plot(c_first_bin_tau * 100,'-.', label="Transfer policy")
# ax[0].plot(c_bin_tauc1[0:] * 100, label="Consumption tax policy")
# ax[0].plot(c_bin_tau1[0:] * 100,'-.', label="Transfer policy")
ax[0].legend(loc='upper right', frameon=False)
ax[0].set_xlabel("Wealth percentile"), ax[0].set_ylabel("Percent deviation from steady state")

ax[1].set_title(r'Labor supply $n$')
ax[1].plot(n_first_bin_tauc * 100, label="Consumption tax policy")
# ax[1].plot(n_first_bin_rstar[0:] * 100, label="Interest rate policy")
ax[1].plot(n_first_bin_tau * 100,'-.', label="Transfer policy")
# ax[1].plot(n_bin_tauc1[0:] * 100, label="Consumption tax policy")
# ax[1].plot(n_bin_tau1[0:] * 100,'-.', label="Transfer policy")
ax[1].legend(loc='upper right', frameon=False)
ax[1].set_xlabel("Wealth percentile"), ax[1].set_ylabel("Percent deviation from steady state")
plt.show()


# =============================================================================
# Dynamic response by quantile
# =============================================================================

# Express as deviation from steady state
c_all_dev_tauc = (c_all_tauc - c_ss[:, :, None]) / c_ss[:, :, None]
n_all_dev_tauc = (n_all_tauc - n_ss[:, :, None]) / n_ss[:, :, None]
c_all_dev_tau = (c_all_tau - c_ss[:, :, None]) / c_ss[:, :, None]
n_all_dev_tau = (n_all_tau - n_ss[:, :, None]) / n_ss[:, :, None]

# # Test
# c_all_dev_tauc = (c_all_tauc - c_ss[:, :, None])
# n_all_dev_tauc = (c_all_tauc - c_ss[:, :, None])
# c_all_dev_tau = (c_all_tauc - c_ss[:, :, None])
# n_all_dev_tau = (c_all_tauc - c_ss[:, :, None])

# Weigh response by mass of agents
c_allw_tauc, n_allw_tauc, c_allw_tau, n_allw_tau = (np.zeros((nA, T)), np.zeros((nA, T)), 
                                                        np.zeros((nA, T)), np.zeros((nA, T)))
for i in range(nA):
    for t in range(T):
        c_allw_tauc[i, t] = c_all_dev_tauc[:, i, t] @ D_ss[:, i, None]
        n_allw_tauc[i, t] = n_all_dev_tauc[:, i, t] @ D_ss[:, i, None]
        c_allw_tau[i, t] = c_all_dev_tau[:, i, t] @ D_ss[:, i, None]
        n_allw_tau[i, t] = n_all_dev_tau[:, i, t] @ D_ss[:, i, None]
     
# Find quartile positions and pool responses into quartiles
D_dist = np.sum(D_ss, axis=0)
D_pct = np.percentile(D_dist, (25, 50, 75, 100))
D_pos = np.searchsorted(D_dist, D_pct)

c_all_tauc_q1 = c_allw_tauc[D_pos[0], :]
c_all_tauc_q2 = c_allw_tauc[D_pos[0]+1:D_pos[1], :].sum(axis=0)
c_all_tauc_q3 = c_allw_tauc[D_pos[1]+1:D_pos[2], :].sum(axis=0)
c_all_tauc_q4 = c_allw_tauc[D_pos[2]+1:D_pos[3], :].sum(axis=0)

c_all_tau_q1 = c_allw_tau[D_pos[0], :]
c_all_tau_q2 = c_allw_tau[D_pos[0]+1:D_pos[1], :].sum(axis=0)
c_all_tau_q3 = c_allw_tau[D_pos[1]+1:D_pos[2], :].sum(axis=0)
c_all_tau_q4 = c_allw_tau[D_pos[2]+1:D_pos[3], :].sum(axis=0)

# Pool into percentiles
# c_all_bin_tauc = c_allw_tauc.reshape((5, 100, 300), order='F').sum(axis=0)
# n_all_bin_tauc = n_allw_tauc.reshape((5, 100, 300), order='F').sum(axis=0)
# c_all_bin_tau = c_allw_tau.reshape((5, 100, 300), order='F').sum(axis=0)
# n_all_bin_tau = n_allw_tau.reshape((5, 100, 300), order='F').sum(axis=0)

# Pool into quartiles PROBABLY WRONG
c_all_bin_tauc = c_allw_tauc.reshape((125, 4, 300), order='F').sum(axis=0)
n_all_bin_tauc = n_allw_tauc.reshape((125, 4, 300), order='F').sum(axis=0)
c_all_bin_tau = c_allw_tau.reshape((125, 4, 300), order='F').sum(axis=0)
n_all_bin_tau = n_allw_tau.reshape((125, 4, 300), order='F').sum(axis=0)     

# Plot results
plt.rcParams["figure.figsize"] = (18,7)
fig, ax = plt.subplots(1,2)
ax[0].set_title(r'Consumption response after consumption tax policy')
ax[0].plot(c_all_tauc_q1[0:22] * 100, label="Quartile 1")
ax[0].plot(c_all_tauc_q2[0:22] * 100, label="Quartile 2")
ax[0].plot(c_all_tauc_q3[0:22] * 100, label="Quartile 3")
ax[0].plot(c_all_tauc_q4[0:22] * 100, label="Quartile 4")
# ax[0].plot(c_all_bin_tauc[4, 0:22] * 100, label="Quintile 5")
ax[0].legend(loc='upper right', frameon=False)

ax[1].set_title(r'Consumption response after transfer policy')
ax[1].plot(c_all_tau_q1[0:22] * 100, label="Quartile 1")
ax[1].plot(c_all_tau_q2[0:22] * 100, label="Quartile 2")
ax[1].plot(c_all_tau_q3[0:22] * 100, label="Quartile 3")
ax[1].plot(c_all_tau_q4[0:22] * 100, label="Quartile 4")
ax[1].legend(loc='upper right', frameon=False)
plt.show()


fig, ax = plt.subplots(1,2)
ax[0].set_title(r'Consumption response after consumption tax policy')
ax[0].plot(c_all_bin_tauc[0, 0:22] * 100, label="Quartile 1")
ax[0].plot(c_all_bin_tauc[1, 0:22] * 100, label="Quartile 2")
ax[0].plot(c_all_bin_tauc[2, 0:22] * 100, label="Quartile 3")
ax[0].plot(c_all_bin_tauc[3, 0:22] * 100, label="Quartile 4")
# ax[0].plot(c_all_bin_tauc[4, 0:22] * 100, label="Quintile 4")
ax[0].legend(loc='upper right', frameon=False)

ax[1].set_title(r'Consumption response after transfer policy')
ax[1].plot(c_all_bin_tau[0, 0:22] * 100, label="Quartile 1")
ax[1].plot(c_all_bin_tau[1, 0:22] * 100, label="Quartile 2")
ax[1].plot(c_all_bin_tau[2, 0:22] * 100, label="Quartile 3")
ax[1].plot(c_all_bin_tau[3, 0:22] * 100, label="Quartile 4")
# ax[1].plot(c_all_bin_tau[4, 0:22] * 100, label="Quintile 4")
ax[1].legend(loc='upper right', frameon=False)
plt.show()



print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))
