"""Distributional responses to shocks in HANK"""

# =============================================================================
# Code used by both policies
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import het, simple, create_model             # functions
from sequence_jacobian import interpolate, grids, misc, estimation  # modules

def household_guess(a_grid, r, z_grid, gamma, T, tauc):
    new_z = np.ones((z_grid.shape[0],1))
    wel = (1 + r) * a_grid[np.newaxis,:] + new_z + T[:,np.newaxis]
    V_prime = (1 + r) * (wel * 0.1) ** (-gamma)
    return V_prime

@het(exogenous='Pi', policy ='a', backward='V_prime', backward_init=household_guess)
def household(V_prime_p, a_grid, z_grid, e_grid, r, T, beta, gamma, tauc):
    c_prime = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma) # c_prime is the new guess for c_t
    new_grid = (1 + tauc) * c_prime + a_grid[np.newaxis,:] - z_grid[:,np.newaxis] - T[:,np.newaxis]
    wel = (1 + r) * a_grid
    c = interpolate.interpolate_y(new_grid,wel,c_prime)
    a = wel + z_grid[:,np.newaxis] + T[:,np.newaxis] - (1 + tauc) * c
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

def income(e_grid, w, N, taun):
    z_grid = (1 - taun) * w * N * e_grid
    return z_grid

def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi, a_grid, pi_e

def transfers(pi_e, Div, Transfer, e_grid):
    tax_rule, div_rule = np.ones(e_grid.size), e_grid #np.ones(e_grid.size)
    div = Div / np.sum(pi_e * div_rule) * div_rule
    transfer =  (Transfer) / np.sum(pi_e * tax_rule) * tax_rule 
    T = div + transfer
    return T

hh_inp = household.add_hetinputs([make_grid,transfers,income])

@simple
def firm(Y, w, Z, pi, mu, kappa):
    N = Y / Z
    Div = Y - w * N - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    return Div, N

@simple
def monetary(pi, rstar, phi_pi):
    r = (1 + rstar(-1) + phi_pi * pi(-1)) / (1 + pi) - 1
    i = rstar
    return r, i

@simple
def fiscal(r, Transfer, B, C, N, tauc, taun, w):
    govt_res = Transfer + (1 + r) * B(-1) - tauc * C - taun * w * N - B
    Deficit = Transfer - tauc * C - taun * w * N # primary deficit
    Trans = Transfer
    return govt_res, Deficit, Trans

@simple
def mkt_clearing(A, C, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    goods_mkt = Y - C - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    return asset_mkt, goods_mkt

@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w

@simple 
def union_ss(w, N, UCE, kappaw, nu, muw, tauc, taun):
    phi = ((1 - taun) * w * N ** (-nu) * UCE) / ((1 + tauc) * muw)
    wnkpc = kappaw * (phi * N ** (1 + nu) - (1 - taun) * w * N * UCE / ((1 + tauc) * muw))
    return wnkpc, phi

@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw

@simple
def union(piw, w, N, UCE, kappaw, phi, nu, muw, beta, tauc, taun):
    wnkpc = (kappaw * (phi * N ** (1+nu) - (1 - taun) * w * N * UCE / ((1 + tauc) * muw)) 
             + beta * (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log))
    return wnkpc

@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


# =============================================================================
# Household iteration policy rule
# =============================================================================

def household_d(V_prime_p, a_grid, z_grid, e_grid, r, T, beta, gamma, tauc):
    c_prime = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma) # c_prime is the new guess for c_t
    new_grid = (1 + tauc) * c_prime + a_grid[np.newaxis,:] - z_grid[:,np.newaxis] - T[:,np.newaxis]
    wel = (1 + r) * a_grid
    c = interpolate.interpolate_y(new_grid,wel,c_prime)
    a = wel + z_grid[:,np.newaxis] + T[:,np.newaxis] - (1 + tauc) * c
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

def iterate_household(foo, V_prime_start, Pi, a_grid, w, N, taun, pi_e, e_grid, r, 
                      Div, Transfer, beta, gamma, tauc, maxit=1000, tol=1E-8):
    ite = 0
    err = 1
    V_prime_p = Pi @ V_prime_start # Pi is the markov chain transition matrix
    V_prime_old = V_prime_start # Initialize, V_prime_start will be set to ss V_prime_p
    T = transfers(pi_e, Div, Transfer, e_grid)
    z_grid = income(e_grid, w, N, taun)
    
    while ite < maxit and err > tol:
        V_prime_temp, a, c, uce = foo(V_prime_p, a_grid, z_grid, e_grid, r, T, beta, gamma, tauc) # foo is a placeholder
        V_prime_p = Pi @ V_prime_temp
        ite += 1
        err = np.max(np.abs(V_prime_old - V_prime_temp))
        V_prime_old = V_prime_temp
    # print("Iteration ", ite, " out of ", maxit, "\n Difference in policy (sup norm):", err)
    return V_prime_temp, a, c, uce 


# =============================================================================
# Transfer policy
# =============================================================================

print("First model")

# Steady state
blocks_ss = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc_ss, union_ss]
hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")
calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0,
               'kappaw': 0.05, 'muw': 1.2, 'N': 1.0, 'tauc': 0.1, 'taun': 0.036}
unknowns_ss = {'beta': 0.986, 'Transfer': -0.03}
targets_ss = {'asset_mkt': 0, 'govt_res': 0}
print("Computing steady state...")
ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, backward_tol=1E-22, solver="hybr")
print("Steady state done")

# Dynamic model
blocks = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc,wage,union]
hank = create_model(blocks, name="One-Asset HANK")
ss = hank.steady_state(ss0)
T = 300
exogenous = ['rstar','Transfer', 'Z', 'tauc']
unknowns = ['pi', 'w', 'Y', 'B']
targets = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']
print("Computing Jacobian...")
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)
print("Jacobian done")

# Shock, parameters, steady-state variables, and aggregate transition dynamics
shock = np.zeros(T)
discount = (1 / (1 + ss['r']))
A, B, C, D, E = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (A - B * (x - E)) * np.exp(-C * (x - E) - D)
    
z_grid = ss0.internals['household']['z_grid']
e_grid = ss0.internals['household']['e_grid']
a_grid = ss0.internals['household']['a_grid']
pi_e =  ss0.internals['household']['pi_e']
Pi = ss0.internals['household']['Pi']
D = ss.internals['household']['Dbeg']

nu = ss0['nu']
beta = ss0['beta']
gamma = ss0['gamma']
tauc = ss0['tauc']
taun = ss0['taun']
nE = ss0['nE']
nA = ss0['nA']

a_ss = ss0.internals['household']['a']
c_ss = ss0.internals['household']['c']
r_ss = ss0['r']
Transfer_ss = ss0['Transfer']
Div_ss = ss0['Div']
T_ss = transfers(pi_e,Div_ss,Transfer_ss,e_grid)
w_ss = ss0['w']
N_ss = ss0['N']

dtstar = shock

path_w = w_ss + G['w']['Transfer'] @ dtstar
path_r = r_ss + G['r']['Transfer'] @ dtstar
path_div = Div_ss + G['Div']['Transfer'] @ dtstar
path_n = N_ss + G['N']['Transfer'] @ dtstar
path_transfer = Transfer_ss + dtstar

# Initialize individual consumption paths
V_prime_p = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
all_c = np.zeros((nE, nA, T))

# Compute all individual consumption paths
for t in range(T-1, -1, -1):
    # print(t)
    V_prime_p, _, c, _ = iterate_household(household_d, V_prime_p, Pi, a_grid, path_w[t], path_n[t], taun, pi_e, 
                                           e_grid, path_r[t], path_div[t], path_transfer[t], beta, gamma, tauc)
    all_c[:, :, t] = c # absolute consumption
    
# Select only the first period
c_first = all_c[:, :, 0]
c_dev = (c_first - c_ss) / c_ss # percent deviation from steady state

# Sort assets into bins
nbin = np.zeros(100)
nbin[1], nbin[2] = 0.5, 1
nbin[3:] = np.arange(1.5, 150, 148.5 / 97)
nbin = np.arange(0, 150, 1.5)
a_bin = np.digitize(a_ss, nbin)

# Weight each consumption response by mass of agents with given asset value
a_dist = D_dist = c_dist =  np.zeros((nE, nA, len(nbin)))
c_pct = D_total = np.zeros(len(nbin))

for i in range(1, len(nbin)):  
    a_dist[:, :, i] = np.where(a_bin == i, 1, 0) # returns 1 if true, 0 otherwise
    D_dist[:, :, i] = np.multiply(D, a_dist[:, :, i])
    # D_total[i] = np.sum(D_dist[:, :, i])
    c_dist[:, :, i] = np.multiply(c_dev, a_dist[:, :, i])
    c_pct[i] = np.sum(c_dist[:, :, i]) #* np.sum(D_dist[:, :, i] )


# =============================================================================
# Interest rate policy 
# =============================================================================

print("Second model")

# Steady state 
blocks_ss = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc_ss, union_ss]
hank_ss = create_model(blocks_ss, name = "One-Asset HANK SS")
calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 1.5, 'B': 6.0,
               'kappaw': 0.05, 'muw': 1.2, 'N': 1.0, 'tauc': 0.1, 'taun': 0.036}
unknowns_ss = {'beta': 0.986, 'Transfer': -0.03}
targets_ss = {'asset_mkt': 0, 'govt_res': 0}
print("Computing steady state...")
ss0r = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, backward_tol = 1E-22, solver="hybr")
print("Steady state done")

# Dynamic model
blocks = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc,wage,union]
hank = create_model(blocks, name = "One-Asset HANK")
ss = hank.steady_state(ss0r)
T = 300
exogenous = ['rstar', 'Z']
unknowns = ['pi', 'w', 'Y', 'Transfer']
targets = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']
print("Computing Jacobian...")
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)
print("Jacobian done")

# Shock, parameters, steady-state variables, and aggregate transition dynamics
rhos = 0.9
drstar = -0.001 * rhos ** (np.arange(T)[:, np.newaxis])

z_grid = ss0r.internals['household']['z_grid']
e_grid = ss0r.internals['household']['e_grid']
a_grid = ss0r.internals['household']['a_grid']
pi_e =  ss0r.internals['household']['pi_e']
Pi = ss0r.internals['household']['Pi']

nu = ss0r['nu']
beta = ss0r['beta']
gamma = ss0r['gamma']
tauc = ss0r['tauc']
taun = ss0r['taun']
nE = ss0r['nE']
nA = ss0r['nA']

c_ss = ss0r.internals['household']['c']
r_ss = ss0r['r']
Transfer_ss = ss0r['Transfer']
Div_ss = ss0r['Div']
T_ss = transfers(pi_e,Div_ss,Transfer_ss,e_grid)
w_ss = ss0r['w']
N_ss = ss0r['N']

path_w = w_ss + G['w']['rstar'] @ drstar
path_r = r_ss + G['r']['rstar'] @ drstar
path_div = Div_ss + G['Div']['rstar'] @ drstar
path_n = N_ss + G['N']['rstar'] @ drstar
path_transfer = Transfer_ss + G['Transfer']['rstar'] @ drstar

# plt.plot(path_w)
# plt.plot(path_r)
# plt.plot(path_n)
# plt.plot(path_transfer)
# plt.show()

# Initialize individual consumption paths
V_prime_p = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
all_c = np.zeros((nE, nA, T))

# Compute all individual consumption paths
for t in range(T-1, -1, -1):
    V_prime_p, _, c, _ = iterate_household(household_d, V_prime_p, Pi, a_grid, path_w[t], path_n[t], taun, pi_e, 
                                           e_grid, path_r[t], path_div[t], path_transfer[t], beta, gamma, tauc)
    all_c[:, :, t] = c # absolute consumption
    
# Select only the first period
c_first = all_c[:, :, 0]
c_dev = (c_first - c_ss) / c_ss # percent deviation from steady state

# Sort assets into bins
a_bin = np.digitize(a_ss, nbin)

# Weight each consumption response by mass of agents with given asset value
a_dist = D_dist = c_dist =  np.zeros((nE, nA, len(nbin)))
c_pct_R = D_total = np.zeros(len(nbin))

for i in range(1, len(nbin)):  
    a_dist[:, :, i] = np.where(a_bin == i, 1, 0) # returns 1 if true, 0 otherwise
    D_dist[:, :, i] = np.multiply(D, a_dist[:, :, i])
    # D_total[i] = np.sum(D_dist[:, :, i])
    c_dist[:, :, i] = np.multiply(c_dev, a_dist[:, :, i])
    c_pct_R[i] = np.sum(c_dist[:, :, i]) #* np.sum(D_dist[:, :, i] )

plt.plot(c_pct[1:], label = "Transfer policy")
plt.plot(c_pct_R[1:], label = "Interest-rate policy")
plt.legend()
plt.show()



# wealth_perc = grids.agrid(amin = 0, amax=1, n=nA)

# plt.plot(wealth_perc,c_asset_dist_t, label = "Transfer shock")
# plt.plot(wealth_perc,c_asset_dist, label = "Interest-rate shock")
# plt.legend()
# plt.show()