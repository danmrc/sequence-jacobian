"""Distributional responses to shocks in HANK"""

# =============================================================================
# Initialize
# =============================================================================

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

def transfers(pi_e, Div, Transfer, e_grid):
    tax_rule, div_rule = np.ones(e_grid.size), e_grid #np.ones(e_grid.size)
    div = Div / np.sum(pi_e * div_rule) * div_rule
    transfer =  Transfer / np.sum(pi_e * tax_rule) * tax_rule 
    T = div + transfer
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


# =============================================================================
# Steady state
# =============================================================================

blocks_ss = [hh_ext, firm, monetary,fiscal, nkpc_ss, mkt_clearing]
hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.92, 'nE': 8,
               'amin': 0, 'amax': 200, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0, 
               'tauc': 0.1, 'taun': 0.036}

unknowns_ss = {'beta': 0.986, 'phi': 0.8, 'Transfer': 0.05}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0, 'govt_res': 0}
print("Computing steady state...")
ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")
print("Steady state solved")


# =============================================================================
# Dynamic model and Jacobian
# =============================================================================

blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc]
hank = create_model(blocks, name="One-Asset HANK")
ss = hank.steady_state(ss0)

T = 300
exogenous = ['rstar', 'Transfer', 'Z', 'tauc']
unknowns = ['pi', 'w', 'Y', 'B']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt', 'govt_res']

print("Computing Jacobian...")
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)
print("Jacobian solved")


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
                      Transfer, beta, gamma, nu, phi, tauc, maxit=1000, tol=1E-8):
    V_prime_p = Pi@V_prime_start
    V_prime_old = V_prime_start    
    ite = 0
    err = 1
    T = transfers(pi_e, Div, Transfer, e_grid)
    
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

# Distributional steady-state variables
e_grid = ss.internals['household']['e_grid']
a_grid = ss.internals['household']['a_grid']
D = ss.internals['household']['Dbeg']
pi_e =  ss.internals['household']['pi_e']
Pi = ss.internals['household']['Pi']
a_ss = ss.internals['household']['a']
c_ss = ss.internals['household']['c']
n_ss = ss.internals['household']['n']

# Aggregate steady-state variables
N_ss = ss['N']
r_ss = ss['r']
Div_ss = ss['Div']
Transfer_ss = ss['Transfer']
w_ss = ss['w']
T_ss = transfers(pi_e, Div_ss, Transfer_ss, e_grid)

# Sort assets into bins
nbin = np.zeros(100)
nbin[1], nbin[2] = 0.5, 1
nbin[3:] = np.arange(1.5, 150, 148.5 / 97)
nbin = np.arange(0, 150, 1.5)
a_bin = np.digitize(a_ss, nbin)

# Zero net present value shock
shock = np.zeros(T)
discount = (1 / (1 + r_ss))
s1, s2, s3, s4, s5 = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 


# =============================================================================
# Consumption tax policy
# =============================================================================

print("FIRST POLICY")
dtauc = - shock 
    
# Aggregate transition dynamics
path_n = N_ss + G['N']['tauc'] @ dtauc
path_r = r_ss + G['r']['tauc'] @ dtauc
path_w = w_ss + G['w']['tauc'] @ dtauc
path_div = Div_ss + G['Div']['tauc'] @ dtauc
path_tauc = tauc + dtauc

# Initialize individual consumption paths
V_prime_p = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
all_c = all_n = np.zeros((nE, nA, T))

# Compute all individual consumption paths
print("Computing individual paths...")
for t in range(T-1, -1, -1):
    V_prime_p, _, c, n = iterate_household(household_d, V_prime_p, Pi, a_grid, path_w[t], taun, pi_e,
                            e_grid, path_r[t], path_div[t], Transfer_ss, beta, gamma, nu, phi, path_tauc[t])
    all_c[:, :, t] = c  
    # all_n[:, :, t] = n

# Select first period only and express as deviation from steady state
c_first = all_c[:, :, 0]
c_dev = (c_first - c_ss) / c_ss

# Weight each consumption response by mass of agents with given asset value
a_dist = D_dist = c_dist =  np.zeros((nE, nA, len(nbin)))
c_tauc = D_total = np.zeros(len(nbin))
for i in range(1, len(nbin)):  
    a_dist[:, :, i] = np.where(a_bin == i, 1, 0) # returns 1 if true, 0 otherwise
    D_dist[:, :, i] = np.multiply(D, a_dist[:, :, i])
    # D_total[i] = np.sum(D_dist[:, :, i])
    c_dist[:, :, i] = np.multiply(c_dev, a_dist[:, :, i])
    c_tauc[i] = np.sum(c_dist[:, :, i]) #* np.sum(D_dist[:, :, i] )
print("Individual paths solved")


plt.plot(c_tauc[1:], label = "Consumption tax policy")
plt.show()


# =============================================================================
# Transfer policy
# =============================================================================

print("SECOND POLICY")
dtau = shock 
    
# Aggregate transition dynamics
path_n = N_ss + G['N']['Transfer'] @ dtau
path_r = r_ss + G['r']['Transfer'] @ dtau
path_w = w_ss + G['w']['Transfer'] @ dtau
path_div = Div_ss + G['Div']['Transfer'] @ dtau
path_transfer = Transfer_ss + dtau

# Initialize individual consumption paths
V_prime_p = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
all_c = all_n = np.zeros((nE, nA, T))

# Compute all individual consumption paths
print("Computing individual paths...")
for t in range(T-1, -1, -1):
    V_prime_p, _, c, _ = iterate_household(household_d, V_prime_p, Pi, a_grid, path_w[t], taun, pi_e,
                            e_grid, path_r[t], path_div[t], path_transfer[t], beta, gamma, nu, phi, tauc)
    all_c[:,:,t] = c  

# Select first period only and express as deviation from steady state
c_first = all_c[:, :, 0]
c_dev = (c_first - c_ss) / c_ss

# Weight each consumption response by mass of agents with given asset value
a_dist = D_dist = c_dist =  np.zeros((nE, nA, len(nbin)))
c_tau = D_total = np.zeros(len(nbin))
for i in range(1, len(nbin)):  
    a_dist[:, :, i] = np.where(a_bin == i, 1, 0) # returns 1 if true, 0 otherwise
    D_dist[:, :, i] = np.multiply(D, a_dist[:, :, i])
    # D_total[i] = np.sum(D_dist[:, :, i])
    c_dist[:, :, i] = np.multiply(c_dev, a_dist[:, :, i])
    c_tau[i] = np.sum(c_dist[:, :, i]) #* np.sum(D_dist[:, :, i] )
print("Individual paths solved")


# =============================================================================
# Plot
# =============================================================================

plt.rcParams["figure.figsize"] = (16,7)
plt.plot(c_tauc[1:], label = "Consumption tax policy")
plt.plot(c_tau[1:], label = "Transfer policy")
plt.legend()
plt.show()

print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))
