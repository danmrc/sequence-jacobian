"""One-asset HANK model with sticky wages: transfers vs interest rate cuts"""

# =============================================================================
# Code used by both models
# =============================================================================

print("STICKY WAGES, TRANSFERS VS INTEREST RATE CUTS (WOLF 2022)")
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import het, simple, create_model    # functions
from sequence_jacobian import interpolate, grids           # modules

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
    # tau_rule, div_rule = np.array((0, 0, 0, 0, 0, 0, 0, 1)), np.array((0, 0, 0, 0, 0, 0, 0, 1)) # all for the rich
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
    Deficit = Tau - taun * w * N # primary deficit, no tauc
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


# =============================================================================
# Household iteration policy rule
# =============================================================================

def household_d(V_prime_p, a_grid, e_grid, z_grid, beta, gamma, r, T, tauc):   
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


# =============================================================================
# First model: transfer policy
# =============================================================================

print("\nMODEL 1: TRANSFER POLICY")

# Steady state
blocks_ss_tau = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc_ss, union_ss]
hank_ss_tau = create_model(blocks_ss_tau, name="One-Asset HANK SS")
calib_tau = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 8,
               'amin': 0, 'amax': 150, 'nA': 50, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0,
               'kappaw': 0.006, 'muw': 1.2, 'N': 1.0, 'tauc': 0.0, 'taun': 0.036}
unknowns_ss_tau = {'beta': 0.986, 'Tau': -0.03}
targets_ss_tau = {'asset_mkt': 0, 'govt_res': 0}
print("Computing steady state...", end=" ")
ss0_tau = hank_ss_tau.solve_steady_state(calib_tau, unknowns_ss_tau, targets_ss_tau, backward_tol=1E-22, solver="hybr")
print("Done")

# Dynamic model
blocks_tau = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc, wage, union]
hank_tau = create_model(blocks_tau, name="One-Asset HANK")
ss_tau = hank_tau.steady_state(ss0_tau)
T = 300
exogenous_tau = ['rstar','Tau', 'Z', 'tauc']
unknowns_tau = ['pi', 'w', 'Y', 'B']
targets_tau = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']
print("Computing Jacobian...", end=" ")
G_tau = hank_tau.solve_jacobian(ss_tau, unknowns_tau, targets_tau, exogenous_tau, T=T)
print("Done")


# =============================================================================
# Second model: Interest rate policy 
# =============================================================================

print("MODEL 2: INTEREST RATE POLICY")

# Steady state 
blocks_ss_rstar = [hh_inp, firm, monetary2, fiscal2, mkt_clearing, nkpc_ss, union_ss]
hank_ss_rstar = create_model(blocks_ss_rstar, name = "One-Asset HANK SS")
calib_rstar = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 8,
               'amin': 0, 'amax': 150, 'nA': 50, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 1.5, 'B': 6.0,
               'kappaw': 0.006, 'muw': 1.2, 'N': 1.0, 'tauc': 0.0, 'taun': 0.036, 'rhot': 2.0}
# unknowns_ss_rstar = {'beta': 0.986, 'Tau': -0.03}
# targets_ss_rstar = {'asset_mkt': 0, 'govt_res': 0}
unknowns_ss_rstar = {'beta': 0.986, 'Tau': -0.03, 'B_ss': 6.0, 'r_ss': 0.005}
targets_ss_rstar = {'asset_mkt': 0, 'govt_res': 0, 'fiscal_rule': 0, 'r_resid': 0}
print("Computing steady state...", end=" ")
ss0_rstar = hank_ss_rstar.solve_steady_state(calib_rstar, unknowns_ss_rstar, targets_ss_rstar, backward_tol=1E-22, solver="hybr")
print("Done")

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
print("Computing Jacobian...", end=" ")
G_rstar = hank_rstar.solve_jacobian(ss_rstar, unknowns_rstar, targets_rstar, exogenous_rstar, T=T)
print("Done")


# =============================================================================
# Steady-state properties
# =============================================================================
    
ss_param = [['Discount factor', ss_tau['beta'], 'Intertemporal elasticity', ss_tau['gamma']],
            ['Labor supply elasticity', 1 / ss_tau['nu'], 'Labor supply disutility', ss_tau['phi']],  
            ['Goods substitutability', ss_tau['mu'] / (ss_tau['mu'] - 1) , 'Price markup', ss_tau['mu']],
            ['Labor substitutability', ss_tau['muw'] / (ss_tau['muw'] - 1) , 'Wage markup', ss_tau['muw']],
            ['Price Phillips slope', ss_tau['kappa'], 'Taylor rule inflation ', ss_tau['phi_pi']],
            ['Wage Phillips slope', ss_tau['kappaw'], 'Taylor rule output ', 0],
            ['Consumption tax rate', ss_tau['tauc'], 'Labor tax rate', ss_tau['taun']]]

ss_var_tau = [['Output', ss_tau['Y'], 'Government debt', ss_tau['A']],
              ['Consumption', ss_tau['C'], 'Transfers', ss_tau['Tau']],
              ['Hours', ss_tau['N'], 'Dividends', ss_tau['Div']], 
              ['Wage', ss_tau['w'], 'Marginal cost', ss_tau['w'] / ss_tau['Z']],
              ['Inflation', ss_tau['pi'], 'Consumption tax revenue', ss_tau['tauc'] * ss_tau['C']],
              ['Nominal interest rate', ss_tau['r']*(1+ss_tau['pi']), 'Labor tax revenue', ss_tau['taun']*ss_tau['N']*ss_tau['w']],
              ['Real interest rate', ss_tau['r'], 'Debt servicing  cost', ss_tau['r'] * ss_tau['A']]]

ss_var_rstar = [['Output', ss_rstar['Y'], 'Government debt', ss_rstar['A']],
                ['Consumption', ss_rstar['C'], 'Transfers', ss_rstar['Tau']],
                ['Hours', ss_rstar['N'], 'Dividends', ss_rstar['Div']], 
                ['Wage', ss_rstar['w'], 'Marginal cost', ss_rstar['w'] / ss_rstar['Z']],
                ['Inflation', ss_rstar['pi'], 'Consumption tax revenue', ss_rstar['tauc'] * ss_rstar['C']],
                ['Nominal interest rate', ss_rstar['r']*(1+ss_rstar['pi']), 'Labor tax revenue', ss_rstar['taun']*ss_rstar['N']*ss_rstar['w']],
                ['Real interest rate', ss_rstar['r'], 'Debt servicing  cost', ss_rstar['r'] * ss_rstar['A']]]

ss_mkt = [['Bond market', ss_tau['asset_mkt'], 'Goods market (resid)', ss_tau['goods_mkt']],
          ['Government budget', ss_tau['govt_res'], '', float('nan')]]

# Show steady state
dash = '-' * 73
# print(dash)
print('\nPARAMETERS')
for i in range(len(ss_param)):
    print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_param[i][0],ss_param[i][1],ss_param[i][2],ss_param[i][3]))
print('\nMODEL 1 STEADY STATE')
for i in range(len(ss_var_tau)):
    print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_var_tau[i][0],ss_var_tau[i][1],ss_var_tau[i][2],ss_var_tau[i][3]))
print('\nMODEL 2 STEADY STATE')
for i in range(len(ss_var_rstar)):
    print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_var_rstar[i][0],ss_var_rstar[i][1],ss_var_rstar[i][2],ss_var_rstar[i][3]))
      # print('{:<24s}{:>12.0e}   {:24s}{:>10.0e}'.format(ss_mkt[i][0],ss_mkt[i][1],ss_mkt[i][2],ss_mkt[i][3]))


# =============================================================================
# Impulse response functions
# =============================================================================

# Standard shock
rhos = 0.78
# dtau = 0.01 * rhos ** np.arange(T)
drstar = -0.002 * rhos ** np.arange(T)

# Zero net present value sock
discount = (1 / (1 + ss_tau['r']))
shock = np.zeros(T)
s1, s2, s3, s4, s5 = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
dtau = shock

dY = [G_tau['Y']['Tau'] @ dtau, G_rstar['Y']['rstar'] @ drstar]
dC = [G_tau['C']['Tau'] @ dtau, G_rstar['C']['rstar'] @ drstar]
dN = [G_tau['N']['Tau'] @ dtau, G_rstar['N']['rstar'] @ drstar]
dB = [G_tau['B']['Tau'] @ dtau, G_rstar['A']['rstar'] @ drstar]
dw = [G_tau['w']['Tau'] @ dtau, G_rstar['w']['rstar'] @ drstar]
dp = [G_tau['pi']['Tau'] @ dtau, G_rstar['pi']['rstar'] @ drstar]
dr = [G_tau['r']['Tau'] @ dtau, G_rstar['r']['rstar'] @ drstar]
dD = [G_tau['Deficit']['Tau'] @ dtau, G_rstar['Deficit']['rstar'] @ drstar]
dd = [G_tau['Div']['Tau'] @ dtau, G_rstar['Div']['rstar'] @ drstar]
dT = [G_tau['Trans']['Tau'] @ dtau, G_rstar['Trans']['rstar'] @ drstar]
di = [np.zeros(T), G_rstar['i']['rstar'] @ drstar]

plt.rcParams["figure.figsize"] = (20,7)
fig, ax = plt.subplots(2, 4)
fig.suptitle('Consumption tax cut versus transfer increase, sticky wages', size=16)
iT = 30

ax[0, 0].set_title(r'Output $Y$')
ax[0, 0].plot(dY[0][:iT], label="Transfer policy")
ax[0, 0].plot(dY[1][:iT],'-.', label="Monetary policy")
ax[0, 0].legend(loc='upper right', frameon=False)

ax[0, 1].set_title(r'Consumption $C$')
ax[0, 1].plot(dC[0][:iT])
ax[0, 1].plot(dC[1][:iT],'-.')

ax[0, 2].set_title(r'Government debt $B$')
ax[0, 2].plot(dB[0][:iT])
ax[0, 2].plot(dB[1][:iT],'-.')

ax[0, 3].set_title(r'Transfer $\tau$')
ax[0, 3].plot(dT[0][:iT])
ax[0, 3].plot(dT[1][:iT],'-.')
ax[0, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 0].set_title(r'Wage $w$')
ax[1, 0].plot(dw[0][:iT])
ax[1, 0].plot(dw[1][:iT],'-.')

ax[1, 1].set_title(r'Inflation $\pi$')
ax[1, 1].plot(dp[0][:iT])
ax[1, 1].plot(dp[1][:iT],'-.')

ax[1, 2].set_title(r'Dividends $d$')
ax[1, 2].plot(dd[0][:iT])
ax[1, 2].plot(dd[1][:iT],'-.')

# ax[1, 2].set_title(r'Government budget deficit')
# ax[1, 2].plot(-dD[0][:50])
# ax[1, 2].plot(-dD[1][:50],'-.')

ax[1, 3].set_title(r'Nominal interest rate $i$')
ax[1, 3].plot(di[0][:iT])
ax[1, 3].plot(di[1][:iT],'-.')
plt.show()


# =============================================================================
# Impact response by wealth percentile
# =============================================================================

# Model 1: Transfer policy
print("\nMODEL 1: TRANSFER POLICY")
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
D_ss_tau = ss_tau.internals['household']['Dbeg']

Div_ss_tau = ss_tau['Div']
N_ss_tau = ss_tau['N']
r_ss_tau = ss_tau['r']
Tau_ss_tau = ss_tau['Tau']
w_ss_tau = ss_tau['w']

# Aggregate transition dynamics
path_div_tau = Div_ss_tau + G_tau['Div']['Tau'] @ dtau
path_n_tau = N_ss_tau + G_tau['N']['Tau'] @ dtau
path_r_tau = r_ss_tau + G_tau['r']['Tau'] @ dtau
path_tau_tau = Tau_ss_tau + dtau
path_w_tau = w_ss_tau + G_tau['w']['Tau'] @ dtau

# Compute all individual consumption paths
print("Computing individual paths...", end=" ")
V_prime_p_tau = (1 + r_ss_tau) / (1 + tauc) * c_ss_tau ** (-gamma)
c_all_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                       path_div_tau[t], path_n_tau[t], path_r_tau[t], path_tau_tau[t], tauc, taun, path_w_tau[t])
    c_all_tau[:, :, t] = c  
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
V_prime_p_tau = (1 + r_ss_tau) / (1 + tauc) * c_ss_tau ** (-gamma)
c_direct_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                       Div_ss_tau, N_ss_tau, r_ss_tau, path_tau_tau[t], tauc, taun, w_ss_tau)
    c_direct_tau[:, :, t] = c
print("Done")


# Model 2: Interest rate policy
print("MODEL 2: INTEREST RATE POLICY")
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
D_ss_rstar = ss_rstar.internals['household']['Dbeg']

Div_ss_rstar = ss_rstar['Div']
N_ss_rstar = ss_rstar['N']
r_ss_rstar = ss_rstar['r']
Tau_ss_rstar = ss_rstar['Tau']
w_ss_rstar = ss_rstar['w']

# Aggregate transition dynamics
path_div_rstar = Div_ss_rstar + G_rstar['Div']['rstar'] @ drstar
path_n_rstar = N_ss_rstar + G_rstar['N']['rstar'] @ drstar
path_r_rstar = r_ss_rstar + G_rstar['r']['rstar'] @ drstar
path_tau_rstar = Tau_ss_rstar + G_rstar['Tau']['rstar'] @ drstar
path_w_rstar = w_ss_rstar + G_rstar['w']['rstar'] @ drstar

# Compute all individual consumption paths
print("Computing individual paths...", end=" ")
V_prime_p_rstar = (1 + r_ss_rstar) / (1 + tauc) * c_ss_rstar ** (-gamma)
c_all_rstar = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_rstar, _, c, _ = iterate_h(household_d, V_prime_p_rstar, a_grid_rstar, e_grid_rstar, Pi_rstar, pi_e_rstar, beta, gamma,
                                         path_div_rstar[t], path_n_rstar[t], path_r_rstar[t], path_tau_rstar[t], tauc, taun, path_w_rstar[t])
    c_all_rstar[:, :, t] = c
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
V_prime_p_rstar = (1 + r_ss_rstar) / (1 + tauc) * c_ss_rstar ** (-gamma)
c_direct_rstar = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_rstar, _, c, _ = iterate_h(household_d, V_prime_p_rstar, a_grid_rstar, e_grid_rstar, Pi_rstar, pi_e_rstar, beta, gamma,
                                         Div_ss_rstar, N_ss_rstar, path_r_rstar[t], path_tau_rstar[t], tauc, taun, w_ss_rstar)
    c_direct_rstar[:, :, t] = c
print("Done")

# Select first period only and express as deviation from steady state
c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau
c_first_dev_tau_direct = (c_direct_tau[:, :, 0] - c_ss_tau) / c_ss_tau
c_first_dev_rstar = (c_all_rstar[:, :, 0] - c_ss_rstar) / c_ss_rstar
c_first_dev_rstar_direct = (c_direct_rstar[:, :, 0] - c_ss_rstar) / c_ss_rstar

# Weigh response by mass of agents
c_first_tau, c_first_tau_direct, c_first_rstar, c_first_rstar_direct = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
for i in range(nA):
    # c_first_tau[i] = c_first_dev_tau[:, i] @ D_ss_tau[:, i]
    # c_first_rstar[i] = c_first_dev_rstar[:, i] @ D_ss_rstar[:, i] 
    c_first_tau[i] = (c_first_dev_tau[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])
    c_first_tau_direct[i] = (c_first_dev_tau_direct[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])
    c_first_rstar[i] = (c_first_dev_rstar[:, i] @ D_ss_rstar[:, i]) / np.sum(D_ss_rstar[:,i])
    c_first_rstar_direct[i] = (c_first_dev_rstar_direct[:, i] @ D_ss_rstar[:, i]) / np.sum(D_ss_rstar[:,i])

# Compute indirect effects
c_first_tau_indirect = c_first_tau - c_first_tau_direct
c_first_rstar_indirect = c_first_rstar - c_first_rstar_direct

# # Pool into percentile bins
# c_first_bin_tau = c_first_tau.reshape(-1, 100, order='F').mean(axis=0)
# c_first_bin_tau_direct = c_first_tau_direct.reshape(-1, 100, order='F').mean(axis=0) 
# c_first_bin_tau_indirect = c_first_bin_tau - c_first_bin_tau_direct
# c_first_bin_rstar = c_first_rstar.reshape(-1, 100, order='F').mean(axis=0)  
# c_first_bin_rstar_direct = c_first_rstar_direct.reshape(-1, 100, order='F').mean(axis=0) 
# c_first_bin_rstar_indirect = c_first_bin_rstar - c_first_bin_rstar_direct

# Smoothing function
import scipy as sp
def kernel_smoothing(vec, bandwidth):
    n = np.size(vec)
    result = np.zeros(n)
    for i in range(n):
        kernel = sp.stats.norm(vec[i], bandwidth)
        weights = kernel.pdf(vec)
        weights = weights/np.sum(weights)
        result[i] = weights @ vec
    return result

# # Smooth curves
# bandwith = 0.00
# c_first_bin_tau = kernel_smoothing(c_first_bin_tau, bandwith)
# c_first_bin_tau_direct = kernel_smoothing(c_first_bin_tau, bandwith)
# c_first_bin_tau_indirect = kernel_smoothing(c_first_bin_tau, bandwith)
# c_first_bin_rstar = kernel_smoothing(c_first_bin_tau, bandwith)
# c_first_bin_rstar_direct = kernel_smoothing(c_first_bin_tau, bandwith)
# c_first_bin_rstar_indirect = kernel_smoothing(c_first_bin_tau, bandwith)

# X-axis
D_ss_quant = 100 * np.cumsum(np.sum(D_ss_tau, axis=0))

# First percentile
D_ss_quant = np.append(0, D_ss_quant)
c_first_tau_direct = np.append(c_first_tau_direct[0], c_first_tau_direct)
c_first_tau_indirect =  np.append(c_first_tau_indirect[0], c_first_tau_indirect)
c_first_rstar_direct = np.append(c_first_rstar_direct[0], c_first_rstar_direct)
c_first_rstar_indirect =  np.append(c_first_rstar_indirect[0], c_first_rstar_indirect)
 
# Plot results
color_map = ["#FFFFFF", "#D95319"] # myb: "#0072BD"
fig, ax = plt.subplots(1,2)
ax[0].set_title(r'Interest rate policy')
ax[0].plot(D_ss_quant, 100 * c_first_rstar_direct, label="Direct effect", linewidth=3)  
ax[0].stackplot(D_ss_quant, 100 * c_first_rstar_direct, 100 * c_first_rstar_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)  
# ax[0].plot(c_first_bin_rstar_indirect, label="indirect effect") 
ax[0].legend(loc='upper left', frameon=False)
ax[0].set_xlabel("Wealth percentile"), ax[0].set_ylabel("Percent deviation from steady state")

ax[1].set_title(r'Transfer policy')
ax[1].plot(D_ss_quant, 100 * c_first_tau_direct, label="Direct effect", linewidth=3)    
ax[1].stackplot(D_ss_quant, 100 * c_first_tau_direct, 100 * c_first_tau_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)   
ax[1].legend(loc='upper right', frameon=False)
ax[1].set_xlabel("Wealth percentile")
plt.show()
     
# # plot results
# plt.title(r'impact response of consumption $c$ to transfer policy versus interest rate policy')
# plt.plot(c_first_tau * 100, label="transfer policy")
# plt.plot(c_first_rstar * 100,'-.', label="interest rate policy")
# # plt.legend(loc='upper right', frameon=false)
# plt.xlabel("wealth percentile"), plt.ylabel("percent deviation from steady state")
# plt.show()


# =============================================================================
# Individual vs aggregate impact responses
# =============================================================================

c_agg_tau_tot = dC[0][0] / ss_tau['C'] * 100
c_tau_tot = np.sum(c_all_tau[:, :, 0] * D_ss_tau) * 100
c_dev_tau_tot = np.sum((c_all_tau[:, :, 0] - c_ss_tau) * D_ss_tau) * 100 # absolute deviation from ss
c_dev_tau_tot = np.sum((c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau * D_ss_tau) * 100 # percent deviation from ss
print("Aggregate impact consumption response              = ", round(c_agg_tau_tot, 3), "%")
print("Sum of all individual impact consumption responses = ", round(c_dev_tau_tot, 3), "%")


# Plot
fig, ax = plt.subplots(2, 3)
# fig.suptitle('Individual vs Aaggregate responses', size=16)
iT = 30
ax[0, 0].set_title(r'Hours')
ax[0, 0].plot(N_ss_tau + dN[0][:iT], label="Aggregate")
ax[0, 0].plot(path_n_tau[:iT],'-.', label="Individual")
ax[0, 0].legend(loc='upper right', frameon=False)

ax[0, 1].set_title(r'Real interest rate')
ax[0, 1].plot(r_ss_tau + dr[0][:iT], label="Aggregate")
ax[0, 1].plot(path_r_tau[:iT],'-.', label="Individual")
ax[0, 1].legend(loc='upper right', frameon=False)

ax[0, 2].set_title(r'Wage')
ax[0, 2].plot(w_ss_tau + dw[0][:iT], label="Aggregate")
ax[0, 2].plot(path_w_tau[:iT],'-.', label="Individual")
ax[0, 2].legend(loc='upper right', frameon=False)

ax[1, 0].set_title(r'Transfers')
ax[1, 0].plot(Tau_ss_tau + dT[0][:iT], label="Aggregate")
ax[1, 0].plot(path_tau_tau[:iT],'-.', label="Individual")
ax[1, 0].legend(loc='upper right', frameon=False)

ax[1, 1].set_title(r'Dividends')
ax[1, 1].plot(Div_ss_tau + dd[0][:iT], label="Aggregate")
ax[1, 1].plot(path_div_tau[:iT],'-.', label="Individual")
ax[1, 1].legend(loc='upper right', frameon=False)
plt.show()





print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))
