"""One-asset HANK model with sticky wages: transfers vs interest rate cuts"""

# =============================================================================
# Code used by both models
# =============================================================================

print("MODEL WITH STICKY WAGES, TRANSFERS VS INTEREST RATE CUTS (WOLF 2022)")
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import het, simple, create_model             # functions
from sequence_jacobian import interpolate, grids, misc, estimation  # modules

def household_guess(a_grid, r, z_grid, gamma, T, tauc):
    new_z = np.ones((z_grid.shape[0],1))
    wel = (1 + r) * a_grid[np.newaxis,:] + new_z + T[:,np.newaxis]
    V_prime = (1 + r) / (1 + tauc) * (wel * 0.1) ** (-gamma)
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

def transfers(pi_e, Div, Tau, e_grid):
    tau_rule, div_rule = np.ones(e_grid.size), e_grid #np.ones(e_grid.size)
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tau =  Tau / np.sum(pi_e * tau_rule) * tau_rule 
    T = div + tau
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
def fiscal(r, Tau, B, C, N, tauc, taun, w):
    govt_res = Tau + (1 + r) * B(-1) - tauc * C - taun * w * N - B
    Deficit = Tau - tauc * C - taun * w * N # primary deficit
    Trans = Tau
    return govt_res, Deficit, Trans

@simple
def fiscal2(B, N, r, rhot, Tau, tauc, taun, w):
    # Tau = taun * w * N + B - (1 + r) * B(-1) # immediate adjustment of transfers, no tauc
    B = (1 + r) * B(-1) 
    govt_res = Tau - rhot * Tau(-1) - (1 - rhot) * (taun * w * N + B - (1 + r) * B(-1)) # delayed adjustment of transfers
    Deficit = Tau - taun * w * N # primary deficit, no tauc
    Trans = Tau
    return Deficit, Trans, govt_res

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

def iterate_h(foo, V_prime_start, Pi, a_grid, w, N, taun, pi_e, e_grid, r, 
                      Div, Tau, beta, gamma, tauc, maxit=1000, tol=1E-8):
    ite = 0
    err = 1
    V_prime_p = Pi @ V_prime_start # Pi is the markov chain transition matrix
    V_prime_old = V_prime_start # Initialize, V_prime_start will be set to ss V_prime_p
    T = transfers(pi_e, Div, Tau, e_grid)
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
# First model: transfer policy
# =============================================================================

print("MODEL 1: TRANSFER POLICY")

# Steady state
blocks_ss_tau = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc_ss, union_ss]
hank_ss_tau = create_model(blocks_ss_tau, name="One-Asset HANK SS")
calib_tau = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0,
               'kappaw': 0.003, 'muw': 1.2, 'N': 1.0, 'tauc': 0.0, 'taun': 0.036}
unknowns_ss_tau = {'beta': 0.986, 'Tau': -0.03}
targets_ss_tau = {'asset_mkt': 0, 'govt_res': 0}
print("Computing steady state...")
ss0_tau = hank_ss_tau.solve_steady_state(calib_tau, unknowns_ss_tau, targets_ss_tau, backward_tol=1E-22, solver="hybr")
print("Done")

# Dynamic model
blocks_tau = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc, wage,union]
hank_tau = create_model(blocks_tau, name="One-Asset HANK")
ss_tau = hank_tau.steady_state(ss0_tau)
T = 300
exogenous_tau = ['rstar','Tau', 'Z', 'tauc']
unknowns_tau = ['pi', 'w', 'Y', 'B']
targets_tau = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']
print("Computing Jacobian...")
G_tau = hank_tau.solve_jacobian(ss_tau, unknowns_tau, targets_tau, exogenous_tau, T=T)
print("Done")


# =============================================================================
# Second model: Interest rate policy 
# =============================================================================

print("MODEL 2: INTEREST RATE POLICY")

# Steady state 
blocks_ss_rstar = [hh_inp, firm, monetary, fiscal2, mkt_clearing, nkpc_ss, union_ss]
hank_ss_rstar = create_model(blocks_ss_rstar, name = "One-Asset HANK SS")
calib_rstar = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 1.5, 'B': 6.0,
               'kappaw': 0.003, 'muw': 1.2, 'N': 1.0, 'tauc': 0.0, 'taun': 0.036, 'rhot': 0.95}
unknowns_ss_rstar = {'beta': 0.986, 'Tau': -0.03}
targets_ss_rstar = {'asset_mkt': 0, 'govt_res': 0}
# unknowns_ss_rstar = {'beta': 0.986}
# targets_ss_rstar = {'asset_mkt': 0}
print("Computing steady state...")
ss0_rstar = hank_ss_rstar.solve_steady_state(calib_rstar, unknowns_ss_rstar, targets_ss_rstar, backward_tol=1E-22, solver="hybr")
print("Done")

# Dynamic model
blocks_rstar = [hh_inp, firm, monetary, fiscal2, mkt_clearing, nkpc,wage,union]
hank_rstar = create_model(blocks_rstar, name = "One-Asset HANK")
ss_rstar = hank_rstar.steady_state(ss0_rstar)
T = 300
exogenous_rstar = ['rstar', 'Z']
unknowns_rstar = ['pi', 'w', 'Y', 'Tau']
targets_rstar = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']
# unknowns_rstar = ['pi', 'w', 'Y']
# targets_rstar = ['nkpc_res', 'asset_mkt', 'wnkpc']
print("Computing Jacobian...")
G_rstar = hank_rstar.solve_jacobian(ss_rstar, unknowns_rstar, targets_rstar, exogenous_rstar, T=T)
print("Done")


# =============================================================================
# Impulse response functions
# =============================================================================

rhos = 0.9
dtau = 0.01 * rhos ** np.arange(T)
drstar = -0.0025 * rhos ** np.arange(T)

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

plt.rcParams["figure.figsize"] = (16,7)
fig, ax = plt.subplots(2, 4)
iT = 30
# fig.suptitle('Consumption tax cut versus transfer increase, sticky wages', size=16)

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

ax[1, 0].set_title(r'Wage $w$')
ax[1, 0].plot(dw[0][:iT])
ax[1, 0].plot(dw[1][:iT],'-.')

ax[1, 1].set_title(r'Inflation $\pi$')
ax[1, 1].plot(dp[0][:iT])
ax[1, 1].plot(dp[1][:iT],'-.')

ax[1, 2].set_title(r'Real interest rate $r$')
ax[1, 2].plot(dr[0][:iT])
ax[1, 2].plot(dr[1][:iT],'-.')

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

# Model 1: Parameters
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
path_w_tau = w_ss_tau + G_tau['w']['Tau'] @ dtau
path_r_tau = r_ss_tau + G_tau['r']['Tau'] @ dtau
path_div_tau = Div_ss_tau + G_tau['Div']['Tau'] @ dtau
path_n_tau = N_ss_tau + G_tau['N']['Tau'] @ dtau
path_tau_tau = Tau_ss_tau + dtau

# Initialize individual consumption paths
V_prime_p_tau = (1 + r_ss_tau) / (1 + tauc) * c_ss_tau ** (-gamma)
c_all_tau, n_all_tau = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))

# Compute all individual consumption paths
print("MODEL 1: Computing individual paths...")
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, n = iterate_h(household_d, V_prime_p_tau, Pi_tau, a_grid_tau, path_w_tau[t], path_n_tau[t], taun, pi_e_tau, 
                                           e_grid_tau, path_r_tau[t], path_div_tau[t], path_tau_tau[t], beta, gamma, tauc)
    c_all_tau[:, :, t] = c  
    n_all_tau[:, :, t] = n
print("Done")


# Model 2: Parameters
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
path_w_rstar = w_ss_rstar + G_rstar['w']['rstar'] @ drstar
path_r_rstar = r_ss_rstar + G_rstar['r']['rstar'] @ drstar
path_div_rstar = Div_ss_rstar + G_rstar['Div']['rstar'] @ drstar
path_n_rstar = N_ss_rstar + G_rstar['N']['rstar'] @ drstar
path_tau_rstar = Tau_ss_rstar + G_rstar['Tau']['rstar'] @ drstar

# Initialize individual consumption paths
V_prime_p_rstar = (1 + r_ss_rstar) / (1 + tauc) * c_ss_rstar ** (-gamma)
c_all_rstar, n_all_rstar = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))

# Compute all individual consumption paths
print("MODEL 2: Computing individual paths...")
for t in range(T-1, -1, -1):
    V_prime_p_rstar, _, c, n = iterate_h(household_d, V_prime_p_rstar, Pi_rstar, a_grid_rstar, path_w_rstar[t], path_n_rstar[t], taun, pi_e_rstar, 
                                         e_grid_rstar, path_r_rstar[t], path_div_rstar[t], path_tau_rstar[t], beta, gamma, tauc)
    c_all_rstar[:, :, t] = c  
    n_all_rstar[:, :, t] = n
print("Done")

# Select first period only and express as deviation from steady state
c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau
c_first_dev_rstar = (c_all_rstar[:, :, 0] - c_ss_rstar) / c_ss_rstar

# Weigh response by mass of agents
c_first_tau, c_first_rstar = np.zeros(nA), np.zeros(nA)
for i in range(nA):
    c_first_tau[i] = c_first_dev_tau[:, i] @ D_ss_tau[:, i]
    c_first_rstar[i] = c_first_dev_rstar[:, i] @ D_ss_rstar[:, i]
       
# Pool into percentile bins
c_first_bin_tau = c_first_tau.reshape(-1, 100, order='F').sum(axis=0)
c_first_bin_rstar = c_first_rstar.reshape(-1, 100, order='F').sum(axis=0)

# Plot results
plt.title(r'Consumption $c$')
plt.plot(c_first_bin_tau * 100, label="Transfer policy")
plt.plot(c_first_bin_rstar * 100,'-.', label="Interest rate policy")
plt.legend(loc='upper right', frameon=False)
plt.xlabel("Wealth percentile"), plt.ylabel("Percent deviation from steady state")
plt.show()




# =============================================================================
# Steady-state and dynamic properties
# =============================================================================

# # Difference consumption tax vs transfers
# dif = [['DIFFERENCE \u03C4c vs \u03C4','IMPACT RATIO','CUMULATIVE SUM'],
#       ['Shocks',np.ndarray.item(- (dtauc[:1] * ss0['tauc']) / (dT[2][:1, :] * ss0['Trans'])), - np.sum(dtauc) - np.sum(dT[2][:300])],
#       ['Output',np.ndarray.item(dY[1][:1, :] / dY[2][:1, :]), np.sum(dY[1][:300, :]) - np.sum(dY[2][:300, :])],
#       ['Consumption',np.ndarray.item(dC[1][:1, :] / dC[2][:1, :]), np.sum(dC[1][:300, :]) - np.sum(dC[2][:300, :])],
#       ['Inflation',np.ndarray.item(dp[1][:1, :] / dp[2][:1, :]), np.sum(dp[1][:300, :]) - np.sum(dp[2][:300, :])],
#       ['Wage',np.ndarray.item(dw[1][:1, :] / dw[2][:1, :]), np.sum(dw[1][:300, :]) - np.sum(dw[2][:300, :])],
#       ['Deficit',np.ndarray.item(dD[1][:1, :] / dD[2][:1, :]), np.sum(dD[1][:300, :]) - np.sum(dD[2][:300, :])]]
# dash = '-' * 50
# for i in range(len(dif)):
#     if i == 0:
#         print(dash)
#         print('{:<20s} {:^12s}  {:>15s}'.format(dif[i][0],dif[i][1],dif[i][2]))
#         print(dash)
#     else:
#         print('{:<20s} {:^12.3f}  {:>15.3f}'.format(dif[i][0],dif[i][1],dif[i][2]))
        
# # Show steady state
# ss_param = [['Discount factor', ss0['beta'], 'Intertemporal elasticity', ss0['gamma']],
#         ['Labor supply elasticity', 1 / ss0['nu'], 'Labor supply disutility', ss0['phi']],  
#         ['Goods substitutability', ss0['mu'] / (ss0['mu'] - 1) , 'Price markup', ss0['mu']],
#         ['Labor substitutability', ss0['muw'] / (ss0['muw'] - 1) , 'Wage markup', ss0['muw']],
#         ['Price Phillips slope', ss0['kappa'], 'Taylor rule inflation ', ss0['phi_pi']],
#         ['Wage Phillips slope', ss0['kappaw'], 'Taylor rule output ', 0],
#         ['Consumption tax rate', ss0['tauc'], 'Labor tax rate', ss0['taun']]]

# ss_var = [['Output', ss0['Y'], 'Government debt', ss0['A']],
#         ['Consumption', ss0['C'], 'Transfers', ss0['Tau']],
#         ['Hours', ss0['N'], 'Dividends', ss0['Div']], 
#         ['Wage', ss0['w'], 'Marginal cost', ss0['w'] / ss0['Z']],
#         ['Inflation', ss0['pi'], 'Consumption tax revenue', ss0['tauc'] * ss0['C']],
#         ['Nominal interest rate', ss0['r']*(1+ss0['pi']), 'Labor tax revenue', ss0['taun']*ss0['N']*ss0['w']],
#         ['Real interest rate', ss0['r'], 'Debt servicing  cost', ss0['r'] * ss0['A']]]
# ss_mkt = [['Bond market', ss0['asset_mkt'], 'Goods market (resid)', ss0['goods_mkt']],
#           ['Government budget', ss0['govt_res'], '', float('nan')]]

# dash = '-' * 73
# print(dash)
# print('PARAMETERS')
# for i in range(len(ss_param)):
#       print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_param[i][0],ss_param[i][1],ss_param[i][2],ss_param[i][3]))
# print('\nVARIABLES')
# for i in range(len(ss_var)):
#       print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_var[i][0],ss_var[i][1],ss_var[i][2],ss_var[i][3]))
# print('\nMARKET CLEARING')
# for i in range(len(ss_mkt)):
#       print('{:<24s}{:>12.0e}   {:24s}{:>10.0e}'.format(ss_mkt[i][0],ss_mkt[i][1],ss_mkt[i][2],ss_mkt[i][3]))
# print(dash)

print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))
