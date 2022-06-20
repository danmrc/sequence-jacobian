"""One-Asset HANK model with exogenous transfers, taxes, and sticky wages"""

# =============================================================================
# Initialize
# =============================================================================

print("STICKY WAGES")
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import het, simple, create_model             # functions
from sequence_jacobian import interpolate, grids, misc, estimation  # modules


# =============================================================================
# Household heterogeneous block
# =============================================================================

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
    tax_rule, div_rule = np.ones(e_grid.size), e_grid #np.ones(e_grid.size)
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tau =  Tau / np.sum(pi_e * tax_rule) * tax_rule 
    T = div + tau
    return T

hh_inp = household.add_hetinputs([make_grid,transfers,income])


# =============================================================================
# Simple blocks
# =============================================================================

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
# Steady state
# =============================================================================

blocks_ss = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc_ss, union_ss]
hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0,
               'kappaw': 0.005, 'muw': 1.1, 'N': 1.0, 'tauc': 0.1, 'taun': 0.036}

unknowns_ss = {'beta': 0.986, 'Tau': -0.03}
targets_ss = {'asset_mkt': 0, 'govt_res': 0}

print("Computing steady state...")
ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")
print("Steady state solved")


# =============================================================================
# Dynamic model and Jacobian
# =============================================================================

blocks = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc,wage,union]
hank = create_model(blocks, name="One-Asset HANK")
ss = hank.steady_state(ss0)
    
T = 300
exogenous = ['rstar','Tau', 'Z', 'tauc']
unknowns = ['pi', 'w', 'Y', 'B']
targets = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']

print("Computing Jacobian...")
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)
print("Jacobian solved")


#==============================================================================
# Steady-state properties
# =============================================================================

# Parameters
nE = ss['nE']
nA = ss['nA']
nu = ss['nu']
phi = ss['phi']
tauc = ss['tauc']
taun = ss['taun']

# Steady-state variables
A_ss = ss['A']
C_ss = ss['C']
Div_ss = ss['Div']
N_ss = ss['N']
r_ss = ss['r']
Tau_ss = ss['Tau']
w_ss = ss['w']
Y_ss = ss['Y']

e_grid = ss.internals['household']['e_grid']
a_grid = ss.internals['household']['a_grid']
D_ss = ss.internals['household']['Dbeg']
pi_e =  ss.internals['household']['pi_e']
Pi = ss.internals['household']['Pi']
a_ss = ss.internals['household']['a']
c_ss = ss.internals['household']['c']
T_ss = transfers(pi_e, Div_ss, Tau_ss, e_grid)

# Share of hand-to-mouth
D_dist = np.sum(D_ss, axis=0)
htm = D_dist[0]

# Share of hand-to-mouth, other method
zero_asset = np.where(a_ss == 0)
htm = np.sum(D_ss[zero_asset])

# Population distribution
l_tot = np.sum(D_ss);
l_dist = np.sum(D_ss, axis=0)
l_bin = l_dist.reshape(-1, 100, order='F').sum(axis=0)

# Wealth distribution
a_tot = np.sum(np.multiply(a_ss, D_ss))
a_dist = np.sum(np.multiply(a_ss, D_ss), axis=0)
a_bin = a_dist.reshape(-1, 100, order='F').sum(axis=0)

# Income distribution
y_ss = ((1 - taun) * w_ss * np.multiply(N_ss, e_grid[:, None]) + r_ss * a_ss + T_ss[:, None]) / (1 + tauc)
y_dist = np.sum(np.multiply(y_ss, D_ss), axis=0)
y_tot = np.sum(y_dist)
y_bin = y_dist.reshape(-1, 100, order='F').sum(axis=0)

# Consumption distribution
c_tot = np.sum(np.multiply(c_ss, D_ss))
c_dist = np.sum(np.multiply(c_ss, D_ss), axis=0)
c_bin = c_dist.reshape(-1, 100, order='F').sum(axis=0)

# # Labor supply distribution
# n_tot = np.sum(np.multiply(n_ss, D_ss))
# n_dist = np.sum(np.multiply(n_ss, D_ss), axis=0)
# n_bin = n_dist.reshape(-1, 100, order='F').sum(axis=0)

# Dividend distribution
d_tot = np.sum(np.multiply(T_ss[:, None] - Tau_ss, D_ss))
d_dist = np.sum(np.multiply(T_ss[:, None] - Tau_ss, D_ss), axis=0)
d_bin = d_dist.reshape(-1, 100, order='F').sum(axis=0)

# Transfer distribution
tau_tot = np.sum(np.multiply(Tau_ss, D_ss))
tau_dist = np.sum(np.multiply(Tau_ss, D_ss), axis=0)
tau_bin = tau_dist.reshape(-1, 100, order='F').sum(axis=0)

# Wealth Lorenz curve
D_grid = np.append(np.zeros(1), np.cumsum(D_dist))
a_lorenz = np.append(np.zeros(1), np.cumsum(a_dist / a_tot))
a_lorenz_area = np.trapz(a_lorenz, x=D_grid) # area below Lorenz curve
a_gini = (0.5 - a_lorenz_area) / 0.5
print("Wealth Gini =", np.round(a_gini, 3))

# Income Lorenz curve
y_lorenz = np.append(np.zeros(1), np.cumsum(y_dist / y_tot))
y_lorenz_area = np.trapz(y_lorenz, x=D_grid) # area below Lorenz curve
y_gini = (0.5 - y_lorenz_area) / 0.5
print("Income Gini =", np.round(y_gini, 3))

# Plot distributions
plt.rcParams["figure.figsize"] = (16,7)
fig, ax = plt.subplots(2, 4)

ax[0, 0].set_title(r'Skill $e$ distribution')
ax[0, 0].plot(e_grid, pi_e)
ax[0, 0].fill_between(e_grid, pi_e)

ax[0, 1].set_title(r'Poplulation distribution')
ax[0, 1].plot(l_bin)
ax[0, 1].fill_between(range(100), l_bin)

ax[0, 2].set_title(r'Wealth $a$ distribution')
ax[0, 2].plot(a_bin / a_tot)
ax[0, 2].fill_between(range(100), a_bin / a_tot)

ax[0, 3].set_title(r'Income $y$ and consumption $c$ distribution')
ax[0, 3].plot(y_bin / y_tot)
ax[0, 3].fill_between(range(100), y_bin / y_tot)
ax[0, 3].plot(c_bin / c_tot)

# ax[0, 3].set_title(r'Labor supply $n$ distribution')
# ax[0, 3].plot(n_bin / n_tot)
# ax[0, 3].fill_between(range(100), n_bin / n_tot)

ax[1, 0].set_title(r'Dividend $d$ distribution')
ax[1, 0].plot(d_bin / d_tot)
ax[1, 0].fill_between(range(100), d_bin / d_tot)

ax[1, 1].set_title(r'Transfer $\tau$ distribution')
ax[1, 1].plot(tau_bin / tau_tot)
ax[1, 1].fill_between(range(100), tau_bin / tau_tot)

ax[1, 2].set_title(r'Wealth Lorenz curve')
ax[1, 2].plot(D_grid, a_lorenz)
ax[1, 2].plot([0, 1], [0, 1], '-')

ax[1, 3].set_title(r'Income Lorenz curve')
ax[1, 3].plot(D_grid, y_lorenz) 
ax[1, 3].plot([0, 1], [0, 1], '-')
plt.show()

# Show steady state
ss_param = [['Discount factor', ss['beta'], 'Intertemporal elasticity', ss['gamma']],
        ['Labor supply elasticity', 1 / ss['nu'], 'Labor supply disutility', ss['phi']],  
        ['Goods substitutability', ss['mu'] / (ss['mu'] - 1) , 'Price markup', ss['mu']],
        ['Phillips curve slope', ss['kappa'], 'Taylor rule inflation ', ss['phi_pi']],
        ['Consumption tax rate', ss['tauc'], 'Labor tax rate', ss['taun']]]

ss_var = [['Output', ss['Y'], 'Government debt', ss['A']],
        ['Consumption', ss['C'], 'Transfers', ss['Tau']],
        ['Hours', ss['N'], 'Dividends', ss['Div']], 
        ['Wage', ss['w'], 'Marginal cost', ss['w'] / ss['Z']],
        ['Inflation', ss['pi'], 'Consumption tax revenue', ss['tauc'] * ss['C']],
        ['Nominal interest rate', ss['r']*(1+ss['pi']), 'Labor tax revenue', ss['taun']*ss['N']*ss['w']],
        ['Real interest rate', ss['r'], 'Debt servicing  cost', ss['r'] * ss['A']]]

ss_mom = [['Share of hand-to-mouth', htm, 'Gini index', a_gini]]

ss_mkt = [['Bond market', ss['asset_mkt'], 'Labor market', ss['labor_mkt']],
          ['Goods market (resid)', ss['goods_mkt'], 'Government budget', ss['govt_res']]]

dash = '-' * 73
print(dash)
print('PARAMETERS')
for i in range(len(ss_param)):
      print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_param[i][0],ss_param[i][1],ss_param[i][2],ss_param[i][3]))
print('\nAGGREGATE VARIABLES IN STEADY STATE')
for i in range(len(ss_var)):
      print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_var[i][0],ss_var[i][1],ss_var[i][2],ss_var[i][3]))
print('\nDISTRIBUTIONAL VARIABLES IN STEADY STATE')
for i in range(len(ss_mom)):
      print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_mom[i][0],ss_mom[i][1],ss_mom[i][2],ss_mom[i][3]))
print('\nMARKET CLEARING')
for i in range(len(ss_mkt)):
      print('{:<24s}{:>12.0e}   {:24s}{:>10.0e}'.format(ss_mkt[i][0],ss_mkt[i][1],ss_mkt[i][2],ss_mkt[i][3]))
print(dash)


# =============================================================================
# Impulse response functions
# =============================================================================

rhos = 0.9
drstar = -0.02 * rhos ** (np.arange(T)[:, np.newaxis])
dtstar = 0.01 * rhos ** (np.arange(T)[:, np.newaxis])
dtauc = - 0.01 * rhos ** (np.arange(T)[:, np.newaxis])

dY = [G['Y']['rstar'] @ drstar, G['Y']['tauc'] @ dtauc, G['Y']['Tau'] @ dtstar]
dC = [G['C']['rstar'] @ drstar, G['C']['tauc'] @ dtauc, G['C']['Tau'] @ dtstar]
dN = [G['N']['rstar'] @ drstar, G['N']['tauc'] @ dtauc, G['N']['Tau'] @ dtstar]
dB = [G['A']['rstar'] @ drstar, G['A']['tauc'] @ dtauc, G['A']['Tau'] @ dtstar]
dw = [G['w']['rstar'] @ drstar, G['w']['tauc'] @ dtauc, G['w']['Tau'] @ dtstar]
dp = [G['pi']['rstar'] @ drstar, G['pi']['tauc'] @ dtauc, G['pi']['Tau'] @ dtstar]
dr = [G['r']['rstar'] @ drstar, G['r']['tauc'] @ dtauc, G['r']['Tau'] @ dtstar]
dD = [G['Deficit']['rstar'] @ drstar, G['Deficit']['tauc'] @ dtauc, G['Deficit']['Tau'] @ dtstar]
dd = [G['Div']['rstar'] @ drstar, G['Div']['tauc'] @ dtauc, G['Div']['Tau'] @ dtstar]
dT = [np.zeros(T), np.zeros(T), G['Trans']['Tau'] @ dtstar]
di = [G['i']['rstar'] @ drstar, np.zeros(T), np.zeros(T)]

plt.rcParams["figure.figsize"] = (16,7)
fig, ax = plt.subplots(2, 4)
fig.suptitle('Consumption tax cut versus transfer increase, sticky wages', size=16)

ax[0, 0].set_title(r'Output $Y$')
#l1, = ax[0, 0].plot(dY[0][:50, :])
ax[0, 0].plot(dY[1][:50, :] * ss0['Y'], label="Consumption tax policy")
ax[0, 0].plot(dY[2][:50, :] * ss0['Y'],'-.', label="Transfer policy")
ax[0, 0].legend(loc='upper right', frameon=False)

ax[0, 1].set_title(r'Consumption $C$')
#ax[0, 1].plot(dC[0][:50, :])
ax[0, 1].plot(dC[1][:50, :] * ss0['C'])
ax[0, 1].plot(dC[2][:50, :] * ss0['C'],'-.')

ax[0, 2].set_title(r'Government debt $B$')
#ax[0, 2].plot(dB[0][:50, :])
ax[0, 2].plot(dB[1][:50, :] * ss0['A'])
ax[0, 2].plot(dB[2][:50, :] * ss0['A'],'-.')

ax[0, 3].set_title(r'Transfer $\tau$')
ax[0, 3].plot(np.zeros(50))
ax[0, 3].plot(dT[2][:50, :] * ss0['Trans'],'-.')

ax[1, 0].set_title(r'Wage $w$')
#ax[1, 0].plot(dw[0][:50, :])
ax[1, 0].plot(dw[1][:50, :] * ss0['w'])
ax[1, 0].plot(dw[2][:50, :] * ss0['w'],'-.')

ax[1, 1].set_title(r'Inflation $\pi$')
#ax[1, 1].plot(dp[0][:50, :])
ax[1, 1].plot(dp[1][:50, :])
ax[1, 1].plot(dp[2][:50, :],'-.')

#ax[1, 2].set_title(r'Real interest rate $r$')
#ax[1, 2].plot(dr[0][:50, :])
#ax[1, 2].plot(dr[1][:50, :])
#ax[1, 2].plot(dr[2][:50, :],'-.')

ax[1, 2].set_title(r'Government budget deficit')
#ax[1, 2].plot(dD[0][:50, :])
ax[1, 2].plot(-dD[1][:50, :] * ss0['Deficit'])
ax[1, 2].plot(-dD[2][:50, :] * ss0['Deficit'],'-.')

ax[1, 3].set_title(r'Consumption tax $\tau_c$')
ax[1, 3].plot(dtauc[:50] * ss0['tauc'])
ax[1, 3].plot(np.zeros(50),'-.')

plt.show()


# =============================================================================
# Dynamic properties
# =============================================================================

# Difference consumption tax vs transfers
dif = [['DIFFERENCE \u03C4c vs \u03C4','IMPACT RATIO','CUMULATIVE SUM'],
      ['Shocks',np.ndarray.item(- (dtauc[:1] * ss0['tauc']) / (dT[2][:1, :] * ss0['Trans'])), - np.sum(dtauc) - np.sum(dT[2][:300])],
      ['Output',np.ndarray.item(dY[1][:1, :] / dY[2][:1, :]), np.sum(dY[1][:300, :]) - np.sum(dY[2][:300, :])],
      ['Consumption',np.ndarray.item(dC[1][:1, :] / dC[2][:1, :]), np.sum(dC[1][:300, :]) - np.sum(dC[2][:300, :])],
      ['Inflation',np.ndarray.item(dp[1][:1, :] / dp[2][:1, :]), np.sum(dp[1][:300, :]) - np.sum(dp[2][:300, :])],
      ['Wage',np.ndarray.item(dw[1][:1, :] / dw[2][:1, :]), np.sum(dw[1][:300, :]) - np.sum(dw[2][:300, :])],
      ['Deficit',np.ndarray.item(dD[1][:1, :] / dD[2][:1, :]), np.sum(dD[1][:300, :]) - np.sum(dD[2][:300, :])]]
dash = '-' * 50
for i in range(len(dif)):
    if i == 0:
        print(dash)
        print('{:<20s} {:^12s}  {:>15s}'.format(dif[i][0],dif[i][1],dif[i][2]))
        print(dash)
    else:
        print('{:<20s} {:^12.3f}  {:>15.3f}'.format(dif[i][0],dif[i][1],dif[i][2]))
        
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
