"""One-Asset HANK model with exogenous transfers, taxes, and sticky wages"""

# =============================================================================
# Model
# =============================================================================

print("STICKY WAGE MODEL")
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
def fiscal(B, C, N, r, Tau, tauc, taun, w):
    govt_res = Tau + (1 + r) * B(-1) - tauc * C - taun * w * N - B
    Deficit = Tau - tauc * C - taun * w * N # primary deficit
    Trans = Tau
    return govt_res, Deficit, Trans

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
# Assemble and solve model
# =============================================================================

# Steady state
blocks_ss = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc_ss, union_ss]
hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'gamma': 1.0, 'nu': 20000.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 8,
               'amin': 0, 'amax': 150, 'nA': 50, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0,
               'kappaw': 5000000, 'muw': 1.1, 'N': 1.0, 'tauc': 0.1, 'taun': 0.036}

unknowns_ss = {'beta': 0.986, 'Tau': -0.03}
targets_ss = {'asset_mkt': 0, 'govt_res': 0}

print("Computing steady state...", end=" ")
ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")
print("Done")

# Dynamic model and Jacobian
blocks = [hh_inp, firm, monetary, fiscal, mkt_clearing, nkpc,wage,union]
hank = create_model(blocks, name="One-Asset HANK")
ss = hank.steady_state(ss0)
    
T = 300
exogenous = ['rstar','Tau', 'Z', 'tauc']
unknowns = ['pi', 'w', 'Y', 'B']
targets = ['nkpc_res', 'asset_mkt', 'wnkpc', 'govt_res']

print("Computing Jacobian...", end=" ")
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)
print("Done")


#==============================================================================
# Steady-state properties
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
# l_bin = l_dist.reshape(-1, 100, order='F').sum(axis=0)

# Wealth distribution
a_tot = np.sum(np.multiply(a_ss, D_ss))
a_dist = np.sum(np.multiply(a_ss, D_ss), axis=0)
# a_bin = a_dist.reshape(-1, 100, order='F').sum(axis=0)

# Income distribution
y_ss = ((1 - taun) * w_ss * np.multiply(N_ss, e_grid[:, None]) + r_ss * a_ss + T_ss[:, None]) / (1 + tauc)
y_dist = np.sum(np.multiply(y_ss, D_ss), axis=0)
y_tot = np.sum(y_dist)
# y_bin = y_dist.reshape(-1, 100, order='F').sum(axis=0)

# Consumption distribution
c_tot = np.sum(np.multiply(c_ss, D_ss))
c_dist = np.sum(np.multiply(c_ss, D_ss), axis=0)
# c_bin = c_dist.reshape(-1, 100, order='F').sum(axis=0)

# # Labor supply distribution
# n_tot = np.sum(np.multiply(n_ss, D_ss))
# n_dist = np.sum(np.multiply(n_ss, D_ss), axis=0)
# n_bin = n_dist.reshape(-1, 100, order='F').sum(axis=0)

# Dividend distribution
d_tot = np.sum(np.multiply(T_ss[:, None] - Tau_ss, D_ss))
d_dist = np.sum(np.multiply(T_ss[:, None] - Tau_ss, D_ss), axis=0)
# d_bin = d_dist.reshape(-1, 100, order='F').sum(axis=0)

# Transfer distribution
tau_tot = np.sum(np.multiply(Tau_ss, D_ss))
tau_dist = np.sum(np.multiply(Tau_ss, D_ss), axis=0)
# tau_bin = tau_dist.reshape(-1, 100, order='F').sum(axis=0)

# Wealth Lorenz curve
D_grid = np.append(np.zeros(1), np.cumsum(D_dist))
a_lorenz = np.append(np.zeros(1), np.cumsum(a_dist / a_tot))
a_lorenz_area = np.trapz(a_lorenz, x=D_grid) # area below Lorenz curve
a_gini = (0.5 - a_lorenz_area) / 0.5
# print("Wealth Gini =", np.round(a_gini, 3))

# Income Lorenz curve
y_lorenz = np.append(np.zeros(1), np.cumsum(y_dist / y_tot))
y_lorenz_area = np.trapz(y_lorenz, x=D_grid) # area below Lorenz curve
y_gini = (0.5 - y_lorenz_area) / 0.5
# print("Income Gini =", np.round(y_gini, 3))

# # Plot distributions
# plt.rcParams["figure.figsize"] = (20,7)
# fig, ax = plt.subplots(2, 4)

# ax[0, 0].set_title(r'Skill $e$ distribution')
# ax[0, 0].plot(e_grid, pi_e)
# ax[0, 0].fill_between(e_grid, pi_e)

# ax[0, 1].set_title(r'Poplulation distribution')
# ax[0, 1].plot(l_bin)
# ax[0, 1].fill_between(range(100), l_bin)

# ax[0, 2].set_title(r'Wealth $a$ distribution')
# ax[0, 2].plot(a_bin / a_tot)
# ax[0, 2].fill_between(range(100), a_bin / a_tot)

# ax[0, 3].set_title(r'Income $y$ and consumption $c$ distribution')
# ax[0, 3].plot(y_bin / y_tot)
# ax[0, 3].fill_between(range(100), y_bin / y_tot)
# ax[0, 3].plot(c_bin / c_tot)

# # ax[0, 3].set_title(r'Labor supply $n$ distribution')
# # ax[0, 3].plot(n_bin / n_tot)
# # ax[0, 3].fill_between(range(100), n_bin / n_tot)

# ax[1, 0].set_title(r'Dividend $d$ distribution')
# ax[1, 0].plot(d_bin / d_tot)
# ax[1, 0].fill_between(range(100), d_bin / d_tot)

# ax[1, 1].set_title(r'Transfer $\tau$ distribution')
# ax[1, 1].plot(tau_bin / tau_tot)
# ax[1, 1].fill_between(range(100), tau_bin / tau_tot)

# ax[1, 2].set_title(r'Wealth Lorenz curve')
# ax[1, 2].plot(D_grid, a_lorenz)
# ax[1, 2].plot([0, 1], [0, 1], '-')

# ax[1, 3].set_title(r'Income Lorenz curve')
# ax[1, 3].plot(D_grid, y_lorenz) 
# ax[1, 3].plot([0, 1], [0, 1], '-')
# plt.show()

# Show steady state
ss_param = [['Discount factor', beta, 'Intertemporal elasticity', gamma],
        ['Labor supply elasticity', 1 / nu, 'Labor supply disutility', phi],  
        ['Goods substitutability', ss['mu'] / (ss['mu'] - 1) , 'Price markup', ss['mu']],
        ['Phillips curve slope', ss['kappa'], 'Taylor rule inflation ', ss['phi_pi']],
        ['Wage Phillips curve slope', ss['kappaw'], 'Wage markup ', ss['muw']],
        ['Consumption tax rate', ss['tauc'], 'Labor tax rate', ss['taun']]]

ss_var = [['Output', ss['Y'], 'Government debt', ss['A']],
        ['Consumption', ss['C'], 'Transfers', ss['Tau']],
        ['Hours', ss['N'], 'Dividends', ss['Div']], 
        ['Wage', ss['w'], 'Marginal cost', ss['w'] / ss['Z']],
        ['Inflation', ss['pi'], 'Consumption tax revenue', ss['tauc'] * ss['C']],
        ['Nominal interest rate', ss['r']*(1+ss['pi']), 'Labor tax revenue', ss['taun']*ss['N']*ss['w']],
        ['Real interest rate', ss['r'], 'Debt servicing  cost', ss['r'] * ss['A']]]

ss_mom = [['Share of hand-to-mouth', htm, 'Gini index', a_gini]]

ss_mkt = [['Bond market', ss['asset_mkt'], 'Government budget', ss['govt_res']],
          ['Goods market (resid)', ss['goods_mkt'], 'Goods market (resid)', ss['goods_mkt']]]

print('\nPARAMETERS')
for i in range(len(ss_param)):
      print('{:<26s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_param[i][0],ss_param[i][1],ss_param[i][2],ss_param[i][3]))
print('\nSTEADY STATE')
for i in range(len(ss_var)):
      print('{:<26s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_var[i][0],ss_var[i][1],ss_var[i][2],ss_var[i][3]))
# print('\nDISTRIBUTIONAL VARIABLES IN STEADY STATE')
for i in range(len(ss_mom)):
      print('{:<26s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_mom[i][0],ss_mom[i][1],ss_mom[i][2],ss_mom[i][3]))
# print('\nMARKET CLEARING')
# for i in range(len(ss_mkt)):
      # print('{:<24s}{:>12.0e}   {:24s}{:>10.0e}'.format(ss_mkt[i][0],ss_mkt[i][1],ss_mkt[i][2],ss_mkt[i][3]))


# =============================================================================
# Impulse response functions
# =============================================================================

# Standard shock
discount = (1 / (1 + r_ss))
rhos = 0.78
drstar = -0.002 * rhos ** np.arange(T)
# dtau = 0.01 * rhos ** np.arange(T)
# dtauc = - dtau

# Zero net present value sock
shock = np.zeros(T)
s1, s2, s3, s4, s5 = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
dtau = shock
dtauc = - shock

# # Integral
# intshock = (1 / (s3-np.log(discount))** 2 * np.exp(s3*s5-s4) * 
#              (discount**T * np.exp(-30*s3) * (np.log(discount) * (s1+s2*(s5-T)) - s1*s3+s2*(-s3)*(s5-T)+s2) 
#               - np.log(discount) * (s1+s2*s5) + s1*s3 + s2*s3*s5 - s2))

# Discounted cumulative sum
# cumshock = lambda i : np.cumsum(discount ** i * dtauc[i])
cumshock = np.zeros(T)
for i in range(T):
    cumshock[i] = discount ** i * dtau[i]
print(np.sum(cumshock))
    
# IRFs
dY = [G['Y']['tauc'] @ dtauc, G['Y']['Tau'] @ dtau, G['Y']['rstar'] @ drstar]
dC = [G['C']['tauc'] @ dtauc, G['C']['Tau'] @ dtau,  G['C']['rstar'] @ drstar]
dN = [G['N']['tauc'] @ dtauc, G['N']['Tau'] @ dtau, G['N']['rstar'] @ drstar]
dB = [G['A']['tauc'] @ dtauc, G['A']['Tau'] @ dtau, G['A']['rstar'] @ drstar]
dw = [G['w']['tauc'] @ dtauc, G['w']['Tau'] @ dtau, G['w']['rstar'] @ drstar]
dp = [G['pi']['tauc'] @ dtauc, G['pi']['Tau'] @ dtau, G['pi']['rstar'] @ drstar]
dr = [G['r']['tauc'] @ dtauc, G['r']['Tau'] @ dtau, G['r']['rstar'] @ drstar]
dD = [G['Deficit']['tauc'] @ dtauc, G['Deficit']['Tau'] @ dtau, G['Deficit']['rstar'] @ drstar]
dd = [G['Div']['tauc'] @ dtauc, G['Div']['Tau'] @ dtau, G['Div']['rstar'] @ drstar]
dT = [np.zeros(T), G['Trans']['Tau'] @ dtau, np.zeros(T)]
dTc = [dtauc, np.zeros(T), np.zeros(T)]
di = [np.zeros(T), np.zeros(T), G['i']['rstar'] @ drstar]

plt.rcParams["figure.figsize"] = (20,7)
fig, ax = plt.subplots(2, 4)
iT = 30
fig.suptitle('Consumption tax cut versus transfer increase, sticky wages', size=16)

ax[0, 0].set_title(r'Output $Y$')
ax[0, 0].plot(dY[0][:iT], label="Consumption tax policy")
ax[0, 0].plot(dY[1][:iT],'-.', label="Transfer policy")
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

# ax[1, 2].set_title(r'government budget deficit')
# ax[1, 2].plot(-dD[0][:50])
# ax[1, 2].plot(-dD[1][:50],'-.')

ax[1, 3].set_title(r'Consumption tax $\tau_c$')
ax[1, 3].plot(dTc[0][:iT])
ax[1, 3].plot(dTc[1][:iT],'-.')
plt.show()

# Discounted cumulative sum
cumtau, cumY, cumC, cumP, cumW, cumD = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
for i in range(T):
    cumtau[i] = discount ** i * (dtauc[i] + dT[1][i])
    cumY[i] = discount ** i * (dY[0][i] - dY[1][i])
    cumC[i] = discount ** i * (dC[0][i] - dC[1][i])
    cumP[i] = discount ** i * (dp[0][i] - dp[1][i])
    cumW[i] = discount ** i * (dw[0][i] - dw[1][i])
    cumD[i] = discount ** i * (dD[0][i] - dD[1][i])

# Impact difference
dif = [['\nDIFFERENCE \u03C4c vs \u03C4','IMPACT RATIO','CUMULATIVE SUM'],
      ['Shocks', - (dtauc[0] * tauc) / (dT[1][0] * Tau_ss), 100 * np.sum(cumtau)],
      ['Output', dY[0][0] / dY[1][0], 100 * np.sum(cumY)],
      ['Consumption', dC[0][0] / dC[1][0], 100 * np.sum(cumC)],
      ['Inflation', dp[1][0] / dp[1][0], 100 * np.sum(cumP)],
      ['Wage', dw[0][0] / dw[1][0], 100 * np.sum(cumW)],
      ['Deficit', dD[0][0] / dD[1][0], 100 * np.sum(cumD)]]
for i in range(len(dif)):
    if i == 0:
        print('{:<20s} {:^10s}  {:>14s}'.format(dif[i][0], dif[i][1], dif[i][2]))
    else:
        print('{:<20s} {:^10.3f}  {:>10.3f}'.format(dif[i][0],dif[i][1],dif[i][2]),"%") 

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
        
# =============================================================================
# Impact response by wealth percentile
# =============================================================================

# Policy 1: consumption tax
print("\nPOLICY 1: CONSUMPTION TAX")

# Aggregate transition dynamics
path_n_tauc = N_ss + G['N']['tauc'] @ dtauc
path_r_tauc = r_ss + G['r']['tauc'] @ dtauc
path_w_tauc = w_ss + G['w']['tauc'] @ dtauc
path_div_tauc = Div_ss + G['Div']['tauc'] @ dtauc
path_tauc_tauc = tauc + dtauc

# Initialize value function
V_prime_p_start = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)

# Compute all individual consumption paths
print("Computing individual paths...", end=" ")
V_prime_p_tauc = V_prime_p_start
c_all_tauc = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tauc, _, c, _ = iterate_h(household_d, V_prime_p_tauc, a_grid, e_grid, Pi, pi_e, beta, gamma,  
                                        path_div_tauc[t], path_n_tauc[t], path_r_tauc[t], Tau_ss, path_tauc_tauc[t], taun, path_w_tauc[t])
    c_all_tauc[:, :, t] = c  
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
V_prime_p_tauc = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)
c_direct_tauc = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tauc, _, c, _ = iterate_h(household_d, V_prime_p_tauc, a_grid, e_grid, Pi, pi_e, beta, gamma,   
                                        Div_ss, N_ss, r_ss, Tau_ss, path_tauc_tauc[t], taun, w_ss)
    c_direct_tauc[:, :, t] = c  
print("Done")


# Policy 2: transfer policy
print("POLICY 2: TRANSFERS")

# Aggregate transition dynamics
path_div_tau = Div_ss + G['Div']['Tau'] @ dtau
path_n_tau = N_ss + G['N']['Tau'] @ dtau
path_r_tau = r_ss + G['r']['Tau'] @ dtau
path_tau_tau = Tau_ss + dtau
path_w_tau = w_ss + G['w']['Tau'] @ dtau

# Compute all individual consumption paths
print("Computing individual paths...", end=" ")
V_prime_p_tau = V_prime_p_start
c_all_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid, e_grid, Pi, pi_e, beta, gamma,
                                        path_div_tau[t], path_n_tau[t], path_r_tau[t], path_tau_tau[t], tauc, taun, path_w_tau[t])
    c_all_tau[:, :, t] = c  
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
V_prime_p_tau = V_prime_p_start
c_direct_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid, e_grid, Pi, pi_e, beta, gamma,
                                        Div_ss, N_ss, r_ss, path_tau_tau[t], tauc, taun, w_ss)
    c_direct_tau[:, :, t] = c  
print("Done")

# Select first period only and express as deviation from steady state
c_first_dev_tauc = (c_all_tauc[:, :, 0] - c_ss) / c_ss
c_first_dev_tauc_direct = (c_direct_tauc[:, :, 0] - c_ss) / c_ss
c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss) / c_ss
c_first_dev_tau_direct = (c_direct_tau[:, :, 0] - c_ss) / c_ss

# Weigh response by mass of agents
c_first_tauc, c_first_tauc_direct, c_first_tau, c_first_tau_direct = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
for i in range(nA):
    c_first_tauc[i] = (c_first_dev_tauc[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    c_first_tauc_direct[i] = (c_first_dev_tauc_direct[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    c_first_tau[i] = (c_first_dev_tau[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    c_first_tau_direct[i] = (c_first_dev_tau_direct[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    
# Compute indirect effects
c_first_tauc_indirect = c_first_tauc - c_first_tauc_direct
c_first_tau_indirect = c_first_tau - c_first_tau_direct
    
# # Pool into percentile bins
# c_first_bin_tauc = c_first_tauc.reshape(-1, 100, order='F').mean(axis=0)
# c_first_bin_tauc_direct = c_first_tauc_direct.reshape(-1, 100, order='F').mean(axis=0) 
# c_first_bin_tauc_indirect = c_first_bin_tauc - c_first_bin_tauc_direct
# c_first_bin_tau = c_first_tau.reshape(-1, 100, order='F').mean(axis=0)  
# c_first_bin_tau_direct = c_first_tau_direct.reshape(-1, 100, order='F').mean(axis=0) 
# c_first_bin_tau_indirect = c_first_bin_tau - c_first_bin_tau_direct

# X-axis
D_ss_quant = 100 * np.cumsum(np.sum(D_ss, axis=0))

# First percentile
D_ss_quant = np.append(0, D_ss_quant)
c_first_tauc_direct = np.append(c_first_tauc_direct[0], c_first_tauc_direct)
c_first_tauc_indirect =  np.append(c_first_tauc_indirect[0], c_first_tauc_indirect)
c_first_tau_direct = np.append(c_first_tau_direct[0], c_first_tau_direct)
c_first_tau_indirect =  np.append(c_first_tau_indirect[0], c_first_tau_indirect)
 
# Plot results
color_map = ["#FFFFFF", "#D95319"] # myb: "#0072BD"
fig, ax = plt.subplots(1,2)
ax[0].set_title(r'Consumption tax policy')
ax[0].plot(D_ss_quant, 100 * c_first_tauc_direct, label="Direct effect", linewidth=3)  
ax[0].stackplot(D_ss_quant, 100 * c_first_tauc_direct, 100 * c_first_tauc_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5) 
# ax[0].plot(c_first_bin_rstar_indirect, label="indirect effect") 
ax[0].legend(loc='upper left', frameon=False)
ax[0].set_xlabel("Wealth percentile"), ax[0].set_ylabel("Percent deviation from steady state")

ax[1].set_title(r'Transfer policy')
ax[1].plot(D_ss_quant, 100 * c_first_tau_direct, label="Direct effect", linewidth=3)    
ax[1].stackplot(D_ss_quant, 100 * c_first_tau_direct, 100 * c_first_tau_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)   
ax[1].legend(loc='upper right', frameon=False)
ax[1].set_xlabel("Wealth percentile")
plt.show()
     
print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))

