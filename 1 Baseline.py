"""One-Asset HANK model with exogenous transfers and taxes"""

# =============================================================================
# Model
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
        mid_point = (a + b) / 2
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

# Household heterogeneous block
def consumption(c, we, rest, gamma, nu, phi, tauc, taun):
    return (1 + tauc) * c - (1 - taun) * we * ((1 - taun) * we / ((1 + tauc) * phi * c ** gamma)) ** (1/nu) - rest

def household_guess(a_grid, e_grid, r, w, gamma, T, tauc, taun):
    wel = (1 + r) * a_grid[np.newaxis,:] + (1 - taun) * w * e_grid[:,np.newaxis] + T[:,np.newaxis]
    V_prime = (1 + r) / (1 + tauc) * (wel * 0.1) ** (-gamma) # check
    return V_prime

@het(exogenous='Pi', policy='a', backward='V_prime', backward_init=household_guess)
def household(V_prime_p, a_grid, e_grid, beta, gamma, nu, phi, tauc, taun, r, w, T):
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

# Simple blocks
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
# Household iteration policy rule
# =============================================================================

def household_d(V_prime_p, a_grid, e_grid, beta, gamma, nu, phi, tauc, taun, r, w, T):
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

def iterate_h(foo, V_prime_start, Pi, pi_e, a_grid, e_grid, beta, gamma, nu, phi,  
              taun, r, w, Div, Tau, tauc, maxit=1000, tol=1E-8):
    V_prime_p = Pi @ V_prime_start
    V_prime_old = V_prime_start    
    ite = 0
    err = 1
    T = transfers(pi_e, Div, Tau, e_grid)
    
    while ite < maxit and err > tol:
        V_prime_temp, a, c, n = foo(V_prime_p, a_grid, e_grid, beta, gamma, nu, phi, tauc, taun, r, w, T)
        V_prime_p = Pi @ V_prime_temp
        ite += 1
        err = np.max(np.abs(V_prime_old - V_prime_temp))
        V_prime_old = V_prime_temp 
    return V_prime_temp, a, c, n


# =============================================================================
# Assemble and solve model
# =============================================================================

# Steady state
blocks_ss = [hh_ext, firm, monetary,fiscal, nkpc_ss, mkt_clearing]
hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'gamma': 1.0, 'nu': 2.0, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 8,
               'amin': 0, 'amax': 200, 'nA': 100, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi_pi': 0.0, 'B': 6.0, 
               'tauc': 0.1, 'taun': 0.036}

unknowns_ss = {'beta': 0.986, 'phi': 0.8, 'Tau': 0.05}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0, 'govt_res': 0}
print("Computing steady state...")
ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")
print("Done")

# Dynamic model and Jacobian
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
n_ss = ss.internals['household']['n']
T_ss = transfers(pi_e, Div_ss, Tau_ss, e_grid)

# Share of hand-to-mouth
D_dist = np.sum(D_ss, axis=0)
htm = D_dist[0]

# Share of hand-to-mouth, other method
zero_asset = np.where(a_ss == 0)
htm = np.sum(D_ss[zero_asset])

# # Population distribution
# l_tot = np.sum(D_ss);
# l_dist = np.sum(D_ss, axis=0)
# l_bin = l_dist.reshape(-1, 100, order='F').sum(axis=0)

# Wealth distribution
a_tot = np.sum(np.multiply(a_ss, D_ss))
a_dist = np.sum(np.multiply(a_ss, D_ss), axis=0)
# a_bin = a_dist.reshape(-1, 100, order='F').sum(axis=0)

# # Income distribution
# y_ss = ((1 - taun) * w_ss * np.multiply(n_ss, e_grid[:, None]) + r_ss * a_ss + T_ss[:, None]) / (1 + tauc)
# y_dist = np.sum(np.multiply(y_ss, D_ss), axis=0)
# y_tot = np.sum(y_dist)
# y_bin = y_dist.reshape(-1, 100, order='F').sum(axis=0)

# # Consumption distribution
# c_tot = np.sum(np.multiply(c_ss, D_ss))
# c_dist = np.sum(np.multiply(c_ss, D_ss), axis=0)
# c_bin = c_dist.reshape(-1, 100, order='F').sum(axis=0)

# # Labor supply distribution
# n_tot = np.sum(np.multiply(n_ss, D_ss))
# n_dist = np.sum(np.multiply(n_ss, D_ss), axis=0)
# n_bin = n_dist.reshape(-1, 100, order='F').sum(axis=0)

# # Dividend distribution
# d_tot = np.sum(np.multiply(T_ss[:, None] - Tau_ss, D_ss))
# d_dist = np.sum(np.multiply(T_ss[:, None] - Tau_ss, D_ss), axis=0)
# d_bin = d_dist.reshape(-1, 100, order='F').sum(axis=0)

# # Transfer distribution
# tau_tot = np.sum(np.multiply(Tau_ss, D_ss))
# tau_dist = np.sum(np.multiply(Tau_ss, D_ss), axis=0)
# tau_bin = tau_dist.reshape(-1, 100, order='F').sum(axis=0)

# Wealth Lorenz curve
D_grid = np.append(np.zeros(1), np.cumsum(D_dist))
a_lorenz = np.append(np.zeros(1), np.cumsum(a_dist / a_tot))
a_lorenz_area = np.trapz(a_lorenz, x=D_grid) # area below Lorenz curve
a_gini = (0.5 - a_lorenz_area) / 0.5
# print("Wealth Gini =", np.round(a_gini, 3))

# # Income Lorenz curve
# y_lorenz = np.append(np.zeros(1), np.cumsum(y_dist / y_tot))
# y_lorenz_area = np.trapz(y_lorenz, x=D_grid) # area below Lorenz curve
# y_gini = (0.5 - y_lorenz_area) / 0.5
# # print("Income Gini =", np.round(y_gini, 3))

# # Plot distributions
# plt.rcParams["figure.figsize"] = (16,7)
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
# ax[0, 3].plot(y_bin / y_tot, label = "Income distribution")
# ax[0, 3].fill_between(range(100), y_bin / y_tot)
# ax[0, 3].plot(c_bin / c_tot, label = "Consumption distribution")
# ax[0, 3].legend(frameon=False)

# ax[1, 0].set_title(r'Labor supply $n$ distribution')
# ax[1, 0].plot(n_bin / n_tot)
# ax[1, 0].fill_between(range(100), n_bin / n_tot)

# ax[1, 1].set_title(r'Transfer $\tau$ and dividend $d$ distribution')
# ax[1, 1].plot(tau_bin / tau_tot, label = "Transfer distribution")
# ax[1, 1].fill_between(range(100), tau_bin / tau_tot)
# ax[1, 1].plot(d_bin / d_tot, label = "Dividend distribution")
# # ax[1, 3].fill_between(range(100), d_bin / d_tot)
# ax[1, 1].legend(frameon=False)

# ax[1, 2].set_title(r'Wealth Lorenz curve')
# ax[1, 2].plot(D_grid, a_lorenz)
# ax[1, 2].plot([0, 1], [0, 1], '-')

# ax[1, 3].set_title(r'Income Lorenz curve')
# ax[1, 3].plot(D_grid, y_lorenz) 
# ax[1, 3].plot([0, 1], [0, 1], '-')
# plt.show()

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
        ['Real interest rate', ss['r'], 'Debt servicing cost', ss['r'] * ss['A']]]

ss_mom = [['Share of hand-to-mouth', htm, 'Wealth Gini index', a_gini]]

ss_mkt = [['Bond market', ss['asset_mkt'], 'Labor market', ss['labor_mkt']],
          ['Goods market (resid)', ss['goods_mkt'], 'Government budget', ss['govt_res']]]

dash = '-' * 73
# print(dash)
print('\nPARAMETERS')
for i in range(len(ss_param)):
      print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_param[i][0],ss_param[i][1],ss_param[i][2],ss_param[i][3]))
print('\nAGGREGATE VARIABLES IN STEADY STATE')
for i in range(len(ss_var)):
      print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_var[i][0],ss_var[i][1],ss_var[i][2],ss_var[i][3]))
print('\nDISTRIBUTIONAL VARIABLES IN STEADY STATE')
for i in range(len(ss_mom)):
      print('{:<24s}{:>12.3f}   {:24s}{:>10.3f}'.format(ss_mom[i][0],ss_mom[i][1],ss_mom[i][2],ss_mom[i][3]))
# print('\nMARKET CLEARING')
# for i in range(len(ss_mkt)):
      # print('{:<24s}{:>12.0e}   {:24s}{:>10.0e}'.format(ss_mkt[i][0],ss_mkt[i][1],ss_mkt[i][2],ss_mkt[i][3]))


# =============================================================================
# Impulse response functions
# =============================================================================

# Standard shock
discount = (1 / (1 + r_ss))
rhos = 0.9
drstar = -0.02 * rhos ** np.arange(T)
dtau = 0.03 * rhos ** np.arange(T)
dtauc = - dtau

# Zero net present value sock
shock = np.zeros(T)
s1, s2, s3, s4, s5 = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
dtau = shock
dtauc = - shock

# # Integral
# intshock = (1 / (s3-np.log(discount))** 2 * np.exp(s3*s5-s4) * 
#             (discount**T * np.exp(-60*s3) * (np.log(discount) * (s1+s2*(s5-T)) - s1*s3+s2*(-s3)*(s5-T)+s2) 
#              - np.log(discount) * (s1+s2*s5) + s1*s3 + s2*s3*s5 - s2))

dY = [G['Y']['tauc'] @ dtauc, G['Y']['Tau'] @ dtau, G['Y']['rstar'] @ drstar]
dC = [G['C']['tauc'] @ dtauc, G['C']['Tau'] @ dtau, G['C']['rstar'] @ drstar]
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

plt.rcParams["figure.figsize"] = (16,7)
fig, ax = plt.subplots(2, 4)
iT = 30
fig.suptitle('Consumption tax cut versus transfer increase, baseline model', size=16)

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
dash = '-' * 50
# print(dash)
for i in range(len(dif)):
    if i == 0:
        print('{:<20s} {:^10s}  {:>14s}'.format(dif[i][0], dif[i][1], dif[i][2]))
    else:
        print('{:<20s} {:^10.3f}  {:>10.3f}'.format(dif[i][0],dif[i][1],dif[i][2]),"%")
# print(dash)   


# =============================================================================
# Impact response by wealth percentile
# =============================================================================

# Policy 1: consumption tax
print("\nPOLICY 1: CONSUMPTION TAX")

# Aggregate transition dynamics
path_r_tauc = r_ss + G['r']['tauc'] @ dtauc
path_w_tauc = w_ss + G['w']['tauc'] @ dtauc
path_div_tauc = Div_ss + G['Div']['tauc'] @ dtauc
path_tauc_tauc = tauc + dtauc

# Initialize value function
V_prime_p_start = (1 + r_ss) / (1 + tauc) * c_ss ** (-gamma)

# Compute all individual consumption paths
print("Computing individual paths...")
V_prime_tauc = V_prime_p_start
c_all_tauc, n_all_tauc = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tauc, _, c, n = iterate_h(household_d, V_prime_p_start, Pi, pi_e, a_grid, e_grid, beta, gamma, nu, phi, taun,
                                        path_r_tauc[t], path_w_tauc[t], path_div_tauc[t], Tau_ss, path_tauc_tauc[t])
    c_all_tauc[:, :, t] = c
    n_all_tauc[:, :, t] = n
print("Done")

# Direct effect of policy
print("Computing direct effect...")
V_prime_tauc = V_prime_p_start
c_direct_tauc, n_direct_tauc = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tauc, _, c, n = iterate_h(household_d, V_prime_p_start, Pi, pi_e, a_grid, e_grid, beta, gamma, nu, phi, taun,  
                                        r_ss, path_w_tauc[t], Div_ss, Tau_ss, path_tauc_tauc[t])
    c_direct_tauc[:, :, t] = c
    n_direct_tauc[:, :, t] = n
print("Done")


# Policy 2: transfers
print("POLICY 2: TRANSFERS")

# Aggregate transition dynamics
path_r_tau = r_ss + G['r']['Tau'] @ dtau
path_w_tau = w_ss + G['w']['Tau'] @ dtau
path_div_tau = Div_ss + G['Div']['Tau'] @ dtau
path_tau_tau = Tau_ss + dtau

# Compute all individual consumption paths
print("Computing individual paths...")
V_prime_tau = V_prime_p_start
c_all_tau, n_all_tau = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, n = iterate_h(household_d, V_prime_p_start, Pi, pi_e, a_grid, e_grid, beta, gamma, nu, phi, taun,  
                                        path_r_tau[t], path_w_tau[t], path_div_tau[t], path_tau_tau[t], tauc)
    c_all_tau[:, :, t] = c
    n_all_tau[:, :, t] = n
print("Done")

# Direct effect of policy
print("Computing direct effect...")
V_prime_tau = V_prime_p_start
c_direct_tau, n_direct_tau = np.zeros((nE, nA, T)), np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, n = iterate_h(household_d, V_prime_p_start, Pi, pi_e, a_grid, e_grid, beta, gamma, nu, phi, taun,  
                                        r_ss, w_ss, Div_ss, path_tau_tau[t], tauc)
    c_direct_tau[:, :, t] = c 
    n_direct_tau[:, :, t] = n
print("Done")

# Select first period only and express as deviation from steady state
c_first_dev_tauc = (c_all_tauc[:, :, 0] - c_ss) / c_ss
c_first_dev_tauc_direct = (c_direct_tauc[:, :, 0] - c_ss) / c_ss
c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss) / c_ss
c_first_dev_tau_direct = (c_direct_tau[:, :, 0] - c_ss) / c_ss

n_first_dev_tauc = (n_all_tauc[:, :, 0] - n_ss) / n_ss
n_first_dev_tauc_direct = (n_direct_tauc[:, :, 0] - n_ss) / n_ss
n_first_dev_tau = (n_all_tau[:, :, 0] - n_ss) / n_ss
n_first_dev_tau_direct = (n_direct_tau[:, :, 0] - n_ss) / n_ss

# Weigh response by mass of agents
c_first_tauc, c_first_tauc_direct, c_first_tau, c_first_tau_direct = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
n_first_tauc, n_first_tauc_direct, n_first_tau, n_first_tau_direct = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
for i in range(nA):
    c_first_tauc[i] = (c_first_dev_tauc[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    c_first_tauc_direct[i] = (c_first_dev_tauc_direct[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    c_first_tau[i] = (c_first_dev_tau[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    c_first_tau_direct[i] = (c_first_dev_tau_direct[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    
    n_first_tauc[i] = (n_first_dev_tauc[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    n_first_tauc_direct[i] = (n_first_dev_tauc_direct[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    n_first_tau[i] = (n_first_dev_tau[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    n_first_tau_direct[i] = (n_first_dev_tau_direct[:, i] @ D_ss[:, i]) / np.sum(D_ss[:,i])
    
# Indirect effects    
c_first_tauc_indirect = c_first_tauc - c_first_tauc_direct
c_first_tau_indirect = c_first_tau - c_first_tau_direct
n_first_tauc_indirect = n_first_tauc - n_first_tauc_direct
n_first_tau_indirect = n_first_tau - n_first_tau_direct  
    
# # Pool into percentile bins
# c_first_bin_tauc = c_first_tauc.reshape(-1, 100, order='F').mean(axis=0)
# c_first_bin_tauc_direct = c_first_tauc_direct.reshape(-1, 100, order='F').mean(axis=0) 
# c_first_bin_tauc_indirect = c_first_bin_tauc - c_first_bin_tauc_direct
# c_first_bin_tau = c_first_tau.reshape(-1, 100, order='F').mean(axis=0)  
# c_first_bin_tau_direct = c_first_tau_direct.reshape(-1, 100, order='F').mean(axis=0) 
# c_first_bin_tau_indirect = c_first_bin_tau - c_first_bin_tau_direct

# n_first_bin_tauc = n_first_tauc.reshape(-1, 100, order='F').mean(axis=0)
# n_first_bin_tauc_direct = n_first_tauc_direct.reshape(-1, 100, order='F').mean(axis=0) 
# n_first_bin_tauc_indirect = n_first_bin_tauc - n_first_bin_tauc_direct
# n_first_bin_tau = n_first_tau.reshape(-1, 100, order='F').mean(axis=0)  
# n_first_bin_tau_direct = n_first_tau_direct.reshape(-1, 100, order='F').mean(axis=0) 
# n_first_bin_tau_indirect = n_first_bin_tau - n_first_bin_tau_direct
 
D_ss_quant = 100 * np.cumsum(np.sum(D_ss, axis=0))

# First percentile
D_ss_quant = np.append(0, D_ss_quant)
c_first_tauc_direct = np.append(c_first_tauc_direct[0], c_first_tauc_direct)
c_first_tauc_indirect =  np.append(c_first_tauc_indirect[0], c_first_tauc_indirect)
c_first_tau_direct = np.append(c_first_tau_direct[0], c_first_tau_direct)
c_first_tau_indirect =  np.append(c_first_tau_indirect[0], c_first_tau_indirect)
n_first_tauc_direct = np.append(n_first_tauc_direct[0], n_first_tauc_direct)
n_first_tauc_indirect =  np.append(n_first_tauc_indirect[0], n_first_tauc_indirect)
n_first_tau_direct = np.append(n_first_tau_direct[0], n_first_tau_direct)
n_first_tau_indirect =  np.append(n_first_tau_indirect[0], n_first_tau_indirect)

# Plot results
color_map = ["#FFFFFF", "#D95319"] # myb: "#0072BD"
fig, ax = plt.subplots(2,2)
ax[0, 0].set_title(r'Consumption response to consumption tax policy')
ax[0, 0].plot(D_ss_quant, 100 * c_first_tauc_direct, label="Direct effect", linewidth=3)  
ax[0, 0].stackplot(D_ss_quant, 100 * c_first_tauc_direct, 100 * c_first_tauc_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)  
ax[0, 0].legend(loc='upper left', frameon=False)
ax[0, 0].set_ylabel("Percent deviation from steady state")

ax[0, 1].set_title(r'Consumption response to transfer policy')
ax[0, 1].plot(D_ss_quant, 100 * c_first_tau_direct, label="Direct effect", linewidth=3)    
ax[0, 1].stackplot(D_ss_quant, 100 * c_first_tau_direct, 100 * c_first_tau_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)   
ax[0, 1].legend(loc='lower left', frameon=False)

ax[1, 0].set_title(r'Labor supply response to consumption tax policy')
ax[1, 0].plot(D_ss_quant, 100 * n_first_tauc_direct, label="Direct effect", linewidth=3)  
ax[1, 0].stackplot(D_ss_quant, 100 * n_first_tauc_direct, 100 * n_first_tauc_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)  
ax[1, 0].legend(loc='lower left', frameon=False)
ax[1, 0].set_xlabel("Wealth percentile"), ax[1, 0].set_ylabel("Percent deviation from steady state")

ax[1, 1].set_title(r'Labor supply response to transfer policy')
ax[1, 1].plot(D_ss_quant, 100 * n_first_tau_direct, label="Direct effect", linewidth=3)    
ax[1, 1].stackplot(D_ss_quant, 100 * n_first_tau_direct, 100 * n_first_tau_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)   
ax[1, 1].legend(loc='upper left', frameon=False)
ax[1, 1].set_xlabel("Wealth percentile")
plt.show()

fig, ax = plt.subplots(1,2)
fig.suptitle("Labor supply response")
ax[0].set_title(r'Consumption tax policy')
ax[0].plot(n_first_tauc_direct)
ax[1].set_title(r'Transfer policy')
ax[1].plot(n_first_tau_direct)
plt.show()


print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))   