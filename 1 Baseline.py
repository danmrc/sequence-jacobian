"""One-Asset HANK model with exogenous transfer and taxes"""

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
e_grid = ss.internals['household']['e_grid']
a_grid = ss.internals['household']['a_grid']
D_ss = ss.internals['household']['Dbeg']
pi_e =  ss.internals['household']['pi_e']
Pi = ss.internals['household']['Pi']
a_ss = ss.internals['household']['a']
c_ss = ss.internals['household']['c']
n_ss = ss.internals['household']['n']
N_ss = ss['N']
r_ss = ss['r']
Div_ss = ss['Div']
Transfer_ss = ss['Transfer']
w_ss = ss['w']
T_ss = transfers(pi_e, Div_ss, Transfer_ss, e_grid)


# Share of hand-to-mouth
D_dist = np.sum(D_ss, axis=0)
htm = D_dist[0]

# Share of hand-to-mouth, other method
zero_asset = np.where(a_ss == 0)
htm = np.sum(D_ss[zero_asset])

# Wealth distribution
a_tot = np.sum(np.multiply(a_ss, D_ss))
a_dist = np.sum(np.multiply(a_ss, D_ss), axis=0)
a_bin = a_dist.reshape(-1, 100, order='F').sum(axis=0)

# Income distribution
y_ss = ((1 - taun) * w_ss * np.multiply(n_ss, e_grid[:, None]) + r_ss * a_ss + T_ss[:, None]) / (1 + tauc)
y_dist = np.sum(np.multiply(y_ss, D_ss), axis=0)
y_tot = np.sum(y_dist)
y_bin = y_dist.reshape(-1, 100, order='F').sum(axis=0)

# Consumption distribution
c_tot = np.sum(np.multiply(c_ss, D_ss))
c_dist = np.sum(np.multiply(c_ss, D_ss), axis=0)
c_bin = c_dist.reshape(-1, 100, order='F').sum(axis=0)

# Labor supply distribution
n_tot = np.sum(np.multiply(n_ss, D_ss))
n_dist = np.sum(np.multiply(n_ss, D_ss), axis=0)
n_bin = n_dist.reshape(-1, 100, order='F').sum(axis=0)

# Wealth Lorenz curve CHECK 
from numpy import trapz
a_lorenz = trapz(np.cumsum(a_dist / a_tot), dx=1/nA) # area below Lorenz curve
a_gini = (0.5 - a_lorenz) / 0.5
print("Wealth Gini =", np.round(a_gini, 3))

# Income Lorenz curve CHECK 
y_lorenz = trapz(np.cumsum(y_dist / y_tot), dx=1/nA) # area below Lorenz curve
y_gini = (0.5 - y_lorenz) / 0.5
print("Income Gini =", np.round(y_gini, 3))

# Plot distributions
fig, ax = plt.subplots(2, 4)

ax[0, 0].set_title(r'Skill distribution $e$')
ax[0, 0].plot(e_grid, pi_e)
ax[0, 0].fill_between(e_grid, pi_e)

ax[0, 1].set_title(r'Wealth distribution $a$')
ax[0, 1].plot(a_bin / a_tot)
ax[0, 1].fill_between(range(100), a_bin / a_tot)

ax[0, 2].set_title(r'Income distribution $y$')
ax[0, 2].plot(y_bin / y_tot)
ax[0, 2].fill_between(range(100), y_bin / y_tot)

ax[0, 3].set_title(r'Consumption distribution $c$')
ax[0, 3].plot(c_bin / c_tot)
ax[0, 3].fill_between(range(100), c_bin / c_tot)

ax[1, 0].set_title(r'Labor supply distribution $n$')
ax[1, 0].plot(n_bin / n_tot)
ax[1, 0].fill_between(range(100), n_bin / n_tot)

ax[1, 1].set_title(r'Wealth Lorenz curve')
# ax[1, 1].plot(np.cumsum(mass[asset_pos]), np.cumsum(asset_mass / total_wealth)) 
ax[1, 1].plot(np.cumsum(D_dist), np.cumsum(a_dist / a_tot)) 
ax[1, 1].plot([0, 1], [0, 1], '-')

ax[1, 2].set_title(r'Income Lorenz curve')
ax[1, 2].plot(np.cumsum(D_dist), np.cumsum(y_dist / y_tot)) 
ax[1, 2].plot([0, 1], [0, 1], '-')
plt.show()

# Show steady state
ss_param = [['Discount factor', ss['beta'], 'Intertemporal elasticity', ss['gamma']],
        ['Labor supply elasticity', 1 / ss['nu'], 'Labor supply disutility', ss['phi']],  
        ['Goods substitutability', ss['mu'] / (ss['mu'] - 1) , 'Price markup', ss['mu']],
        ['Phillips curve slope', ss['kappa'], 'Taylor rule inflation ', ss['phi_pi']],
        ['Consumption tax rate', ss['tauc'], 'Labor tax rate', ss['taun']]]

ss_var = [['Output', ss['Y'], 'Government debt', ss['A']],
        ['Consumption', ss['C'], 'Transfers', ss['Transfer']],
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


# OLD STUFF 

# # Wealth distribution and Gini coefficient on wealth
# ntype = np.size(ss.internals['household']['a'])
# asset_pos = np.argsort(ss.internals['household']['a'], axis=None) # count position of each array element, ascending order
# asset = np.reshape(ss.internals['household']['a'], (ntype, )) # flatten array
# mass = np.reshape(ss.internals['household']['Dbeg'], (ntype, )) # flatten array
# asset_mass = np.multiply(asset[asset_pos], mass[asset_pos]) / ss['A'] # multiply each asset type by its respective density
# total_wealth = np.sum(asset_mass, axis=None)


# # Consumption distribution
# cons_pos = np.argsort(ss.internals['household']['c'], axis=None) # count position of each array element, ascending order
# cons = np.reshape(ss.internals['household']['c'], (ntype, )) # flatten array
# cons_mass = np.multiply(cons[asset_pos], mass[asset_pos]) / ss['C']# multiply each consumption type by its respective density
# # total_cons = np.sum(cons_mass, axis=None)
# # lorenz_cons = trapz(np.cumsum(cons_mass / total_cons), dx=1/np.size(ss.internals['household']['a'])) # area below Lorenz curve
# # gini_cons = (0.5 - lorenz_cons) / 0.5
# # print("Consumption Gini =", np.round(gini_cons, 3))

# # Labor supply distribution
# labor_pos = np.argsort(ss.internals['household']['n'], axis=None) # count position of each array element, ascending order
# labor = np.reshape(ss.internals['household']['n'], (ntype, )) # flatten array
# labor_mass = np.multiply(labor[asset_pos], mass[asset_pos]) / ss['N']  # multiply each consumption type by its respective density


# =============================================================================
# Impulse response functions
# =============================================================================

rhos = 0.9
drstar = -0.02 * rhos ** (np.arange(T)[:, np.newaxis])
dtstar = 0.03 * rhos ** (np.arange(T)[:, np.newaxis])
dtauc = - 0.03 * rhos ** (np.arange(T)[:, np.newaxis])

dY = [G['Y']['rstar'] @ drstar, G['Y']['tauc'] @ dtauc, G['Y']['Transfer'] @ dtstar]
dC = [G['C']['rstar'] @ drstar, G['C']['tauc'] @ dtauc, G['C']['Transfer'] @ dtstar]
dN = [G['N']['rstar'] @ drstar, G['N']['tauc'] @ dtauc, G['N']['Transfer'] @ dtstar]
dB = [G['A']['rstar'] @ drstar, G['A']['tauc'] @ dtauc, G['A']['Transfer'] @ dtstar]
dw = [G['w']['rstar'] @ drstar, G['w']['tauc'] @ dtauc, G['w']['Transfer'] @ dtstar]
dP = [G['pi']['rstar'] @ drstar, G['pi']['tauc'] @ dtauc, G['pi']['Transfer'] @ dtstar]
dp = [G['cpi']['rstar'] @ drstar, G['cpi']['tauc'] @ dtauc, G['cpi']['Transfer'] @ dtstar]
dr = [G['r']['rstar'] @ drstar, G['r']['tauc'] @ dtauc, G['r']['Transfer'] @ dtstar]
dD = [G['Deficit']['rstar'] @ drstar, G['Deficit']['tauc'] @ dtauc, G['Deficit']['Transfer'] @ dtstar]
dd = [G['Div']['rstar'] @ drstar, G['Div']['tauc'] @ dtauc, G['Div']['Transfer'] @ dtstar]
dT = [np.zeros(T), np.zeros(T), G['Trans']['Transfer'] @ dtstar]
di = [G['i']['rstar'] @ drstar, np.zeros(T), np.zeros(T)]

plt.rcParams["figure.figsize"] = (16,7)
fig, ax = plt.subplots(2, 4)
fig.suptitle('Consumption tax cut versus transfer increase', size=16)

ax[0, 0].set_title(r'Output $Y$')
#l1, = ax[0, 0].plot(dY[0][:50, :])
l1, = ax[0, 0].plot(dY[1][:50, :] * ss['Y'])
l2, = ax[0, 0].plot(dY[2][:50, :] * ss['Y'],'-.')

ax[0, 1].set_title(r'Consumption $C$')
#ax[0, 1].plot(dC[0][:50, :])
ax[0, 1].plot(dC[1][:50, :] * ss['C'])
ax[0, 1].plot(dC[2][:50, :] * ss['C'],'-.')

ax[0, 2].set_title(r'Government debt $B$')
#ax[0, 2].plot(dB[0][:50, :])
ax[0, 2].plot(dB[1][:50, :] * ss['A'])
ax[0, 2].plot(dB[2][:50, :] * ss['A'],'-.')

ax[0, 3].set_title(r'Transfer $\tau$')
ax[0, 3].plot(np.zeros(50))
ax[0, 3].plot(dT[2][:50, :] * ss['Trans'],'-.')

ax[1, 0].set_title(r'Wage $w$')
#ax[1, 0].plot(dw[0][:50, :])
ax[1, 0].plot(dw[1][:50, :] * ss['w'])
ax[1, 0].plot(dw[2][:50, :] * ss['w'],'-.')

ax[1, 1].set_title(r' Inflation $\pi$')
#ax[1, 1].plot(dP[0][:50, :])
ax[1, 1].plot(dP[1][:50, :])
ax[1, 1].plot(dP[2][:50, :],'-.')

#ax[1, 2].set_title(r'Real interest rate $r$')
#ax[1, 2].plot(dr[0][:50, :])
#ax[1, 2].plot(dr[1][:50, :])
#ax[1, 2].plot(dr[2][:50, :],'-.')

ax[1, 2].set_title(r'Government budget deficit')
#ax[1, 2].plot(dD[0][:50, :])
ax[1, 2].plot(-dD[1][:50, :] * ss['Deficit'])
ax[1, 2].plot(-dD[2][:50, :] * ss['Deficit'],'-.')

ax[1, 3].set_title(r'Consumption tax $\tau_c$')
ax[1, 3].plot(dtauc[:50] * ss['tauc'])
ax[1, 3].plot(np.zeros(50),'-.')
plt.show()


# =============================================================================
# Steady-state and dynamic properties
# =============================================================================

# Discounted cumulative sum
cumtau, cumY, cumC, cumP, cumW, cumD = np.zeros(T),np.zeros(T),np.zeros(T),np.zeros(T),np.zeros(T),np.zeros(T)
discount = (1 / (1 + ss['r']))
#discount = ss['beta']
for i in range(T):
    cumtau[i] = discount ** i * (dtauc[i] + dT[2][i, :])
    cumY[i] = discount ** i * (dY[1][i, :] - dY[2][i, :])
    cumC[i] = discount ** i * (dC[1][i, :] - dC[2][i, :])
    cumP[i] = discount ** i * (dP[1][i, :] - dP[2][i, :])
    cumW[i] = discount ** i * (dw[1][i, :] - dw[2][i, :])
    cumD[i] = discount ** i * (dD[1][i, :] - dD[2][i, :])

# Show difference between consumption tax and transfers
dif = [['DIFFERENCE \u03C4c vs \u03C4','IMPACT RATIO','CUMULATIVE SUM'],
      ['Shocks',np.ndarray.item(- (dtauc[:1] * ss['tauc']) / (dT[2][:1, :] * ss['Trans'])),np.sum(cumtau)],
      ['Output',np.ndarray.item(dY[1][:1, :] / dY[2][:1, :]), np.sum(cumY) * ss['Y']],
      ['Consumption',np.ndarray.item(dC[1][:1, :] / dC[2][:1, :]), np.sum(cumC)],
      ['Inflation',np.ndarray.item(dP[1][:1, :] / dP[2][:1, :]), np.sum(cumP)],
      ['Wage',np.ndarray.item(dw[1][:1, :] / dw[2][:1, :]), np.sum(cumW)],
      ['Deficit',np.ndarray.item(dD[1][:1, :] / dD[2][:1, :]), np.sum(cumD)]]
dash = '-' * 50
for i in range(len(dif)):
    if i == 0:
        print(dash)
        print('{:<20s} {:^12s}  {:>15s}'.format(dif[i][0],dif[i][1],dif[i][2]))
        print(dash)
    else:
        print('{:<20s} {:^12.3f}  {:>15.3f}'.format(dif[i][0],dif[i][1],dif[i][2]))


# =============================================================================
# Zero net present value shocks
# =============================================================================

shock = np.zeros(T)
discount = (1 / (1 + ss['r']))
A, B, C, D, E = 1, 0.5, 0.19499, 5, 3
for x in range(T):
    shock[x] = discount ** x * (A - B * (x - E)) * np.exp(-C * (x - E) - D) 

# Integral
#intshock = 1 / (C ** 2) * (np.exp(C * (E - T) - D) * (np.exp(50 * C) * (A * C + B * (C * E - 1)) 
#                                                       - A * C + B * (- C) * (E - 50) + B))
intshock = (1 / (C-np.log(discount))** 2 * np.exp(C*E-D) * 
            (discount**T * np.exp(-60*C) * (np.log(discount) * (A+B*(E-T)) - A*C+B*(-C)*(E-T)+B) 
             - np.log(discount) * (A+B*E) + A*C + B*C*E - B))

dY = [G['Y']['tauc'] @ (-shock), G['Y']['Transfer'] @ shock]
dC = [G['C']['tauc'] @ (-shock), G['C']['Transfer'] @ shock]
dN = [G['N']['tauc'] @ (-shock), G['N']['Transfer'] @ shock]
dB = [G['A']['tauc'] @ (-shock), G['A']['Transfer'] @ shock]
dw = [G['w']['tauc'] @ (-shock), G['w']['Transfer'] @ shock]
dP = [G['pi']['tauc'] @ (-shock), G['pi']['Transfer'] @ shock]
dp = [G['cpi']['tauc'] @ (-shock), G['cpi']['Transfer'] @ shock]
dr = [G['r']['tauc'] @ (-shock), G['r']['Transfer'] @ shock]
dD = [G['Deficit']['tauc'] @ (-shock), G['Deficit']['Transfer'] @ shock]
dd = [G['Div']['tauc'] @ (-shock), G['Div']['Transfer'] @ shock]
dT = [np.zeros(T), G['Trans']['Transfer'] @ shock]
di = [np.zeros(T), np.zeros(T)]

plt.rcParams["figure.figsize"] = (16,7)
fig, ax = plt.subplots(2, 4)
fig.suptitle('Consumption tax cut versus transfer increase, zero net present value policies', size=16)

ax[0, 0].set_title(r'Output $Y$')
l1, = ax[0, 0].plot(dY[0][:50] * ss['Y'])
l2, = ax[0, 0].plot(dY[1][:50] * ss['Y'],'-.')

ax[0, 1].set_title(r'Consumption $C$')
ax[0, 1].plot(dC[0][:50] * ss['C'])
ax[0, 1].plot(dC[1][:50] * ss['C'],'-.')

ax[0, 2].set_title(r'Government debt $B$')
ax[0, 2].plot(dB[0][:50] * ss['A'])
ax[0, 2].plot(dB[1][:50] * ss['A'],'-.')

ax[0, 3].set_title(r'Transfer $\tau$')
ax[0, 3].plot(np.zeros(50))
ax[0, 3].plot(dT[1][:50] * ss['Trans'],'-.')

ax[1, 0].set_title(r'Wage $w$')
ax[1, 0].plot(dw[0][:50] * ss['w'])
ax[1, 0].plot(dw[1][:50] * ss['w'],'-.')

ax[1, 1].set_title(r' Inflation $\pi$')
ax[1, 1].plot(dP[0][:50])
ax[1, 1].plot(dP[1][:50],'-.')

ax[1, 2].set_title(r'Government budget deficit')
ax[1, 2].plot(-dD[0][:50] * ss['Deficit'])
ax[1, 2].plot(-dD[1][:50] * ss['Deficit'],'-.')

ax[1, 3].set_title(r'Consumption tax $\tau_c$')
ax[1, 3].plot(- dT[1][:50] * ss['tauc'])
ax[1, 3].plot(np.zeros(50),'-.')
plt.show()

#print(intshock)
##print(np.sum(shock))
#plt.plot(-shock[0:50])
#plt.plot([0, 50], [0, 0], 'k-', lw=.5)
#plt.show()

# Discounted cumulative sum
cumtau, cumY, cumC, cumP, cumW, cumD = np.zeros(T),np.zeros(T),np.zeros(T),np.zeros(T),np.zeros(T),np.zeros(T)
discount = (1 / (1 + ss['r']))
#discount = ss['beta']
for i in range(T):
    cumtau[i] = discount ** i * (-dT[1][i] + dT[1][i])
    cumY[i] = discount ** i * (dY[0][i] - dY[1][i])
    cumC[i] = discount ** i * (dC[0][i] - dC[1][i])
    cumP[i] = discount ** i * (dP[0][i] - dP[1][i])
    cumW[i] = discount ** i * (dw[0][i] - dw[1][i])
    cumD[i] = discount ** i * (dD[0][i] - dD[1][i])

# Show difference between consumption tax and transfers
dif = [['DIFFERENCE \u03C4c vs \u03C4','IMPACT RATIO','CUMULATIVE SUM'],
      ['Shocks',np.ndarray.item((dT[1][:1] * ss['tauc']) / (dT[1][:1] * ss['Trans'])),np.sum(cumtau)],
      ['Output',np.ndarray.item(dY[0][:1] / dY[1][:1]), np.sum(cumY) * ss['Y']],
      ['Consumption',np.ndarray.item(dC[0][:1] / dC[1][:1]), np.sum(cumC)],
      ['Inflation',np.ndarray.item(dP[0][:1] / dP[1][:1]), np.sum(cumP)],
      ['Wage',np.ndarray.item(dw[0][:1] / dw[1][:1]), np.sum(cumW)],
      ['Deficit',np.ndarray.item(dD[0][:1] / dD[1][:1]), np.sum(cumD)]]
dash = '-' * 50
for i in range(len(dif)):
    if i == 0:
        print(dash)
        print('{:<20s} {:^12s}  {:>15s}'.format(dif[i][0],dif[i][1],dif[i][2]))
        print(dash)
    else:
        print('{:<20s} {:^12.3f}  {:>15.3f}'.format(dif[i][0],dif[i][1],dif[i][2]))
        
print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))   