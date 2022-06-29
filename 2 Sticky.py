"""Sticky-wage HANK model: tax vs transfer vs interest rate policy"""

# =============================================================================
# Models
# =============================================================================

print("STICKY-WAGE HANK: TAXES vs TRANSFERS vs INTEREST RATE CUTS")
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
def fiscal1(B, C, N, r, Tau, tauc, taun, w): # for model 1: transfer/tax policy
    # gov = Tau + (1 + r) * B(-1) - tauc * C - taun * w * N - B # government BC
    gov = B - Tau - (1 + r) * B(-1) + tauc * C + taun * w * N # government BC 
    Deficit = Tau - tauc * C - taun * w * N # primary deficit
    Trans = Tau
    return gov, Deficit, Trans

@simple
def fiscal2(B, B_ss, C, N, r, r_ss, rhot, Tau, tauc, taun, w): # for model 2: interest rate policy
    # Tau = taun * w * N + B - (1 + r) * B(-1) # immediate adjustment of transfers, no tauc
    gov = B - Tau - (1 + r) * B(-1) + tauc * C + taun * w * N # government BC
    # gov = Tau - rhot * Tau(-1) - (1 - rhot) * (taun * w * N + B - (1 + r) * B(-1)) # delayed adjustment of transfers
    Deficit = Tau - taun * w * N + (1 + r) * B(-1) - B # primary deficit, no tauc
    fiscal_rule = (B - B_ss) - (B(-1) - B_ss) - rhot * (r - r_ss)  # delayed adjustment of transfers
    Trans = Tau
    return Deficit, Trans, gov, fiscal_rule

@simple
def fiscal3(B, B_ss, C, N, r, rhot, sigma, Tau, Tau_ss, tauc, taun, w, Y): # for model 2: interest rate policy, variable debt
    gov = B - Tau - (1 + r) * B(-1) + tauc * C + taun * w * N # government BC 
    fiscal_rule = Tau - rhot * Tau(-1) - (1 - rhot) * Tau_ss + (1 - rhot) * sigma * (B / Y - B_ss / 1) # delayed adjustment of transfers
    Deficit = tauc * C + taun * w * N - Tau # primary surplus
    Trans = Tau
    return gov, fiscal_rule, Deficit, Trans

@simple
def fiscal3_ss(B, B_ss, C, N, r, tauc, taun, w): # for model 2: interest rate policy, variable debt
    Tau_ss = tauc * C + taun * w * N - r * B # government BC in steady state
    debt = B - B_ss
    return Tau_ss, debt

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
def nkpc(kappa, mu, pi, r, w, Y, Z):
    nkpc = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc

@simple
def union(beta, kappaw, muw, nu, phi, piw, N, UCE, tauc, taun, w):
    wnkpc = (kappaw * (phi * N ** (1+nu) - (1 - taun) * w * N * UCE / ((1 + tauc) * muw)) 
             + beta * (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log))
    return wnkpc

@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw


# =============================================================================
# Household iteration policy rule
# =============================================================================

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


# =============================================================================
# Common calibration
# =============================================================================

# Grid parameters
nE = 8;
sd_e = 0.5
rho_e = 0.966
nA = 50
amin = 0
amax = 150

# Steady-state variables
B_ss = 6.0
N_ss = 1.0
pi_ss = 0.0
rstar = 0.01
Y_ss = 1.0
Z_ss = 1.0

# Structural parameters
gamma = 1.0
kappa = 0.06
kappaw = 0.06
mu = 1.2
muw = 1.2
nu = 2.0
rhot = 0.9
sigma = 0.09
tauc = 0.1
taun = 0.072


# =============================================================================
# Model 1: Tax or transfer policy
# =============================================================================

print("\nMODEL 1: TAX OR TRANSFER POLICY")

# Steady state
blocks_ss_tau = [hh_inp, firm, monetary, fiscal1, mkt_clearing, nkpc_ss, union_ss]
hank_ss_tau = create_model(blocks_ss_tau)
calib_tau = {'nE': nE, 'sd_e': sd_e, 'rho_e': rho_e, 'nA': nA, 'amin': amin, 'amax': amax,
             'B': B_ss, 'N': N_ss, 'pi': pi_ss, 'rstar': rstar, 'Y': Y_ss, 'Z': Z_ss,
             'gamma': gamma, 'kappa': kappa, 'kappaw': kappaw, 'mu': mu, 'muw': muw, 
             'nu': nu, 'phi_pi': 0.0, 'tauc': tauc, 'taun': taun}
             
unknowns_ss_tau = {'beta': 0.986, 'Tau': 0.02}
targets_ss_tau = {'asset_mkt': 0, 'gov': 0}
print("Computing steady state...", end=" ")
ss0_tau = hank_ss_tau.solve_steady_state(calib_tau, unknowns_ss_tau, targets_ss_tau, backward_tol=1E-22, solver="hybr")
print("Done")

# Dynamics
blocks_tau = [hh_inp, firm, monetary, fiscal1, mkt_clearing, nkpc, wage, union]
hank_tau = create_model(blocks_tau)
ss_tau = hank_tau.steady_state(ss0_tau)
T = 300
exogenous_tau = ['rstar','Tau', 'tauc', 'Z']
unknowns_tau = ['pi', 'w', 'Y', 'B']
targets_tau = ['nkpc', 'asset_mkt', 'wnkpc', 'gov']
print("Computing Jacobian...", end=" ")
G_tau = hank_tau.solve_jacobian(ss_tau, unknowns_tau, targets_tau, exogenous_tau, T=T)
print("Done")


# =============================================================================
# Model 2: Interest rate policy 
# =============================================================================

print("MODEL 2: INTEREST RATE POLICY")

# Steady state 
blocks_ss_rstar = [hh_inp, firm, monetary, fiscal3, fiscal3_ss, mkt_clearing, nkpc_ss, union_ss]
hank_ss_rstar = create_model(blocks_ss_rstar)
calib_rstar = {'nE': nE, 'sd_e': sd_e, 'rho_e': rho_e, 'nA': nA, 'amin': amin, 'amax': amax,
             'B': B_ss, 'N': N_ss, 'pi': pi_ss, 'rstar': rstar, 'Y': Y_ss, 'Z': Z_ss,
             'gamma': gamma, 'kappa': kappa, 'kappaw': kappaw, 'mu': mu, 'muw': muw, 
             'nu': nu, 'phi_pi': 1.5, 'tauc': tauc, 'taun': taun, 'rhot': rhot, 'sigma': sigma}

# unknowns_ss_rstar = {'beta': 0.986, 'Tau': -0.03}
# targets_ss_rstar = {'asset_mkt': 0, 'gov': 0}
unknowns_ss_rstar = {'beta': 0.986, 'Tau': 0.02, 'B_ss': 6.0}
targets_ss_rstar = {'asset_mkt': 0, 'gov': 0, 'fiscal_rule': 0}
print("Computing steady state...", end=" ")
ss0_rstar = hank_ss_rstar.solve_steady_state(calib_rstar, unknowns_ss_rstar, targets_ss_rstar, backward_tol=1E-22, solver="hybr")
print("Done")

# Dynamics
blocks_rstar = [hh_inp, firm, monetary, fiscal3, mkt_clearing, nkpc, wage, union]
hank_rstar = create_model(blocks_rstar)
ss_rstar = hank_rstar.steady_state(ss0_rstar)
T = 300
exogenous_rstar = ['rstar', 'Z']
# unknowns_rstar = ['pi', 'w', 'Y', 'Tau']
# targets_rstar = ['nkpc', 'asset_mkt', 'wnkpc', 'gov']
unknowns_rstar = ['pi', 'w', 'Y', 'B', 'Tau']
targets_rstar = ['asset_mkt', 'gov', 'fiscal_rule', 'nkpc', 'wnkpc']
print("Computing Jacobian...", end=" ")
G_rstar = hank_rstar.solve_jacobian(ss_rstar, unknowns_rstar, targets_rstar, exogenous_rstar, T=T)
print("Done")


# =============================================================================
# Steady-state properties
# =============================================================================
    
ss_param = [['Discount factor', ss_tau['beta'], 'Intertemporal elasticity', gamma],
            ['Labor supply elasticity', 1 / nu, 'Labor supply disutility', ss_tau['phi']],  
            ['Goods substitutability', mu / (mu - 1) , 'Price markup', mu],
            ['Labor substitutability', muw / (muw - 1) , 'Wage markup', muw],
            ['Price Phillips slope', kappa, 'Taylor rule inflation ', ss_tau['phi_pi']],
            ['Wage Phillips slope', kappaw, 'Taylor rule output ', 0],
            ['Consumption tax rate', tauc, 'Labor tax rate', taun]]

ss_var_tau = [['Output', ss_tau['Y'], 'Government debt', ss_tau['A']],
              ['Consumption', ss_tau['C'], 'Transfers', ss_tau['Tau']],
              ['Hours', ss_tau['N'], 'Dividends', ss_tau['Div']], 
              ['Wage', ss_tau['w'], 'Marginal cost', ss_tau['w'] / ss_tau['Z']],
              ['Inflation', ss_tau['pi'], 'Consumption tax revenue', ss_tau['tauc'] * ss_tau['C']],
              ['Nominal interest rate', ss_tau['r']*(1+ss_tau['pi']), 'Labor tax revenue', ss_tau['taun'] * ss_tau['N'] * ss_tau['w']],
              ['Real interest rate', ss_tau['r'], 'Debt servicing  cost', ss_tau['r'] * ss_tau['A']]]

ss_var_rstar = [['Output', ss_rstar['Y'], 'Government debt', ss_rstar['A']],
                ['Consumption', ss_rstar['C'], 'Transfers', ss_rstar['Tau']],
                ['Hours', ss_rstar['N'], 'Dividends', ss_rstar['Div']], 
                ['Wage', ss_rstar['w'], 'Marginal cost', ss_rstar['w'] / ss_rstar['Z']],
                ['Inflation', ss_rstar['pi'], 'Consumption tax revenue', ss_rstar['tauc'] * ss_rstar['C']],
                ['Nominal interest rate', ss_rstar['r']*(1+ss_rstar['pi']), 'Labor tax revenue', ss_rstar['taun'] * ss_rstar['N'] * ss_rstar['w']],
                ['Real interest rate', ss_rstar['r'], 'Debt servicing  cost', ss_rstar['r'] * ss_rstar['A']]]

ss_mkt = [['Bond market', ss_tau['asset_mkt'], 'Goods market (resid)', ss_tau['goods_mkt']],
          ['Government budget', ss_tau['gov'], '', float('nan')]]

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

# Steady-state variables
B_ss_tau = ss_tau['B']
C_ss_tau = ss_tau['C']
D_ss_tau = ss_tau['Deficit']
Div_ss_tau = ss_tau['Div']
N_ss_tau = ss_tau['N']
r_ss_tau = ss_tau['r']
T_ss_tau = ss_tau['Trans']
Tau_ss_tau = ss_tau['Tau']
w_ss_tau = ss_tau['w']
Y_ss_tau = ss_tau['Y']

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


# =============================================================================
# IRF: Transfer vs monetary policy
# =============================================================================

# # Standard shock
# discount = (1 / (1 + rstar))
# #discount = ss_tau['beta']
# rhos = 0.7
# dtau = 0.02 * rhos ** np.arange(T)
# drstar = -0.00285 * rhos ** np.arange(T)

# # Zero net present value sock
# shock = np.zeros(T)
# # s1, s2, s3, s4, s5 = 1, 0.5, 0.1723464735, 5, 3 # rstar = 0.005
# s1, s2, s3, s4, s5 = 1, 0.5, 0.162420896, 5, 3 # rstar = 0.01
# for x in range(T):
#     shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
# dtau = shock

# # # Plot shock
# # cumshock = np.zeros(T)
# # for i in range(T):
# #     cumshock[i] = discount ** i * dtauc[i] # discounted cumulative sum
# # plt.plot(shock[0:40], linewidth=2)
# # plt.plot([0, 40], [0, 0], '--', color='gray', linewidth=0.5)
# # plt.title("Zero net present value shock: " + str(round(np.sum(cumshock), 10)))
# # plt.margins(x=0, y=0)
# # plt.show()

# # IRFs
# dY = [G_tau['Y']['Tau'] @ dtau, G_rstar['Y']['rstar'] @ drstar]
# dC = [G_tau['C']['Tau'] @ dtau, G_rstar['C']['rstar'] @ drstar]
# dN = [G_tau['N']['Tau'] @ dtau, G_rstar['N']['rstar'] @ drstar]
# dB = [G_tau['B']['Tau'] @ dtau, G_rstar['A']['rstar'] @ drstar]
# dw = [G_tau['w']['Tau'] @ dtau, G_rstar['w']['rstar'] @ drstar]
# dp = [G_tau['pi']['Tau'] @ dtau, G_rstar['pi']['rstar'] @ drstar]
# dr = [G_tau['r']['Tau'] @ dtau, G_rstar['r']['rstar'] @ drstar]
# dD = [G_tau['Deficit']['Tau'] @ dtau, G_rstar['Deficit']['rstar'] @ drstar]
# dd = [G_tau['Div']['Tau'] @ dtau, G_rstar['Div']['rstar'] @ drstar]
# dT = [G_tau['Trans']['Tau'] @ dtau, G_rstar['Trans']['rstar'] @ drstar]
# di = [np.zeros(T), G_rstar['i']['rstar'] @ drstar]

# plt.rcParams["figure.figsize"] = (20,7)
# fig, ax = plt.subplots(2, 4)
# fig.suptitle('Transfer vs monetary policy, HANK with sticky wages', size=16)
# iT = 30

# ax[0, 0].set_title(r'Output $Y$')
# ax[0, 0].plot(100 * dY[0][:iT] / Y_ss_tau, label="Transfer policy")
# ax[0, 0].plot(100 * dY[1][:iT] / Y_ss_rstar,'-.', label="Monetary policy")
# ax[0, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
# ax[0, 0].legend(loc='upper right', frameon=False)

# ax[0, 1].set_title(r'Consumption $C$')
# ax[0, 1].plot(100 * dC[0][:iT] / C_ss_tau)
# ax[0, 1].plot(100 * dC[1][:iT] / C_ss_rstar,'-.')
# ax[0, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[0, 2].set_title(r'Government debt $B$')
# ax[0, 2].plot(100 * dB[0][:iT] / B_ss_tau)
# ax[0, 2].plot(100 * dB[1][:iT] / B_ss_rstar,'-.')
# ax[0, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[0, 3].set_title(r'Transfer $\tau$')
# ax[0, 3].plot(100 * dT[0][:iT])
# ax[0, 3].plot(100 * dT[1][:iT],'-.')
# ax[0, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 0].set_title(r'Wage $w$')
# ax[1, 0].plot(100 * dw[0][:iT] / w_ss_tau)
# ax[1, 0].plot(100 * dw[1][:iT] / w_ss_rstar,'-.')
# ax[1, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 1].set_title(r'Inflation $\pi$')
# ax[1, 1].plot(100 * dp[0][:iT])
# ax[1, 1].plot(100 * dp[1][:iT],'-.')
# ax[1, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 2].set_title(r'Nominal interest rate $i$')
# ax[1, 2].plot(100 * di[0][:iT])
# ax[1, 2].plot(100 * di[1][:iT],'-.')
# ax[1, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 3].set_title(r'Budget deficit $D$')
# ax[1, 3].plot(100 * dD[0][:iT])
# ax[1, 3].plot(100 * dD[1][:iT],'-.')
# ax[1, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
# plt.show()

# # Discounted cumulative sum
# cumtau, cumY, cumC, cumP, cumW, cumB, cumT = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
# dcumtau, dcumY, dcumC, dcumP, dcumW, dcumB, dcumT = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
# for i in range(T):
#     cumtau[i] = discount ** i * dT[0][i]
#     cumC[i] = discount ** i * dC[0][i]
#     cumY[i] = discount ** i * dY[0][i]
#     cumP[i] = discount ** i * dp[0][i]
#     cumW[i] = discount ** i * dw[0][i]
#     cumT[i] = discount ** i * dT[0][i]
#     cumB[i] = discount ** i * dB[0][i]
#     dcumtau[i] = discount ** i * (dT[0][i] + dT[1][i])
#     dcumY[i] = discount ** i * (dY[0][i] - dY[1][i])
#     dcumC[i] = discount ** i * (dC[0][i] - dC[1][i])
#     dcumP[i] = discount ** i * (dp[0][i] - dp[1][i])
#     dcumW[i] = discount ** i * (dw[0][i] - dw[1][i])
#     dcumT[i] = discount ** i * (dT[0][i] - dT[1][i])
#     dcumB[i] = discount ** i * (dB[0][i] - dB[1][i])

# # Difference tau vs rstar
# dif = [['\nTRANSFER v INTEREST RATE', 'CUM SUM \u03C4','CUM SUM \u03C4-i*','IMPACT \u03C4/i*'],
#       ['Shocks', 100 * np.sum(cumtau), 100 * np.sum(dcumtau), -dtau[0] / di[1][0]],
#       ['Output', 100 * np.sum(cumY), 100 * np.sum(dcumY), dY[0][0] / dY[1][0]],
#       ['Consumption', 100 * np.sum(cumC), 100 * np.sum(dcumC), dC[0][0] / dC[1][0]],
#       ['Inflation', 100 * np.sum(cumP), 100 * np.sum(dcumP), dp[0][0] / dp[1][0]],
#       ['Wage', 100 * np.sum(cumW), 100 * np.sum(dcumW), dw[0][0] / dw[1][0]],
#       ['Debt', 100 * np.sum(cumB), 100 * np.sum(dcumB), dB[0][0] / dB[1][0]],
#       ['Transfer', 100 * np.sum(cumT), 100 * np.sum(dcumT), dT[0][0] / dT[1][0]]]
# for i in range(len(dif)):
#     if i == 0:
#         print('{:<27} {:^14s} {:^15s} {:^15s}'.format(dif[i][0], dif[i][1], dif[i][2], dif[i][3]))
#     else:
#         print('{:<20s} {:>14.3f} {:s} {:>14.3f} {:s} {:>14.3f}'.format(dif[i][0], dif[i][1], "%", dif[i][2], "%", dif[i][3]))


# =============================================================================
# IRF: Consumption tax vs monetary policy
# =============================================================================

# # Standard shock
# discount = (1 / (1 + rstar))
# #discount = ss_tau['beta']
# rhos = 0.67
# dtauc = - 0.02 * rhos ** np.arange(T)
# drstar = -0.00585 * rhos ** np.arange(T)

# # Zero net present value sock
# shock = np.zeros(T)
# # s1, s2, s3, s4, s5 = 1, 0.5, 0.1723464735, 5, 3 # rstar = 0.005
# s1, s2, s3, s4, s5 = 1, 0.5, 0.162420896, 5, 3 # rstar = 0.01
# for x in range(T):
#     shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
# dtau = shock
# dtauc = - dtau

# # # Plot shock
# # cumshock = np.zeros(T)
# # for i in range(T):
# #     cumshock[i] = discount ** i * dtauc[i] # discounted value
# # plt.plot(shock[0:40], linewidth=2)
# # plt.plot([0, 40], [0, 0], '--', color='gray', linewidth=0.5)
# # plt.title("Zero net present value shock: " + str(round(np.sum(cumshock), 10)))
# # plt.margins(x=0, y=0)
# # plt.show()

# # IRFs
# dYc = [G_tau['Y']['tauc'] @ dtauc, G_rstar['Y']['rstar'] @ drstar]
# dCc = [G_tau['C']['tauc'] @ dtauc, G_rstar['C']['rstar'] @ drstar]
# dNc = [G_tau['N']['tauc'] @ dtauc, G_rstar['N']['rstar'] @ drstar]
# dBc = [G_tau['B']['tauc'] @ dtauc, G_rstar['A']['rstar'] @ drstar]
# dwc = [G_tau['w']['tauc'] @ dtauc, G_rstar['w']['rstar'] @ drstar]
# dpc = [G_tau['pi']['tauc'] @ dtauc, G_rstar['pi']['rstar'] @ drstar]
# drc = [G_tau['r']['tauc'] @ dtauc, G_rstar['r']['rstar'] @ drstar]
# dDc = [G_tau['Deficit']['tauc'] @ dtauc, G_rstar['Deficit']['rstar'] @ drstar]
# ddc = [G_tau['Div']['tauc'] @ dtauc, G_rstar['Div']['rstar'] @ drstar]
# dTc = [np.zeros(T), G_rstar['Trans']['rstar'] @ drstar]
# dTcc = [dtauc, np.zeros(T)]
# dic = [np.zeros(T), G_rstar['i']['rstar'] @ drstar]

# plt.rcParams["figure.figsize"] = (20,7)
# fig, ax = plt.subplots(2, 4)
# fig.suptitle('Consumption tax vs monetary policy, HANK with sticky wages', size=16)
# iT = 30

# ax[0, 0].set_title(r'Output $Y$')
# ax[0, 0].plot(100 * dYc[0][:iT] / Y_ss_tau, label="Consumption tax policy")
# ax[0, 0].plot(100 * dYc[1][:iT] / Y_ss_rstar,'-.', label="Monetary policy")
# ax[0, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
# ax[0, 0].legend(loc='upper right', frameon=False)

# ax[0, 1].set_title(r'Consumption $C$')
# ax[0, 1].plot(100 * dCc[0][:iT] / C_ss_tau)
# ax[0, 1].plot(100 * dCc[1][:iT] / C_ss_rstar,'-.')
# ax[0, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[0, 2].set_title(r'Government debt $B$')
# ax[0, 2].plot(100 * dBc[0][:iT] / B_ss_tau)
# ax[0, 2].plot(100 * dBc[1][:iT] / B_ss_rstar,'-.')
# ax[0, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[0, 3].set_title(r'Transfer $\tau$')
# ax[0, 3].plot(100 * dTc[0][:iT])
# ax[0, 3].plot(100 * dTc[1][:iT],'-.')
# ax[0, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 0].set_title(r'Wage $w$')
# ax[1, 0].plot(100 * dwc[0][:iT] / w_ss_tau)
# ax[1, 0].plot(100 * dwc[1][:iT] / w_ss_rstar,'-.')
# ax[1, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 1].set_title(r'Inflation $\pi$')
# ax[1, 1].plot(100 * dpc[0][:iT])
# ax[1, 1].plot(100 * dpc[1][:iT],'-.')
# ax[1, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# # ax[1, 2].set_title(r'Dividends $d$')
# # ax[1, 2].plot(ddc[0][:iT])
# # ax[1, 2].plot(ddc[1][:iT])
# # ax[1, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 2].set_title(r'Nominal interest rate $i$')
# ax[1, 2].plot(100 * dic[0][:iT])
# ax[1, 2].plot(100 * dic[1][:iT],'-.')
# ax[1, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 3].set_title(r'Consumption tax $\tau_c$')
# ax[1, 3].plot(100 * dTcc[0][:iT])
# ax[1, 3].plot(100 * dTcc[1][:iT],'-.')
# ax[1, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
# plt.show()

# # Discounted cumulative sum
# cumtau, cumY, cumC, cumP, cumW, cumB, cumT = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
# dcumtau, dcumY, dcumC, dcumP, dcumW, dcumB, dcumT = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
# for i in range(T):
#     cumtau[i] = discount ** i * dTc[0][i]
#     cumC[i] = discount ** i * dCc[0][i]
#     cumY[i] = discount ** i * dYc[0][i]
#     cumP[i] = discount ** i * dpc[0][i]
#     cumW[i] = discount ** i * dwc[0][i]
#     cumB[i] = discount ** i * dBc[0][i]
#     cumT[i] = discount ** i * dTc[0][i]
#     dcumtau[i] = discount ** i * (dTc[0][i] + dTc[1][i])
#     dcumY[i] = discount ** i * (dYc[0][i] - dYc[1][i])
#     dcumC[i] = discount ** i * (dCc[0][i] - dCc[1][i])
#     dcumP[i] = discount ** i * (dpc[0][i] - dpc[1][i])
#     dcumW[i] = discount ** i * (dwc[0][i] - dwc[1][i])
#     dcumB[i] = discount ** i * (dBc[0][i] - dBc[1][i])
#     dcumT[i] = discount ** i * (dTc[0][i] - dTc[1][i])
    
# # Difference tauc vs rstar
# dif = [['\nTAX v INTEREST RATE', 'CUM SUM \u03C4c','CUM SUM \u03C4c-i*','IMPACT \u03C4c/i*'],
#       ['Shocks', 100 * np.sum(cumtau), 100 * np.sum(dcumtau), dtauc[0] / dic[1][0]],
#       ['Output', 100 * np.sum(cumY), 100 * np.sum(dcumY), dYc[0][0] / dYc[1][0]],
#       ['Consumption', 100 * np.sum(cumC), 100 * np.sum(dcumC), dCc[0][0] / dCc[1][0]],
#       ['Inflation', 100 * np.sum(cumP), 100 * np.sum(dcumP), dpc[0][0] / dpc[1][0]],
#       ['Wage', 100 * np.sum(cumW), 100 * np.sum(dcumW), dwc[0][0] / dwc[1][0]],
#       ['Debt', 100 * np.sum(cumB), 100 * np.sum(dcumB), dBc[0][0] / dBc[1][0]],
#       ['Transfer', 100 * np.sum(cumT), 100 * np.sum(dcumT), dTc[0][0] / dTc[1][0]]]
# for i in range(len(dif)):
#     if i == 0:
#         print('{:<27s} {:^15s} {:^15s} {:^15s}'.format(dif[i][0], dif[i][1], dif[i][2], dif[i][3]))
#     else:
#         print('{:<20s} {:>14.3f} {:s} {:>14.3f} {:s} {:>14.3f}'.format(dif[i][0], dif[i][1], "%", dif[i][2], "%", dif[i][3]))


# =============================================================================
# IRF: Consumption tax vs transfer
# =============================================================================

# Standard shock
discount = (1 / (1 + rstar))
rhos = 0.67
dtau = 0.02 * rhos ** np.arange(T)
dtauc = - dtau

# Zero net present value sock
shock = np.zeros(T)
s1, s2, s3, s4, s5 = 1, 0.5, 0.162420896, 5, 3 # rstar = 0.01
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
dtau = shock
dtauc = - dtau

# # Plot shock
# cumshock = np.zeros(T)
# for i in range(T):
#     cumshock[i] = discount ** i * dtauc[i] # discounted value
# plt.plot(shock[0:40], linewidth=2)
# plt.plot([0, 40], [0, 0], '--', color='gray', linewidth=0.5)
# plt.title("Zero net present value shock: " + str(round(np.sum(cumshock), 10)))
# plt.margins(x=0, y=0)
# plt.show()

# IRFs
dY = [G_tau['Y']['tauc'] @ dtauc, G_tau['Y']['Tau'] @ dtau]
dC = [G_tau['C']['tauc'] @ dtauc, G_tau['C']['Tau'] @ dtau]
dN = [G_tau['N']['tauc'] @ dtauc, G_tau['N']['Tau'] @ dtau]
dB = [G_tau['B']['tauc'] @ dtauc, G_tau['A']['Tau'] @ dtau]
dw = [G_tau['w']['tauc'] @ dtauc, G_tau['w']['Tau'] @ dtau]
dp = [G_tau['pi']['tauc'] @ dtauc, G_tau['pi']['Tau'] @ dtau]
dr = [G_tau['r']['tauc'] @ dtauc, G_tau['r']['Tau'] @ dtau]
dD = [G_tau['Deficit']['tauc'] @ dtauc, G_tau['Deficit']['Tau'] @ dtau]
dd = [G_tau['Div']['tauc'] @ dtauc, G_tau['Div']['Tau'] @ dtau]
dT = [np.zeros(T), G_tau['Trans']['Tau'] @ dtau]
dTc = [dtauc, np.zeros(T)]
di = [np.zeros(T), np.zeros(T)]

plt.rcParams["figure.figsize"] = (20,7)
fig, ax = plt.subplots(2, 4)
fig.suptitle('Consumption tax vs transfer, HANK with sticky wages', size=16)
iT = 30

ax[0, 0].set_title(r'Output $Y$')
ax[0, 0].plot(100 * dY[0][:iT] / Y_ss_tau, label="Consumption tax policy")
ax[0, 0].plot(100 * dY[1][:iT] / Y_ss_tau,'-.', label="Transfer policy")
ax[0, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
ax[0, 0].legend(loc='upper right', frameon=False)

ax[0, 1].set_title(r'Consumption $C$')
ax[0, 1].plot(100 * dC[0][:iT] / C_ss_tau)
ax[0, 1].plot(100 * dC[1][:iT] / C_ss_tau,'-.')
ax[0, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[0, 2].set_title(r'Government debt $B$')
ax[0, 2].plot(100 * dB[0][:iT] / B_ss_tau)
ax[0, 2].plot(100 * dB[1][:iT] / B_ss_tau,'-.')
ax[0, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[0, 3].set_title(r'Transfer $\tau$')
ax[0, 3].plot(100 * dT[0][:iT])
ax[0, 3].plot(100 * dT[1][:iT],'-.')
ax[0, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 0].set_title(r'Wage $w$')
ax[1, 0].plot(100 * dw[0][:iT] / w_ss_tau)
ax[1, 0].plot(100 * dw[1][:iT] / w_ss_tau,'-.')
ax[1, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 1].set_title(r'Inflation $\pi$')
ax[1, 1].plot(100 * dp[0][:iT])
ax[1, 1].plot(100 * dp[1][:iT],'-.')
ax[1, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 2].set_title(r'Nominal interest rate $i$')
ax[1, 2].plot(100 * di[0][:iT])
ax[1, 2].plot(100 * di[1][:iT],'-.')
ax[1, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 3].set_title(r'Consumption tax $\tau_c$')
ax[1, 3].plot(100 * dTc[0][:iT])
ax[1, 3].plot(100 * dTc[1][:iT],'-.')
ax[1, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
plt.show()

# Discounted cumulative sum
cumtau, cumY, cumC, cumP, cumW, cumB, cumT = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
dcumtau, dcumY, dcumC, dcumP, dcumW, dcumB, dcumT = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
for i in range(T):
    cumtau[i] = discount ** i * dT[0][i]
    cumC[i] = discount ** i * dC[0][i]
    cumY[i] = discount ** i * dY[0][i]
    cumP[i] = discount ** i * dp[0][i]
    cumW[i] = discount ** i * dw[0][i]
    cumB[i] = discount ** i * dB[0][i]
    cumT[i] = discount ** i * dT[0][i]
    dcumtau[i] = discount ** i * (dTc[0][i] + dT[1][i])
    dcumY[i] = discount ** i * (dY[0][i] - dY[1][i])
    dcumC[i] = discount ** i * (dC[0][i] - dC[1][i])
    dcumP[i] = discount ** i * (dp[0][i] - dp[1][i])
    dcumW[i] = discount ** i * (dw[0][i] - dw[1][i])
    dcumB[i] = discount ** i * (dB[0][i] - dB[1][i])
    dcumT[i] = discount ** i * (dT[0][i] - dT[1][i])
    
# Difference tauc vs rstar
dif = [['\nTAX v TRANSFER', 'CUM SUM \u03C4c','CUM SUM \u03C4c-\u03C4','IMPACT \u03C4c/\u03C4'],
      ['Shocks', 100 * np.sum(cumtau), 100 * np.sum(dcumtau), - dtauc[0] / dtau[0]],
      ['Output', 100 * np.sum(cumY), 100 * np.sum(dcumY), dY[0][0] / dY[1][0]],
      ['Consumption', 100 * np.sum(cumC), 100 * np.sum(dcumC), dC[0][0] / dC[1][0]],
      ['Inflation', 100 * np.sum(cumP), 100 * np.sum(dcumP), dp[0][0] / dp[1][0]],
      ['Wage', 100 * np.sum(cumW), 100 * np.sum(dcumW), dw[0][0] / dw[1][0]],
      ['Debt', 100 * np.sum(cumB), 100 * np.sum(dcumB), dB[0][0] / dB[1][0]],
      ['Transfer', 100 * np.sum(cumT), 100 * np.sum(dcumT), dT[0][0] / dT[1][0]]]
for i in range(len(dif)):
    if i == 0:
        print('{:<27s} {:^15s} {:^15s} {:^15s}'.format(dif[i][0], dif[i][1], dif[i][2], dif[i][3]))
    else:
        print('{:<20s} {:>14.3f} {:s} {:>14.3f} {:s} {:>14.3f}'.format(dif[i][0], dif[i][1], "%", dif[i][2], "%", dif[i][3]))
        

# =============================================================================
# Distribution: Transfer vs monetary policy
# =============================================================================

# # Model 1: Transfer policy
# print("\nMODEL 1: TRANSFER POLICY")

# # Parameters and steady-state variables
# beta = ss_tau['beta']
# a_grid_tau = ss_tau.internals['household']['a_grid']
# e_grid_tau = ss_tau.internals['household']['e_grid']
# c_ss_tau = ss_tau.internals['household']['c']
# Pi_tau = ss_tau.internals['household']['Pi']
# pi_e_tau = ss_tau.internals['household']['pi_e']
# # D_ss_tau = ss_tau.internals['household']['Dbeg']
# D_ss_tau = ss_tau.internals['household']['D']

# # Aggregate transition dynamics
# path_div_tau = Div_ss_tau + G_tau['Div']['Tau'] @ dtau
# path_n_tau = N_ss_tau + G_tau['N']['Tau'] @ dtau
# path_r_tau = r_ss_tau + G_tau['r']['Tau'] @ dtau
# path_tau_tau = Tau_ss_tau + dtau
# path_w_tau = w_ss_tau + G_tau['w']['Tau'] @ dtau

# # Compute all individual consumption paths
# print("Computing individual paths...", end=" ")
# V_prime_p_tau = (1 + r_ss_tau) / (1 + tauc) * c_ss_tau ** (-gamma)
# c_all_tau = np.zeros((nE, nA, T))
# for t in range(T-1, -1, -1):
#     V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
#                                         path_div_tau[t], path_n_tau[t], path_r_tau[t], path_tau_tau[t], tauc, taun, path_w_tau[t])
#     c_all_tau[:, :, t] = c  
# print("Done")

# # Direct effect of policy
# print("Computing direct effect...", end=" ")
# V_prime_p_tau = (1 + r_ss_tau) / (1 + tauc) * c_ss_tau ** (-gamma)
# c_direct_tau = np.zeros((nE, nA, T))
# for t in range(T-1, -1, -1):
#     V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
#                                         Div_ss_tau, N_ss_tau, r_ss_tau, path_tau_tau[t], tauc, taun, w_ss_tau)
#     # V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
#                                         # Div_ss_tau, N_ss_tau, r_ss_tau, Tau_ss_tau, tauc, taun, w_ss_tau)
#     c_direct_tau[:, :, t] = c
# print("Done")


# # Model 2: Interest rate policy
# print("MODEL 2: INTEREST RATE POLICY")

# # Parameters and steady-state variables
# beta = ss_rstar['beta']
# a_grid_rstar = ss_rstar.internals['household']['a_grid']
# e_grid_rstar = ss_rstar.internals['household']['e_grid']
# c_ss_rstar = ss_rstar.internals['household']['c']
# Pi_rstar = ss_rstar.internals['household']['Pi']
# pi_e_rstar = ss_rstar.internals['household']['pi_e']
# # D_ss_rstar = ss_rstar.internals['household']['Dbeg']
# D_ss_rstar = ss_rstar.internals['household']['D']

# # Aggregate transition dynamics
# path_div_rstar = Div_ss_rstar + G_rstar['Div']['rstar'] @ drstar
# path_n_rstar = N_ss_rstar + G_rstar['N']['rstar'] @ drstar
# path_r_rstar = r_ss_rstar + G_rstar['r']['rstar'] @ drstar
# path_tau_rstar = Tau_ss_rstar + G_rstar['Tau']['rstar'] @ drstar
# path_w_rstar = w_ss_rstar + G_rstar['w']['rstar'] @ drstar

# # Aggregate dynamics, multiplicative shock
# # path_div_rstar = Div_ss_rstar * (1 + G_rstar['Div']['rstar'] @ drstar)
# # path_n_rstar = N_ss_rstar * (1 + G_rstar['N']['rstar'] @ drstar)
# # path_r_rstar = r_ss_rstar * (1 + G_rstar['r']['rstar'] @ drstar)
# # path_tau_rstar = Tau_ss_rstar * (1 + G_rstar['Tau']['rstar'] @ drstar)
# # path_w_rstar = w_ss_rstar * (1 + G_rstar['w']['rstar'] @ drstar)

# # Compute all individual consumption paths
# print("Computing individual paths...", end=" ")
# V_prime_p_rstar = (1 + r_ss_rstar) / (1 + tauc) * c_ss_rstar ** (-gamma)
# c_all_rstar = np.zeros((nE, nA, T))
# for t in range(T-1, -1, -1):
#     V_prime_p_rstar, _, c, _ = iterate_h(household_d, V_prime_p_rstar, a_grid_rstar, e_grid_rstar, Pi_rstar, pi_e_rstar, beta, gamma,
#                                           path_div_rstar[t], path_n_rstar[t], path_r_rstar[t], path_tau_rstar[t], tauc, taun, path_w_rstar[t])
#     c_all_rstar[:, :, t] = c
# print("Done")

# # Direct effect of policy
# print("Computing direct effect...", end=" ")
# V_prime_p_rstar = (1 + r_ss_rstar) / (1 + tauc) * c_ss_rstar ** (-gamma)
# c_direct_rstar = np.zeros((nE, nA, T))
# for t in range(T-1, -1, -1):
#     V_prime_p_rstar, _, c, _ = iterate_h(household_d, V_prime_p_rstar, a_grid_rstar, e_grid_rstar, Pi_rstar, pi_e_rstar, beta, gamma,
#                                           Div_ss_rstar, N_ss_rstar, path_r_rstar[t], Tau_ss_rstar, tauc, taun, w_ss_rstar)
#     c_direct_rstar[:, :, t] = c
# print("Done")

# # Select first period only and express as deviation from steady state
# c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau
# c_first_dev_tau_direct = (c_direct_tau[:, :, 0] - c_ss_tau) / c_ss_tau
# c_first_dev_rstar = (c_all_rstar[:, :, 0] - c_ss_rstar) / c_ss_rstar
# c_first_dev_rstar_direct = (c_direct_rstar[:, :, 0] - c_ss_rstar) / c_ss_rstar

# # c_first_dev_tau = (c_all_tau[:, :, 0]) / c_ss_tau - 1
# # c_first_dev_tau_direct = (c_direct_tau[:, :, 0]) / c_ss_tau - 1
# # c_first_dev_rstar = (c_all_rstar[:, :, 0] ) / c_ss_rstar - 1
# # c_first_dev_rstar_direct = (c_direct_rstar[:, :, 0]) / c_ss_rstar- 1

# # Weigh response by mass of agents
# c_first_tau, c_first_tau_direct, c_first_rstar, c_first_rstar_direct = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
# for i in range(nA):
#     c_first_tau[i] = (c_first_dev_tau[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])
#     c_first_tau_direct[i] = (c_first_dev_tau_direct[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])
#     c_first_rstar[i] = (c_first_dev_rstar[:, i] @ D_ss_rstar[:, i]) / np.sum(D_ss_rstar[:,i])
#     c_first_rstar_direct[i] = (c_first_dev_rstar_direct[:, i] @ D_ss_rstar[:, i]) / np.sum(D_ss_rstar[:,i])
#     # c_first_tau[i] = (c_first_dev_tau[:, i] @ D_ss_tau[:, i])
#     # c_first_tau_direct[i] = (c_first_dev_tau_direct[:, i] @ D_ss_tau[:, i])
#     # c_first_rstar[i] = (c_first_dev_rstar[:, i] @ D_ss_rstar[:, i])
#     # c_first_rstar_direct[i] = (c_first_dev_rstar_direct[:, i] @ D_ss_rstar[:, i])

# # Compute indirect effects
# c_first_tau_indirect = c_first_tau - c_first_tau_direct
# c_first_rstar_indirect = c_first_rstar - c_first_rstar_direct

# # # Pool into percentile bins
# # c_first_bin_tau = c_first_tau.reshape(-1, 100, order='F').mean(axis=0)
# # c_first_bin_tau_direct = c_first_tau_direct.reshape(-1, 100, order='F').mean(axis=0) 
# # c_first_bin_tau_indirect = c_first_bin_tau - c_first_bin_tau_direct
# # c_first_bin_rstar = c_first_rstar.reshape(-1, 100, order='F').mean(axis=0)  
# # c_first_bin_rstar_direct = c_first_rstar_direct.reshape(-1, 100, order='F').mean(axis=0) 
# # c_first_bin_rstar_indirect = c_first_bin_rstar - c_first_bin_rstar_direct

# # # Smoothing function
# # import scipy as sp
# # def kernel_smoothing(vec, bandwidth):
# #     n = np.size(vec)
# #     result = np.zeros(n)
# #     for i in range(n):
# #         kernel = sp.stats.norm(vec[i], bandwidth)
# #         weights = kernel.pdf(vec)
# #         weights = weights/np.sum(weights)
# #         result[i] = weights @ vec
# #     return result

# # # Smooth curves
# # bandwith = 0.00
# # c_first_bin_tau = kernel_smoothing(c_first_bin_tau, bandwith)
# # c_first_bin_tau_direct = kernel_smoothing(c_first_bin_tau, bandwith)
# # c_first_bin_tau_indirect = kernel_smoothing(c_first_bin_tau, bandwith)
# # c_first_bin_rstar = kernel_smoothing(c_first_bin_tau, bandwith)
# # c_first_bin_rstar_direct = kernel_smoothing(c_first_bin_tau, bandwith)
# # c_first_bin_rstar_indirect = kernel_smoothing(c_first_bin_tau, bandwith)

# # X-axis
# D_ss_quant = 100 * np.cumsum(np.sum(D_ss_tau, axis=0))

# # First percentile
# D_ss_quant = np.append(0, D_ss_quant)
# c_first_tau_direct = np.append(c_first_tau_direct[0], c_first_tau_direct)
# c_first_tau_indirect =  np.append(c_first_tau_indirect[0], c_first_tau_indirect)
# c_first_rstar_direct = np.append(c_first_rstar_direct[0], c_first_rstar_direct)
# c_first_rstar_indirect =  np.append(c_first_rstar_indirect[0], c_first_rstar_indirect)
 
# # Plot results
# color_map = ["#FFFFFF", "#D95319"] # myb: "#0072BD"
# fig, ax = plt.subplots(1,2)
# ax[0].set_title(r'Interest rate policy')
# ax[0].plot(D_ss_quant, 100 * c_first_rstar_direct, label="Direct effect", linewidth=3)  
# ax[0].stackplot(D_ss_quant, 100 * c_first_rstar_direct, 100 * c_first_rstar_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)  
# ax[0].legend(loc='upper left', frameon=False)
# ax[0].set_xlabel("Wealth percentile"), ax[0].set_ylabel("Percent deviation from steady state")

# ax[1].set_title(r'Transfer policy')
# ax[1].plot(D_ss_quant, 100 * c_first_tau_direct, label="Direct effect", linewidth=3)    
# ax[1].stackplot(D_ss_quant, 100 * c_first_tau_direct, 100 * c_first_tau_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)   
# ax[1].legend(loc='upper right', frameon=False)
# ax[1].set_xlabel("Wealth percentile")
# plt.show()
     
# # # plot results
# # plt.title(r'impact response of consumption $c$ to transfer policy versus interest rate policy')
# # plt.plot(c_first_tau * 100, label="transfer policy")
# # plt.plot(c_first_rstar * 100,'-.', label="interest rate policy")
# # # plt.legend(loc='upper right', frameon=false)
# # plt.xlabel("wealth percentile"), plt.ylabel("percent deviation from steady state")
# # plt.show()


# =============================================================================
# Distribution: Consumption tax vs transfer
# =============================================================================

# Common parameters and steady-state variables
beta = ss_tau['beta']
a_grid_tau = ss_tau.internals['household']['a_grid']
e_grid_tau = ss_tau.internals['household']['e_grid']
c_ss_tau = ss_tau.internals['household']['c']
Pi_tau = ss_tau.internals['household']['Pi']
pi_e_tau = ss_tau.internals['household']['pi_e']
# D_ss_tau = ss_tau.internals['household']['Dbeg']
D_ss_tau = ss_tau.internals['household']['D']
V_prime_tau_ss = (1 + rstar) / (1 + tauc) * c_ss_tau ** (-gamma)

# Policy 1: consumption tax
print("\nPOLICY 1: CONSUMPTION TAX")

# Aggregate transition dynamics
path_div_tauc = Div_ss_tau + G_tau['Div']['tauc'] @ dtauc
path_n_tauc = N_ss_tau + G_tau['N']['tauc'] @ dtauc
path_r_tauc = r_ss_tau + G_tau['r']['tauc'] @ dtauc
path_tauc_tauc = tauc + dtauc
path_w_tauc = w_ss_tau + G_tau['w']['tauc'] @ dtauc

# Compute all individual consumption paths
print("Computing individual paths...", end=" ")
V_prime_p_tau = V_prime_tau_ss
c_all_tauc = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                        path_div_tauc[t], path_n_tauc[t], path_r_tauc[t], Tau_ss_tau, path_tauc_tauc[t], taun, path_w_tauc[t])
    c_all_tauc[:, :, t] = c  
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
V_prime_p_tau = V_prime_tau_ss
c_direct_tauc = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                        Div_ss_tau, N_ss_tau, r_ss_tau, Tau_ss_tau, path_tauc_tauc[t], taun, w_ss_tau)
    c_direct_tauc[:, :, t] = c
print("Done")


# Policy 2: transfer
print("\nPOLICY 2: TRANSFER")

# Aggregate transition dynamics
path_div_tau = Div_ss_tau + G_tau['Div']['Tau'] @ dtau
path_n_tau = N_ss_tau + G_tau['N']['Tau'] @ dtau
path_r_tau = r_ss_tau + G_tau['r']['Tau'] @ dtau
path_tau_tau = Tau_ss_tau + dtau
path_w_tau = w_ss_tau + G_tau['w']['Tau'] @ dtau

# Compute all individual consumption paths
print("Computing individual paths...", end=" ")
V_prime_p_tau = V_prime_tau_ss
c_all_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                        path_div_tau[t], path_n_tau[t], path_r_tau[t], path_tau_tau[t], tauc, taun, path_w_tau[t])
    c_all_tau[:, :, t] = c  
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
V_prime_p_tau = V_prime_tau_ss
c_direct_tau = np.zeros((nE, nA, T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, c, _ = iterate_h(household_d, V_prime_p_tau, a_grid_tau, e_grid_tau, Pi_tau, pi_e_tau, beta, gamma,
                                        Div_ss_tau, N_ss_tau, r_ss_tau, path_tau_tau[t], tauc, taun, w_ss_tau)
    c_direct_tau[:, :, t] = c
print("Done")

# Select first period only and express as deviation from steady state
c_first_dev_tauc = (c_all_tauc[:, :, 0] - c_ss_tau) / c_ss_tau
c_first_dev_tauc_direct = (c_direct_tauc[:, :, 0] - c_ss_tau) / c_ss_tau
c_first_dev_tau = (c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau
c_first_dev_tau_direct = (c_direct_tau[:, :, 0] - c_ss_tau) / c_ss_tau

# Weigh response by mass of agents
c_first_tauc, c_first_tauc_direct, c_first_tau, c_first_tau_direct = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
for i in range(nA):
    c_first_tauc[i] = (c_first_dev_tauc[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])
    c_first_tauc_direct[i] = (c_first_dev_tauc_direct[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])
    c_first_tau[i] = (c_first_dev_tau[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])
    c_first_tau_direct[i] = (c_first_dev_tau_direct[:, i] @ D_ss_tau[:, i]) / np.sum(D_ss_tau[:,i])

# Compute indirect effects
c_first_tauc_indirect = c_first_tauc - c_first_tauc_direct
c_first_tau_indirect = c_first_tau - c_first_tau_direct

# X-axis
D_ss_quant = 100 * np.cumsum(np.sum(D_ss_tau, axis=0))

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
ax[0].legend(loc='upper left', frameon=False)
ax[0].set_xlabel("Wealth percentile"), ax[0].set_ylabel("Percent deviation from steady state")

ax[1].set_title(r'Transfer policy')
ax[1].plot(D_ss_quant, 100 * c_first_tau_direct, label="Direct effect", linewidth=3)    
ax[1].stackplot(D_ss_quant, 100 * c_first_tau_direct, 100 * c_first_tau_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)   
ax[1].legend(loc='upper right', frameon=False)
ax[1].set_xlabel("Wealth percentile")
plt.show()


# # # =============================================================================
# # # Individual vs aggregate impact responses
# # # =============================================================================

# # c_agg_tau_tot = dC[0][0] / ss_tau['C'] * 100
# # c_tau_tot = np.sum(c_all_tau[:, :, 0] * D_ss_tau) * 100
# # c_dev_tau_tot = np.sum((c_all_tau[:, :, 0] - c_ss_tau) * D_ss_tau) * 100 # absolute deviation from ss
# # c_dev_tau_tot = np.sum((c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau * D_ss_tau) * 100 # percent deviation from ss
# # print("Aggregate impact consumption response              = ", round(c_agg_tau_tot, 3), "%")
# # print("Sum of all individual impact consumption responses = ", round(c_dev_tau_tot, 3), "%")


# # # Plot
# # fig, ax = plt.subplots(2, 3)
# # # fig.suptitle('Individual vs Aaggregate responses', size=16)
# # iT = 30
# # ax[0, 0].set_title(r'Hours')
# # # ax[0, 0].plot(N_ss_tau + dN[0][:iT], label="Aggregate")
# # ax[0, 0].plot(N_ss_tau * (1 + dN[0][:iT]), label="Aggregate")
# # ax[0, 0].plot(path_n_tau[:iT],'-.', label="Individual")
# # ax[0, 0].legend(loc='upper right', frameon=False)

# # ax[0, 1].set_title(r'Real interest rate')
# # # ax[0, 1].plot(r_ss_tau + dr[0][:iT], label="Aggregate")
# # ax[0, 1].plot(r_ss_tau * (1 + dr[0][:iT]), label="Aggregate")
# # ax[0, 1].plot(path_r_tau[:iT],'-.', label="Individual")
# # ax[0, 1].legend(loc='upper right', frameon=False)

# # ax[0, 2].set_title(r'Wage')
# # # ax[0, 2].plot(w_ss_tau + dw[0][:iT], label="Aggregate")
# # ax[0, 2].plot(w_ss_tau * (1 + dw[0][:iT]), label="Aggregate")
# # ax[0, 2].plot(path_w_tau[:iT],'-.', label="Individual")
# # ax[0, 2].legend(loc='upper right', frameon=False)

# # ax[1, 0].set_title(r'Transfers')
# # # ax[1, 0].plot(Tau_ss_tau + dT[0][:iT], label="Aggregate")
# # ax[1, 0].plot(Tau_ss_tau * (1 + dT[0][:iT]), label="Aggregate")
# # ax[1, 0].plot(path_tau_tau[:iT],'-.', label="Individual")
# # ax[1, 0].legend(loc='upper right', frameon=False)

# # ax[1, 1].set_title(r'Dividends')
# # # ax[1, 1].plot(Div_ss_tau + dd[0][:iT], label="Aggregate")
# # ax[1, 1].plot(Div_ss_tau * (1 + dd[0][:iT]), label="Aggregate")
# # ax[1, 1].plot(path_div_tau[:iT],'-.', label="Individual")
# # ax[1, 1].legend(loc='upper right', frameon=False)
# # plt.show()





print("\nTime elapsed: %s seconds" % (round(time.time() - start_time, 0)))
