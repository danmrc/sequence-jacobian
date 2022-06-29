"""Basic New Keynesian model with taxes"""

# =============================================================================
# Models
# =============================================================================

print("RANK MODEL")
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import simple, create_model    # functions

@simple
def household(gamma, N, nu, phi, tauc, taun, w):
    # C = ((1 - taun) * w / (phi * N ** nu)) ** (1/gamma) # no labor supply effect of tauc
    C = ((1 - taun) * w / ((1 + tauc) * phi * N ** nu)) ** (1/gamma)
    return C
    
@simple
def firm(kappa, mu, pi, w, tauc, Y, Z):
    N = Y / Z
    Div = Y - w * N - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    return N, Div

@simple
def monetary(phi_pi, pi, rstar):
    r = (1 + rstar(-1) + phi_pi * pi(-1)) / (1 + pi) - 1
    i = rstar
    return r, i

@simple
def fiscal1(B, C, N, r, Tau, tauc, taun, w): # for model 1: transfer/tax policy
    gov = Tau + (1 + r) * B(-1) - tauc * C - taun * w * N - B # government BC
    Deficit = tauc * C + taun * w * N - Tau # primary surplus
    Trans = Tau
    return gov, Deficit, Trans

@simple
def fiscal2(B, C, N, r, tauc, taun, w, Y): # for model 2: interest rate policy, constant debt
    Tau = tauc * C + taun * w * N - r * B # Immediate adjustment of transfers
    Deficit = tauc * C + taun * w * N - Tau # primary surplus
    return Tau, Deficit

@simple
def fiscal3(B, B_ss, C, N, r, rhot, sigma, Tau, Tau_ss, tauc, taun, w, Y): # for model 2: interest rate policy, variable debt
    gov = B - Tau - (1 + r) * B(-1) + tauc * C + taun * w * N # government BC 
    fiscal_rule = Tau - rhot * Tau(-1) - (1 - rhot) * Tau_ss + (1 - rhot) * sigma * (B / Y - B_ss / 1) # delayed adjustment of transfers
    Deficit = tauc * C + taun * w * N - Tau # primary surplus
    return gov, fiscal_rule, Deficit

@simple
def fiscal3_ss(B, B_ss, C, N, r, tauc, taun, w): # for model 2: interest rate policy, variable debt
    Tau_ss = tauc * C + taun * w * N - r * B # government BC in steady state
    debt = B - B_ss
    return Tau_ss, debt

@simple
def mkt_clearing(B, beta, C, gamma, kappa, mu, pi, r, tauc, Y):
    goods_mkt = Y - C - mu / (mu - 1) / (2 * kappa) * (1 + pi).apply(np.log) ** 2 * Y
    # euler = C ** (-gamma) - beta * (1 + r(+1)) * C(+1) ** (-gamma) # no tauc
    euler = C ** (-gamma) - beta * (1 + tauc) / (1 + tauc(+1)) * (1 + r(+1)) * C(+1) ** (-gamma)
    return goods_mkt, euler

@simple
def nkpc_ss(mu, Z):
    w = Z / mu
    return w

@simple
def nkpc(kappa, mu, pi, r, w, Y, Z):
    nkpc = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc


#==============================================================================
# Common calibration
# =============================================================================

# Steady-state variables
B_ss = 6.0
N_ss = 1.0
pi_ss = 0.0
rstar = 0.01
Y_ss = 1.0
Z_ss = 1.0

# Structural parameters
gamma = 1.0
kappa = 0.05
mu = 1.2
nu = 2.0
rhot = 0.95
sigma = 0.05
tauc = 0.1
taun = 0.036


# =============================================================================
# Model 1: Tax policy
# =============================================================================

print("\nMODEL 1: TAX POLICY")

# Steady state
nk_ss_tauc = create_model([household, firm, monetary, fiscal1, mkt_clearing, nkpc_ss])

calib_tauc = {'B': B_ss, 'pi': pi_ss, 'rstar': rstar, 'Y': Y_ss, 'Z': Z_ss,
              'gamma': gamma, 'kappa': kappa, 'mu': mu, 'nu': nu, 'phi_pi': 0.0,  
              'tauc': tauc, 'taun': taun}
               
unknowns_ss_tauc = {'beta': 0.99, 'phi': 0.75, 'Tau': 0.03}
targets_ss_tauc = {'euler': 0, 'goods_mkt': 0, 'gov': 0}
print("Computing steady state and Jacobian...", end=" ")
ss0_tauc = nk_ss_tauc.solve_steady_state(calib_tauc, unknowns_ss_tauc, targets_ss_tauc, solver="hybr")

# Dynamic model and Jacobian
nk_tauc = create_model([household, firm, monetary, fiscal1, mkt_clearing, nkpc])
ss_tauc = nk_tauc.steady_state(ss0_tauc)
T = 300
exogenous_tauc = ['rstar', 'Z', 'Tau', 'tauc']
unknowns_tauc = ['pi', 'w', 'Y', 'B']
targets_tauc = ['euler', 'goods_mkt', 'gov', 'nkpc']
G_tauc = nk_tauc.solve_jacobian(ss_tauc, unknowns_tauc, targets_tauc, exogenous_tauc, T=T)
print("Done")


# =============================================================================
# Model 2: Interest rate policy
# =============================================================================

print("\nMODEL 2: INTEREST RATE POLICY")

# Steady state
# nk_ss_rstar = create_model([household, firm, monetary, fiscal2, mkt_clearing, nkpc_ss]) # constant debt
nk_ss_rstar = create_model([household, firm, monetary, fiscal3, fiscal3_ss, mkt_clearing, nkpc_ss]) # variable debt

calib_rstar = {'B': B_ss, 'pi': pi_ss, 'rstar': rstar, 'Y': Y_ss, 'Z': Z_ss,
              'gamma': gamma, 'kappa': kappa, 'mu': mu, 'nu': nu, 'phi_pi': 1.5,  
              'tauc': tauc, 'taun': taun, 'rhot': rhot, 'sigma': sigma}
               
# unknowns_ss_rstar = {'beta': 0.99, 'phi': 0.75} # constant debt (fiscal 2)
# targets_ss_rstar = {'euler': 0, 'goods_mkt': 0} # constant debt (fiscal 2)
unknowns_ss_rstar = {'beta': 0.99, 'phi': 0.75, 'Tau': 0.03, 'B_ss': 6.0} # variable debt (fiscal 3)
targets_ss_rstar = {'euler': 0, 'goods_mkt': 0, 'gov': 0, 'debt': 0} # variable debt (fiscal 3)

print("Computing steady state and Jacobian...", end=" ")
ss0_rstar = nk_ss_rstar.solve_steady_state(calib_rstar, unknowns_ss_rstar, targets_ss_rstar, solver="hybr")

# Dynamic model and Jacobian
# nk_rstar = create_model([household, firm, monetary, fiscal2, mkt_clearing, nkpc]) # constant debt
nk_rstar = create_model([household, firm, monetary, fiscal3, mkt_clearing, nkpc]) # variable debt
ss_rstar = nk_rstar.steady_state(ss0_rstar)
T = 300

# exogenous_rstar = ['rstar', 'Z', 'tauc'] # constant debt (fiscal 2)
# unknowns_rstar = ['pi', 'w', 'Y'] # constant debt (fiscal 2)
# targets_rstar = ['euler', 'goods_mkt', 'nkpc'] # constant debt (fiscal 2)
exogenous_rstar = ['rstar', 'Z', 'tauc'] # variable debt (fiscal 3)
unknowns_rstar = ['pi', 'w', 'Y', 'B', 'Tau'] # variable debt (fiscal 3)
targets_rstar = ['euler', 'goods_mkt', 'nkpc', 'gov', 'fiscal_rule'] # variable debt (fiscal 3)

G_rstar = nk_rstar.solve_jacobian(ss_rstar, unknowns_rstar, targets_rstar, exogenous_rstar, T=T)
print("Done")


# =============================================================================
# Steady-state properties
# =============================================================================

ss_param = [['Discount factor', ss_tauc['beta'], 'Intertemporal elasticity', gamma],
            ['Labor supply elasticity', 1 / nu, 'Labor supply disutility', ss_tauc['phi']],  
            ['Goods substitutability', mu / (mu - 1) , 'Price markup', mu],
            ['Price Phillips slope', kappa, 'Taylor rule inflation ', ss_tauc['phi_pi']],
            ['Consumption tax rate', tauc, 'Labor tax rate', taun]]

ss_var_tau = [['Output', ss_tauc['Y'], 'Government debt', ss_tauc['B']],
              ['Consumption', ss_tauc['C'], 'Transfers', ss_tauc['Tau']],
              ['Hours', ss_tauc['N'], 'Dividends', ss_tauc['Div']], 
              ['Wage', ss_tauc['w'], 'Marginal cost', ss_tauc['w'] / ss_tauc['Z']],
              ['Inflation', ss_tauc['pi'], 'Consumption tax revenue', ss_tauc['tauc'] * ss_tauc['C']],
              ['Nominal interest rate', ss_tauc['r'] * (1 + ss_tauc['pi']), 'Labor tax revenue', ss_tauc['taun'] * ss_tauc['N'] * ss_tauc['w']],
              ['Real interest rate', ss_tauc['r'], 'Debt servicing  cost', ss_tauc['r'] * ss_tauc['B']]]

ss_var_rstar = [['Output', ss_rstar['Y'], 'Government debt', ss_rstar['B']],
                ['Consumption', ss_rstar['C'], 'Transfers', ss_rstar['Tau']],
                ['Hours', ss_rstar['N'], 'Dividends', ss_rstar['Div']], 
                ['Wage', ss_rstar['w'], 'Marginal cost', ss_rstar['w'] / ss_rstar['Z']],
                ['Inflation', ss_rstar['pi'], 'Consumption tax revenue', ss_rstar['tauc'] * ss_rstar['C']],
                ['Nominal interest rate', ss_rstar['r']*(1+ss_rstar['pi']), 'Labor tax revenue', ss_rstar['taun'] * ss_rstar['N'] * ss_rstar['w']],
                ['Real interest rate', ss_rstar['r'], 'Debt servicing  cost', ss_rstar['r'] * ss_rstar['B']]]
# Show steady state
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
discount = (1 / (1 + rstar))
rhos = 0.55
drstar = -0.01572 * rhos ** np.arange(T) 
# drstar = -0.01887 * rhos ** np.arange(T) # no tauc in labor supply 
dtau = 0.0223 * rhos ** np.arange(T)
dtauc = - dtau

# Zero net present value sock
shock = np.zeros(T)
# s1, s2, s3, s4, s5 = 1, 0.5, 0.1723464735, 5, 3 # rstar = 0.005
s1, s2, s3, s4, s5 = 1, 0.5, 0.162420896, 5, 3 # rstar = 0.01
for x in range(T):
    shock[x] = discount ** x * (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
dtau = shock
dtauc = - dtau

# # Plot shock
# cumshock = np.zeros(T)
# for i in range(T):
#     cumshock[i] = discount ** i * dtauc[i] # discounted cumulative sum
# plt.plot(shock[0:40], linewidth=2)
# plt.plot([0, 40], [0, 0], '--', color='gray', linewidth=0.5)
# plt.title("Zero net present value shock: " + str(round(np.sum(cumshock), 10)))
# plt.margins(x=0, y=0)
# plt.show()

dY = [G_tauc['Y']['tauc'] @ dtauc, G_tauc['Y']['Tau'] @ dtau, G_rstar['Y']['rstar'] @ drstar]
dC = [G_tauc['C']['tauc'] @ dtauc, G_tauc['C']['Tau'] @ dtau, G_rstar['C']['rstar'] @ drstar]
dN = [G_tauc['N']['tauc'] @ dtauc, G_tauc['N']['Tau'] @ dtau, G_rstar['N']['rstar'] @ drstar]
# dB = [G_tauc['B']['tauc'] @ dtauc, G_tauc['B']['Tau'] @ dtau, np.zeros(T)] # constant debt
dB = [G_tauc['B']['tauc'] @ dtauc, G_tauc['B']['Tau'] @ dtau, G_rstar['B']['rstar'] @ drstar] # variable debt
dW = [G_tauc['w']['tauc'] @ dtauc, G_tauc['w']['Tau'] @ dtau, G_rstar['w']['rstar'] @ drstar]
dP = [G_tauc['pi']['tauc'] @ dtauc, G_tauc['pi']['Tau'] @ dtau, G_rstar['pi']['rstar'] @ drstar]
dr = [G_tauc['r']['tauc'] @ dtauc, G_tauc['r']['Tau'] @ dtau, G_rstar['r']['rstar'] @ drstar]
dD = [G_tauc['Deficit']['tauc'] @ dtauc, G_tauc['Deficit']['Tau'] @ dtau, G_rstar['Deficit']['rstar'] @ drstar]
dd = [G_tauc['Div']['tauc'] @ dtauc, G_tauc['Div']['Tau'] @ dtau, G_rstar['Div']['rstar'] @ drstar]
dT = [np.zeros(T), G_tauc['Trans']['Tau'] @ dtau, G_rstar['Tau']['rstar'] @ drstar]
dTc = [dtauc, np.zeros(T), np.zeros(T)]
di = [np.zeros(T), np.zeros(T), G_rstar['i']['rstar'] @ drstar]

plt.rcParams["figure.figsize"] = (20,7)
fig, ax = plt.subplots(2, 4)
fig.suptitle('Responses to different policies, RANK', size=16)
iT = 30

ax[0, 0].set_title(r'Output $Y$')
ax[0, 0].plot(100 * dY[0][:iT], label="Consumption tax policy")
# ax[0, 0].plot(100 * dY[1][:iT], '-.', label="Transfer policy")
ax[0, 0].plot(100 * dY[2][:iT], '-.', label="Monetary policy")
ax[0, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
ax[0, 0].legend(loc='upper right', frameon=False)

ax[0, 1].set_title(r'Consumption $C$')
ax[0, 1].plot(100 * dC[0][:iT])
# ax[0, 1].plot(100 * dC[1][:iT], '-.')
ax[0, 1].plot(100 * dC[2][:iT], '-.')
ax[0, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[0, 2].set_title(r'Government debt $B$')
ax[0, 2].plot(100 * dB[0][:iT])
# ax[0, 2].plot(100 * dB[1][:iT], '-.')
ax[0, 2].plot(100 * dB[2][:iT], '-.')
ax[0, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[0, 3].set_title(r'Transfer $\tau$')
ax[0, 3].plot(100 * dT[0][:iT])
# ax[0, 3].plot(100 * dT[1][:iT], '-.',)
ax[0, 3].plot(100 * dT[2][:iT], '-.')
ax[0, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 0].set_title(r'Wage $w$')
ax[1, 0].plot(100 * dW[0][:iT])
# ax[1, 0].plot(100 * dW[1][:iT], '-.')
ax[1, 0].plot(100 * dW[2][:iT], '-.')
ax[1, 0].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 1].set_title(r'Inflation $\pi$')
ax[1, 1].plot(100 * dP[0][:iT])
# ax[1, 1].plot(100 * dP[1][:iT], '-.')
ax[1, 1].plot(100 * dP[2][:iT], '-.')
ax[1, 1].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

ax[1, 2].set_title(r'Nominal interest rate $i$')
ax[1, 2].plot(100 * di[0][:iT])
# ax[1, 2].plot(100 * di[1][:iT], '-.')
ax[1, 2].plot(100 * di[2][:iT], '-.')
ax[1, 2].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)

# ax[1, 2].set_title(r'government budget deficit')
# ax[1, 2].plot(-dD[0][:50])
# ax[1, 2].plot(-dD[1][:50], '-.')

ax[1, 3].set_title(r'Consumption tax $\tau_c$')
ax[1, 3].plot(100 * dTc[0][:iT])
# ax[1, 3].plot(100 * dTc[1][:iT], '-.')
ax[1, 3].plot(100 * dTc[2][:iT], '-.')
ax[1, 3].plot([0, iT], [0, 0], '--', color='gray', linewidth=0.5)
plt.show()

# Discounted cumulative sum
cumtau, cumY, cumC, cumP, cumW, cumD, cumB = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
dcumtau, dcumY, dcumC, dcumP, dcumW, dcumD, dcumB = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
for i in range(T):
    cumtau[i] = discount ** i * dT[0][i]
    cumC[i] = discount ** i * dC[0][i]
    cumY[i] = discount ** i * dY[0][i]
    cumP[i] = discount ** i * dP[0][i]
    cumW[i] = discount ** i * dW[0][i]
    # cumD[i] = discount ** i * dD[0][i]
    cumB[i] = discount ** i * dB[0][i]
    dcumtau[i] = discount ** i * (dT[0][i] + dT[2][i])
    dcumY[i] = discount ** i * (dY[0][i] - dY[2][i])
    dcumC[i] = discount ** i * (dC[0][i] - dC[2][i])
    dcumP[i] = discount ** i * (dP[0][i] - dP[2][i])
    dcumW[i] = discount ** i * (dW[0][i] - dW[2][i])
    # dcumD[i] = discount ** i * (dD[0][i] - dD[2][i])
    dcumB[i] = discount ** i * (dB[0][i] - dB[2][i])

# Impact difference
dif = [['\nDYNAMICS', 'CUM SUM \u03C4c','CUM SUM \u03C4c-i*','IMPACT \u03C4c/i*'],
      ['Shocks', 100 * np.sum(cumtau), 100 * np.sum(dcumtau), dtauc[0] * tauc / di[2][0]],
      ['Output', 100 * np.sum(cumY), 100 * np.sum(dcumY), dY[0][0] / dY[2][0]],
      ['Consumption', 100 * np.sum(cumC), 100 * np.sum(dcumC), dC[0][0] / dC[2][0]],
      ['Inflation', 100 * np.sum(cumP), 100 * np.sum(dcumP), dP[0][0] / dP[2][0]],
      ['Wage', 100 * np.sum(cumW), 100 * np.sum(dcumW), dW[0][0] / dW[2][0]],
      ['Debt', 100 * np.sum(cumB), 100 * np.sum(dcumB), dB[0][0] / dB[2][0]]]
for i in range(len(dif)):
    if i == 0:
        print('{:<27s} {:^15s} {:^15s} {:^15s}'.format(dif[i][0], dif[i][1], dif[i][2], dif[i][3]))
    else:
        print('{:<21s} {:>14.3f} {:s} {:>14.3f} {:s} {:>14.3f}'.format(dif[i][0], dif[i][1], "%", dif[i][2], "%", dif[i][3]))


# =============================================================================
# Household iteration policy rule
# =============================================================================

def household(V_prime_p, a_init, beta, gamma, nu, phi, Div, r, w, Tau, tauc, taun):
    C = (beta * (1 + tauc) * V_prime_p) ** (-1/gamma)
    # C = ((1 - taun) * w / ((1 + tauc) * phi * N ** nu)) ** (1/gamma) # foc
    N = ((1 - taun) * w / ((1 + tauc) * phi * C ** gamma)) ** (1/nu) # foc labor 
    A = (1 + r) * a_init + (1 - taun) * w * N + Tau + Div - (1 + tauc) * C 
    V_prime = (1 + r) / (1 + tauc) * C ** (-gamma)
    return V_prime, A, C, N 


def iterate_h(foo, V_prime_start, a_init, beta, gamma, nu, phi, taun, Div, r, w,  
              Tau, tauc, maxit=1000, tol=1E-8):
    V_prime_p = V_prime_start
    V_prime_old = V_prime_start    
    ite = 0
    err = 1    
    while ite < maxit and err > tol:
        # foo is a placeholder, will be household function defined above
        V_prime_temp, A, C, N = foo(V_prime_p, a_init, beta, gamma, nu, phi, Div, r, w, Tau, tauc, taun)
        V_prime_p = V_prime_temp
        ite += 1
        err = np.max(np.abs(V_prime_old - V_prime_temp))
        # print(ite)
        V_prime_old = V_prime_temp 
    return V_prime_temp, A, C, N


# =============================================================================
# Decompose effects
# =============================================================================

# Parameters and steady-state variables
beta = ss_tauc['beta']
phi = ss_tauc['phi']

A_ss = ss_tauc['B']
C_ss = ss_tauc['C']
Div_ss = ss_tauc['Div']
r_ss = ss_tauc['r']
Tau_ss = ss_tauc['Tau']
w_ss = ss_tauc['w']
V_prime_ss = (1 + rstar) / (1 + tauc) * C_ss ** (-gamma)

# Policy 1: consumption tax
print("\nPOLICY 1: CONSUMPTION TAX")

# Aggregate transition dynamics
path_div_tauc = Div_ss + G_tauc['Div']['tauc'] @ dtauc
path_n_tauc = N_ss + G_tauc['N']['tauc'] @ dtauc
path_r_tauc = r_ss + G_tauc['r']['tauc'] @ dtauc
path_tauc_tauc = tauc + dtauc
path_w_tauc = w_ss + G_tauc['w']['tauc'] @ dtauc

# Compute consumption path
print("Computing consumption path...", end=" ")
a_init = A_ss
C_all_tauc, N_all_tauc = np.zeros((T)), np.zeros((T))
for t in range(T-1, -1, -1):
# for t in range(T-1):    
    # print(t)
    V_prime_p_tau, _, C, N = iterate_h(household, V_prime_ss, a_init, beta, gamma, nu, phi, taun, 
                                       path_div_tauc[t], path_r_tauc[t], path_w_tauc[t], Tau_ss, path_tauc_tauc[t])
    C_all_tauc[t] = C
    N_all_tauc[t] = N
print("Done")

# Direct effect of policy
print("Computing direct effect...", end=" ")
a_init = A_ss
V_prime_p_tau = V_prime_ss
C_direct_tauc, N_direct_tauc = np.zeros((T)), np.zeros((T))
for t in range(T-1, -1, -1):
    V_prime_p_tau, _, C, N = iterate_h(household, V_prime_p_tau, a_init, beta, gamma, nu, phi, taun, 
                                        Div_ss, r_ss, w_ss, Tau_ss, tauc)
    C_direct_tauc[t] = C
    N_direct_tauc[t] = N
print("Done")

plt.plot(C_all_tauc[:30], label="Iteration")
plt.plot(dC[0][:30] * 100, label="Aggregate")
plt.legend()
plt.show()


# =============================================================================
# Individual vs aggregate impact responses
# =============================================================================

# c_agg_tau_tot = dC[0][0] / ss_tau['C'] * 100
# c_tau_tot = np.sum(c_all_tau[:, :, 0] * D_ss_tau) * 100
# c_dev_tau_tot = np.sum((c_all_tau[:, :, 0] - c_ss_tau) * D_ss_tau) * 100 # absolute deviation from ss
# c_dev_tau_tot = np.sum((c_all_tau[:, :, 0] - c_ss_tau) / c_ss_tau * D_ss_tau) * 100 # percent deviation from ss
# print("Aggregate impact consumption response              = ", round(c_agg_tau_tot, 3), "%")
# print("Sum of all individual impact consumption responses = ", round(c_dev_tau_tot, 3), "%")

# Plot
fig, ax = plt.subplots(2, 3)
# fig.suptitle('Individual vs Aaggregate responses', size=16)
iT = 30
ax[0, 0].set_title(r'Hours')
ax[0, 0].plot(N_ss + dN[0][:iT], label="Aggregate")
ax[0, 0].plot(path_n_tauc[:iT],'-.', label="Individual")
ax[0, 0].legend(loc='upper right', frameon=False)

ax[0, 1].set_title(r'Real interest rate')
# ax[0, 1].plot(r_ss * (1 + dr[0][:iT]), label="Aggregate")
ax[0, 1].plot(r_ss + dr[0][:iT], label="Aggregate")
ax[0, 1].plot(path_r_tauc[:iT],'-.', label="Individual")
ax[0, 1].legend(loc='upper right', frameon=False)

ax[0, 2].set_title(r'Wage')
ax[0, 2].plot(w_ss + dW[0][:iT], label="Aggregate")
ax[0, 2].plot(path_w_tauc[:iT],'-.', label="Individual")
ax[0, 2].legend(loc='upper right', frameon=False)

ax[1, 0].set_title(r'Transfers')
ax[1, 0].plot(Tau_ss + dT[0][:iT], label="Aggregate")
ax[1, 0].plot(Tau_ss,'-.', label="Individual")
ax[1, 0].legend(loc='upper right', frameon=False)

ax[1, 1].set_title(r'Dividends')
ax[1, 1].plot(Div_ss + dd[0][:iT], label="Aggregate")
ax[1, 1].plot(path_div_tauc[:iT],'-.', label="Individual")
ax[1, 1].legend




print("Time elapsed: %s seconds" % (round(time.time() - start_time, 0)))   