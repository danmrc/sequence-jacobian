# Can only run on a notebook, decorators do not work on emacs?!

import copy
import numpy as np
import matplotlib.pyplot as plt

from sequence_jacobian import het, simple, create_model              # functions
from sequence_jacobian import interpolate, grids, misc, estimation   # modules

def household_guess(a_grid,e_grid,r,w,gamma):
    wel = (1+r)*a_grid[np.newaxis,:] + w*e_grid[:,np.newaxis]
    V_prime = (1+r)*(wel/2)**(-gamma)
    return V_prime

# Important: V_prime MUST be named v_prime_p in the argument of the function! Otherwise, raises error

@het(exogenous = 'Pi',policy = 'a', backward = 'V_prime', backward_init=household_guess)
def household(V_prime_p,a_grid,e_grid,r,w,beta,gamma):
    c_prime = (beta*V_prime_p)**(-1/gamma)
    new_grid = c_prime + a_grid
    wel = (1+r)*a_grid[np.newaxis,:] + w*e_grid[:,np.newaxis]

    a = interpolate.interpolate_y(new_grid,wel,a_grid)
    misc.setmin(a,a_grid[0])
    c = wel - a
    V_prime= (1+r)*c**(-gamma)

    return V_prime,a,c

print(household)
print(f'Inputs: {household.inputs}')
print(f'Macro outputs: {household.outputs}')
print(f'Micro outputs: {household.internals}')

def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi, a_grid


household_ext = household.add_hetinputs([make_grid])

print(household_ext)
print(f'Inputs: {household_ext.inputs}')

@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt

ks = create_model([household_ext, firm, mkt_clearing], name="Krusell-Smith")
print(ks.inputs)

calibration = {'gamma': 1, 'delta': 0.025, 'alpha': 0.11, 'rho_e': 0.966, 'sd_e': 0.5, 'L': 1.0,
               'nE': 7, 'nA': 500, 'amin': 0, 'amax': 200}
unknowns_ss = {'beta': 0.98, 'Z': 0.85, 'K': 3.}
targets_ss = {'r': 0.01, 'Y': 1., 'asset_mkt': 0.}

ss = ks.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='hybr')

print(ss)
