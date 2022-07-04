#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:52:13 2022

@author: danielc
"""

a_all_tau_zero = a_all_tau[:,:,0]
next_a_binned = np.digitize(a_all_tau_zero,a_grid_tau)
next_a_binned = next_a_binned - 1

D = np.zeros((nE,nA,T))
D[:,:,0] = D_ss_tau

for t in range(T-1):
    next_a_binned = np.digitize(a_all_tau[:,:,t],a_grid_tau)
    next_a_binned = next_a_binned - 1
    for i in range(nA):
        indexes = np.nonzero(next_a_binned == i)
        marg_D[i,t+1] = np.sum(marg_D[indexes,t])

plt.plot(marg_D[:,1],label = "t=0")
plt.plot(marg_D_ss, label = "Steady State")
plt.legend()
plt.show()