#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:45:58 2022

@author: danielcoutinho
"""

# Pool into percentile bins

D_ss_rstar = np.append(0, D_ss_rstar)
D_ss_tau = np.append(0,D_ss_tau)

c_first_tau_direct = np.append(c_first_tau_direct[0], c_first_tau_direct)
c_first_tau_indirect =  np.append(c_first_tau_indirect[0], c_first_tau_indirect)
c_first_rstar_direct = np.append(c_first_rstar_direct[0], c_first_rstar_direct)
c_first_rstar_indirect =  np.append(c_first_rstar_indirect[0], c_first_rstar_indirect)

color_map = ["#FFFFFF", "#D95319"] # myb: "#0072BD"
fig, ax = plt.subplots(1,2)
ax[0].set_title(r'Interest rate policy')
ax[0].plot(D_quant_rstar, 100 * c_first_bin_rstar_direct, label="Direct effect", linewidth=3)  
ax[0].stackplot(D_quant_rstar, 100 * c_first_bin_rstar_direct, 100 * c_first_bin_rstar_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)  
ax[0].legend(loc='upper left', frameon=False)
ax[0].set_xlabel("Wealth percentile"), ax[0].set_ylabel("Percent deviation from steady state")

ax[1].set_title(r'Transfer policy')
ax[1].plot(D_quant_tau, 100 * c_first_bin_tau_direct, label="Direct effect", linewidth=3)    
ax[1].stackplot(D_quant_tau, 100 * c_first_bin_tau_direct, 100 * c_first_bin_tau_indirect, colors=color_map, labels=["", "+ GE"], alpha=0.5)   
ax[1].legend(loc='upper right', frameon=False)
ax[1].set_xlabel("Wealth percentile")
plt.show()
     