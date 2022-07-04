#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:57:11 2022

@author: danielc
"""

import scipy as sp

discount = (1 / (1 + ss_tau['r']))
shock = np.zeros(T)
s1, s2, s3, s4, s5 = 1, 0.5, 0.1782, 5, 2
for x in range(T):
    shock[x] = (s1 - s2 * (x - s5)) * np.exp(-s3 * (x - s5) - s4) 
dtau = shock

discount_vec = discount ** np.arange(T)

plt.plot(dtau)
np.sum(dtau*discount_vec)

def ar_roots(phi1,phi2):
    mat = np.zeros((2,2))
    mat[0,0] = phi1
    mat[0,1] = phi2
    mat[1,0] = 1
    return sp.linalg.eig(mat)

def ar_rule(x1,x2,phi1,phi2):
    return phi1*x1 + phi2*x2

def compute_value(transfer,x0,c,phi1,phi2,discount,T):
    discount_vec = discount ** np.arange(T)
    policy_path = np.zeros(T)
    policy_path[0] = transfer
    policy_path[1] = x0
    policy_path[2] = c*x0
    for i in range(3,T):
        policy_path[i] = ar_rule(policy_path[i-1],policy_path[i-2],phi1,phi2)
    
    return np.sum(policy_path*discount_vec)

def compute_path(transfer,x0,c,phi1,phi2,discount,T):
    discount_vec = discount ** np.arange(T)
    policy_path = np.zeros(T)
    policy_path[0] = transfer
    policy_path[1] = x0
    policy_path[2] = c*x0
    for i in range(3,T):
         policy_path[i] = ar_rule(policy_path[i-1],policy_path[i-2],phi1,phi2)
    return policy_path

phi1 = 0.4
phi2 = 0.2
c = 1
transfer = 0.03

ar_rt = ar_roots(phi1, phi2)
print(ar_rt[0])

if np.max(np.abs(ar_rt[0])) > 1:
    print("Non Stationary")
else:
    x0 = sp.optimize.brentq(lambda x : compute_value(transfer,x,c,phi1,phi2,discount,T), -transfer,0, full_output=True)
    
    pth = compute_path(transfer,x0[0],2,phi1,phi2,discount,T)

    plt.plot(range(20),pth[0:20])

inp = input("Accept?y/N")

if inp.lower() == "n" or inp.lower() == '':
    print("Not accepted")
elif inp.lower() == "y":
    dtau = pth
else:
    print("Select either Y or N")
