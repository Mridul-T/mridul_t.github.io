#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is distributed for the course:
    Course code: APL 747
    Course name: Uncertainty Quantification and Propagation 
    Affiliation: Indian Institute of Technology Delhi 
    Semester: Spring 2024

@author: APL747
"""

import numpy as np
import matplotlib.pyplot as plt
import HW3_utils_data as utils

# %%

T = 0.5               # Total time of integration
dt = 0.001          # Time step
N = int(T/dt)       # No of time points
a = 1               # Length of space
J = 256             # Space discretization points

x = np.arange(0,a,a/J)           # Discretization points
u0 = 0.25*np.sin(2*np.pi*x)                # Initial condition

print("STARTING MCS")
ut_mcs=[]
i=0
alphas = np.random.normal(loc=1,scale=0.1,size=1000)
for alpha in alphas:
    i+=1
    print(i)
    ut, t = utils.pde_oned_Galerkin(u0, T, a, N, J, epsilon=0,alpha=alpha)
    ut_mcs.append(ut)
ut_mcs=np.array(ut_mcs)
print(ut_mcs.shape)
print("FINISHING MCS")
# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

fig1 = plt.figure(figsize=(12,8), dpi=100)
heatmap=plt.imshow(np.mean(ut_mcs,axis=0), aspect='auto', cmap='jet')
plt.xlabel('Time (T)')
plt.ylabel('Spatial dimension (x)')
cbar = plt.colorbar(heatmap)
plt.title('1D Burgers Equation', fontweight='bold')
plt.show()

fig1 = plt.figure(figsize=(12,8), dpi=100)
heatmap=plt.imshow(np.std(ut_mcs,axis=0), aspect='auto', cmap='jet')
plt.xlabel('Time (T)')
plt.ylabel('Spatial dimension (x)')
cbar = plt.colorbar(heatmap)
plt.title('1D Burgers Equation', fontweight='bold')
plt.show()
