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
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft

"""
Response Generation: Duffing Oscillator:
"""
def duffing(x0, tparam, sysparam):
    # Function for the dynamical systems:
    def F(t, x, params):
        k, c, k3 = params
        force = 10**2*np.sin(2*np.pi*t)*0.1*np.sin(np.pi*t)
        y = np.dot(np.array([[0, 1], [-k, -c]]), x) + np.dot([0, -k3], (x[0]**3)) \
            + np.dot([0, 1], force)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(t0, T, dt)
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval= t_eval, args=(sysparam,))
    xt = np.vstack(sol.y)
    
    return xt, t_eval

"""
Response Generation: Burgers equation:
"""
def pde_oned_Galerkin(u0,T,a,N,J,epsilon,alpha):
    Dx = a/J
    fun_handle = lambda us: us * FiniteDiff(us, Dx, d=1)
    Dt = T/N
    t = np.arange(0,T,Dt)
    ut = np.zeros((J, N+1))
    
    ## set linear operator
    lambda_ = 2*np.pi*np.concatenate((np.arange(0,J/2+1,1), np.arange(-J/2+1,0)))/a 
    M = epsilon*lambda_**2
    EE = 1/(1 + Dt*M)            # diagonal of (1+ Dt M)^{-1}
    ut[:, 0] = u0
    u = u0 
    uh = fft(u);                 # set initial condition
    for n in range(N):           # time loop
        fhu =  alpha*fft(fun_handle(u))    # evaluate fhat(u)
        uh_new = EE *(uh + Dt * fhu)   # semi-implicit Euler step
        u = np.real(ifft(uh_new))
        ut[:,n+1] = u 
        uh = uh_new

    ut[J-1,:] = ut[0,:]          # make periodic
    return ut, t
   
"""
Finite difference derivative:
"""
def FiniteDiff(u, dx, d):
    n = u.size
    ux = np.zeros(n)
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)

"""
RBF random field:
"""
def grf(x): 
    kxx = np.zeros((len(x), len(x)))
    sigma = 0.1
    l = 0.1
    for i in range(len(x)):
        for j in range(len(x)):
            kxx[i,j] = sigma**2 * np.exp(-(x[i]-x[j])**2/(2*l**2))

    xval = np.random.multivariate_normal(np.zeros(len(x)), kxx)
    return xval
