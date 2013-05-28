#Galerkin Method Implementation
#==============================

import numpy as np
import random
from Computations import Func_dI, WienerFunction


def GalerkinMethodFPF(N, func_a, func_h, sigmaB, Xmean, Xsigma, T, dt, dZ, func_psi, func_gradpsi):

    t = np.linspace(0, T, T/dt)
    Nt = len(t)

    mu_PF = np.zeros(Nt)
    Sigma_PF = np.zeros(Nt)

    X_PF = np.random.normal(Xmean, Xsigma, N)

    for i in range(Nt-1):
        mu_PF[i] = np.mean(X_PF);
        Sigma_PF[i] = np.var(X_PF);
        dX = GalerkinMethodUpdate(X_PF, dZ[i], func_a, func_h, dt, sigmaB, func_psi, func_gradpsi)
        X_PF = X_PF + dX

    mu_PF[Nt-1] = np.mean(X_PF)
    Sigma_PF[Nt-1] = np.var(X_PF)

    return mu_PF, Sigma_PF



def GalerkinMethodUpdate(X, dZ, func_a, func_h, dt, sigmaB, func_psi, func_gradpsi):

    a = func_a(X)
    dB = WienerFunction(len(X), dt)
    dI = Func_dI(X, dZ, func_h, dt)

    K = GalerkinMethodGain(X, func_h, func_psi, func_gradpsi)

    dX = a*dt + sigmaB*dB + np.dot(K,dI)

    return dX



def GalerkinMethodGain(X, func_h, func_psi, func_gradpsi):
# X is the vector (over particles) of state-values at a particular time.
# BasisID is an integer that indicates choice of basis function
# BasisOrder is the order of basis function
# func_h is the filtering function h(X) for the observation equation
# print 'GalerkinMethodGain: np.shape(X)', np.shape(X) 
    
    Psi = np.array(func_psi(X))
    GradPsi = np.array(func_gradpsi(X))
  
    N = len(X) #Number of particles
		
    h = func_h(X)
    h_diff = h - np.mean(h)

    A = np.dot(GradPsi, GradPsi.T)/N
    b = np.dot(Psi, h_diff)/N
    kappa = np.linalg.solve(A,b)

    K = np.dot(GradPsi.T, kappa)

    return K

