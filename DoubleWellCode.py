#execfile('DoubleWellCode.py')

import matplotlib.pyplot as plt
import numpy as np

from Galerkin import GalerkinMethodFPF, GalerkinMethodUpdate, GalerkinMethodGain
from Computations import Func_dI, WienerFunction
from SystemSimulation import SimulateSystem

def Func_a(X):
    a = X*(1-X*X)
    return a

def Func_h(X):
    h = X 
    return h

def Func_Psi(X):
    #Psi = [np.sin(X), np.cos(X)]
    Psi = [X, X**2]
    return Psi

def Func_GradPsi(X):
    #GradPsi = [np.cos(X), -np.sin(X)]
    GradPsi = [np.ones_like(X), 2*X]
    return GradPsi

def main():

    N = 100

#    sigmaB = 0.4
#    sigmaW = 0.2
    sigmaB = 0.01
    sigmaW = 50

    Xmean = 0
    Xsigma = 0.1
    
    T = 100
    dt = 0.01

    X, dZ, t, h = SimulateSystem(Func_a, Func_h, sigmaB, sigmaW, Xmean, Xsigma, T, dt)



    mu_PF, Sigma_PF = GalerkinMethodFPF(N, Func_a, Func_h, sigmaB, Xmean, Xsigma, T, dt, dZ,  Func_Psi, Func_GradPsi)

    f1=plt.figure()

    plt.subplot(2,1,1)
    plt.hold(True)
    plt.plot(t, dZ/dt, 'r+')
    plt.plot(t, h, 'bo')
    plt.xlabel('$t$')
    plt.ylabel(('$dZ_t/dt$, $h(X_t)$'))
    plt.legend(('$dZ_t/dt$', '$h(X_t)$'))
    #plt.title(('Linear case: $a(X)=-0.5X$, $h(X)=3X$, $\sigma_B=1$, $\sigma_W=0.5$'))
    plt.title(('Nonlinear case: $a(X)=X(1-X^2)$, $h(X)=X$, $\sigma_B=0.4$, $\sigma_W=0.2$'))

    plt.subplot(2,1,2)
    plt.hold(True)
    plt.plot(t,X,'b-')
    plt.plot(t, mu_PF, 'g--')
    plt.xlabel('$t$')
    plt.ylabel('$X_t$, $\mu^N_t$')
    plt.legend(('$X_t$','$\mu^N_t$'))


    plt.show(f1)



if __name__ == "__main__":
    main()



