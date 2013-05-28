# Simulating the state and observations
#======================================

import numpy as np
from Computations import WienerFunction

def SimulateSystem(func_a, func_h, sigmaB, sigmaW, Xmean, Xsigma, T, dt):

    t = np.linspace(0, T, T/dt)
    Nt = len(t)

    X = np.zeros(Nt)
    dZ = np.zeros(Nt)

    h = np.zeros(Nt)

    X[0] = np.random.normal(Xmean, Xsigma)

    for i in range(Nt-1):

        a = func_a(X[i])
        h[i] = func_h(X[i])

        dB = WienerFunction(1, dt)
        dW = WienerFunction(1, dt)
        dX = a*dt + sigmaB*dB

        dZ[i] = h[i]*dt + sigmaW*dW
        X[i+1] = X[i] + dX
        
    h[i+1] = func_h(X[i+1])
    dW = WienerFunction(1, dt)
    dZ[Nt-1] = h[i+1]*dt + sigmaW*dW

    return X, dZ, t, h

