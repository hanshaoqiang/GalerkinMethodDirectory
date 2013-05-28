#Miscellaneous Computations
#==========================

import numpy as np

def Func_dI(X, dZ, func_h, dt):
    h = func_h(X)
    h_mean = np.mean(h)
    dI = dZ - (h + h_mean)*dt/2
    return dI   		

def WienerFunction(N, dt):
    dB = np.random.normal(0, np.sqrt(dt), N)
    return dB
