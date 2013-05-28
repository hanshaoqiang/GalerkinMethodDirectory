#Test K function
#===============
import numpy as np
import matplotlib.pyplot as plt
from Galerkin import GalerkinMethodGain

def Func_h(X):
    h = np.cos(X) 
    return h

def Func_Psi(X):
    Psi = [np.sin(X), np.cos(X)]
    #Psi = [X, X**2]
    return Psi

def Func_GradPsi(X):
    GradPsi = [np.cos(X), -np.sin(X)]
    #GradPsi = [np.ones_like(X), 2*X]
    return GradPsi

def main():

    X = np.random.uniform(0, 2*np.pi, 1000)
    h = Func_h(X)

    K = GalerkinMethodGain(X, Func_h, Func_Psi, Func_GradPsi)
  
    f1=plt.figure()
    plt.hold(True)
    plt.plot(X, h, 'bo')
    plt.plot(X, K, 'r+')
    plt.xlabel('$X$')
    plt.ylabel('$h(X)$, $K(X)$')
    plt.legend(('$h(X)$', '$K(X)$'))
    plt.show(f1)

if __name__ == "__main__":
    main()

