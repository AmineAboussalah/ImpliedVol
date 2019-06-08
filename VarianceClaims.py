import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from HestonDiscretizations import EulerHeston, MilsteinHeston
from IVSolver import IVSolver


T = np.arange(0.1, 1.1, 0.1)
r = 0

kappa = 3.
eta = 5.
theta = 0.4**2
vol0 = 0.3**2

'''
Claim 1
'''
for t in T:
    FellerProc, AssetProc, Z = MilsteinHeston(1, t, 0.3, 3, 0.4, 5, -0.5, 1000, 5000)

    dt = np.ones((len(FellerProc)-1, 5000)) * t / (len(FellerProc)-1)

    truePrice = np.round((1./kappa)*(1-np.exp(-kappa*t))*(vol0-theta) + (theta*t), 4)
    estPrice = np.round(np.mean(np.sum(FellerProc[:-1] * dt,0)), 4)

    print(('Analytical Price', truePrice, 'Estimated Price', estPrice))


'''
Claim 2
'''
for t in T:
    FellerProc, AssetProc, Z = MilsteinHeston(1, t, 0.3, 3, 0.4, 5, -0.5, 1000, 5000)

    dt = np.ones((len(FellerProc)-1, 5000)) * t / (len(FellerProc)-1)

    gamma = (1./kappa) * (2.*kappa*theta + 2.*eta + eta**2)
    term1 = (gamma/kappa) * (0.5 + 0.5*np.exp(-2.*kappa*t) - np.exp(-kappa*t)) * (vol0-theta)
    term2 = (gamma*theta*0.5) * (t + (0.5/kappa)*np.exp(-2.*kappa*t) - (0.5/kappa))
    term3 = (vol0**2 / (2. * kappa)) * (1 - np.exp(-2. * kappa * t))
    truePrice = np.round(term1 + term2 + term3, 8)
    estPrice = np.round(np.mean(np.sum(FellerProc[:-1]**2 * dt,0)), 8)

    print(('Analytical Price', np.round(truePrice, 4), \
                    'Estimated Price', np.round(estPrice,4)))
