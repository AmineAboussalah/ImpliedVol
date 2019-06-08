import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import describe
from HestonDiscretizations import EulerHeston, MilsteinHeston, MixingHeston
from IVSolver import IVSolver



font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
rc('font', **font)
rc('axes', labelweight='bold')

K = np.arange(0.9, 1.1, 0.01) ## Strikes
T = np.array([0.25, 1]) ## Maturities
r = 0


M = 1000 ## Time steps
N = 5000 ## Simulations
V0 = 0.3**2
S0 = 1
kappa = 3.
eta = 5.
theta = 0.4**2


for t in T:
    dt = t / M
    sqrtdt = np.sqrt(dt)

    '''
    Determine the analytical mean variance E[V_t]
    '''
    increments = np.arange(0, t+dt, dt)
    VtMean = V0*np.exp(-kappa * increments) + theta*(1 - np.exp(-kappa * increments))

    '''
    Simulated Variance Process
    '''
    FellerProc, AssetProc, Z = MilsteinHeston(1, t, 0.3, 3, 0.4, 5, -0.5, 1000, 5000)
    VtSampleMean = np.mean(FellerProc, 1)
    pd.Series(VtMean).plot()
    pd.Series(VtSampleMean).plot()
    plt.show()

    '''
    Simulate the Asset Price process using same randomness as before and using analytical mean
    '''
    Xt = np.ones((M+1,N)) * np.log(S0)
    Vt = Vt = np.ones((M+1,N)) * V0
    for t in np.arange(1,M+1):
        VtSampleMean = np.mean(Vt[t-1]) * np.ones(N)
        Vt[t] = Vt[t-1] + kappa*(theta - VtSampleMean)*dt + \
                eta*np.sqrt(np.maximum(VtSampleMean,0))*sqrtdt*Z[t-1,:,0] + \
                (1./4)*(eta**2)*(dt*Z[t-1,:,0]**2-dt)
        Xt[t] = Xt[t-1] + (-0.5*VtSampleMean*dt) + \
                np.sqrt(np.maximum(VtSampleMean,0))*sqrtdt*Z[t-1,:,1]
    St = np.exp(Xt)
    pd.Series(AssetProc[:,0]).plot()
    pd.Series(St[:,0]).plot()
    plt.show()

    pd.Series(FellerProc[:,0]).plot()
    pd.Series(Vt[:,0]).plot()
    plt.show()
