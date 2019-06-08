import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import describe
from HestonDiscretizations import EulerHeston, MilsteinHeston, MixingHeston
from IVSolver import IVSolver, BlackScholes


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
rc('font', **font)
rc('axes', labelweight='bold')

K = np.arange(0.9, 1.1, 0.01) ## Strikes
T = np.array([0.25, 1]) ## Maturities
r = 0


'''
Estimate the IV Smile with 95% confidence bands using Euler Discretization
'''
Prices, PricesUpper, PricesLower = [], [], [] ## Monte Carlo Option Prices
VolQuarter, VolQuarterUpper, VolQuarterLower = [], [], [] ## IV for T=0.25
VolYear, VolYearUpper, VolYearLower = [], [], [] ## IV for T=1

for t in T:
    FellerProc, AssetProc = EulerHeston(1, t, 0.3, 3, 0.4, 5, -0.5, 1000, 5000)
    for k in K:
        t, k = np.round(t,4), np.round(k,4)
        if k < 1:
            Payoffs = np.exp(-r*t) * np.maximum(k - AssetProc[-1,:], 0)
        else:
            Payoffs = np.exp(-r*t) * np.maximum(AssetProc[-1,:] - k, 0)

        MC = np.mean(Payoffs)
        StdErr = np.std(Payoffs)
        LowerLimit = MC - (1.96 * (StdErr / np.sqrt(5000)))
        UpperLimit = MC + (1.96 * (StdErr / np.sqrt(5000)))
        Prices.append(MC)
        PricesLower.append(LowerLimit)
        PricesUpper.append(UpperLimit)

        if k < 1:
            IV = IVSolver(MC, k, t, 0, 1, call=False)[0]
            IVLower = IVSolver(LowerLimit, k, t, 0, 1, call=False)[0]
            IVUpper = IVSolver(UpperLimit, k, t, 0, 1, call=False)[0]
        else:
            IV = IVSolver(MC, k, t, 0, 1, call=True)[0]
            IVLower = IVSolver(LowerLimit, k, t, 0, 1, call=True)[0]
            IVUpper = IVSolver(UpperLimit, k, t, 0, 1, call=True)[0]
        if t == 0.25:
            VolQuarter.append(IV)
            VolQuarterLower.append(IVLower)
            VolQuarterUpper.append(IVUpper)
        elif t == 1:
            VolYear.append(IV)
            VolYearLower.append(IVLower)
            VolYearUpper.append(IVUpper)

        print((t, np.round(k,4), np.round(LowerLimit,4), np.round(MC, 4), \
                                    np.round(UpperLimit, 4), IV))


'''
Plot the Volatility Smiles from the Euler implementation
'''
plt.plot(K, VolQuarter, 'ro', label='T=0.25')
plt.plot(K, VolQuarterLower, 'r--')
plt.plot(K, VolQuarterUpper, 'r--')
plt.plot(K, VolYear, 'bo', label='T=1')
plt.plot(K, VolYearLower, 'b--')
plt.plot(K, VolYearUpper, 'b--')
plt.tight_layout()
plt.xlim([0.88,1.12])
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Estimated Implied Volatility via Euler Discretization', y=1.02)
plt.legend()
plt.show()

a = pd.Series(FellerProc[:,0])
print(np.mean(np.log(AssetProc[-1,:] / AssetProc[0,:])) * -2.)





'''
Estimate the IV Smile with 95% confidence bands using Milstein Discretization
'''
Prices, PricesUpper, PricesLower = [], [], [] ## Monte Carlo Option Prices
VolQuarter, VolQuarterUpper, VolQuarterLower = [], [], [] ## IV for T=0.25
VolYear, VolYearUpper, VolYearLower = [], [], [] ## IV for T=1

for t in T:
    FellerProc, AssetProc, Z = MilsteinHeston(1, t, 0.3, 3, 0.4, 5, -0.5, 1000, 5000)
    # pd.Series(AssetProc[-1]).plot(kind='kde')
    # plt.show()
    for k in K:
        t, k = np.round(t,4), np.round(k,4)
        if k < 1:
            Payoffs = np.exp(-r*t) * np.maximum(k - AssetProc[-1,:], 0)
        else:
            Payoffs = np.exp(-r*t) * np.maximum(AssetProc[-1,:] - k, 0)

        MC = np.mean(Payoffs)
        StdErr = np.std(Payoffs)
        LowerLimit = MC - (1.96 * (StdErr / np.sqrt(5000)))
        UpperLimit = MC + (1.96 * (StdErr / np.sqrt(5000)))
        Prices.append(MC)
        PricesLower.append(LowerLimit)
        PricesUpper.append(UpperLimit)

        if k < 1:
            IV = IVSolver(MC, k, t, 0, 1, call=False)[0]
            IVLower = IVSolver(LowerLimit, k, t, 0, 1, call=False)[0]
            IVUpper = IVSolver(UpperLimit, k, t, 0, 1, call=False)[0]
        else:
            IV = IVSolver(MC, k, t, 0, 1, call=True)[0]
            IVLower = IVSolver(LowerLimit, k, t, 0, 1, call=True)[0]
            IVUpper = IVSolver(UpperLimit, k, t, 0, 1, call=True)[0]
        if t == 0.25:
            VolQuarter.append(IV)
            VolQuarterLower.append(IVLower)
            VolQuarterUpper.append(IVUpper)
        elif t == 1:
            VolYear.append(IV)
            VolYearLower.append(IVLower)
            VolYearUpper.append(IVUpper)

        print((t, np.round(k,4), np.round(LowerLimit,4), np.round(MC, 4), \
                                    np.round(UpperLimit, 4), IV))
pd.Series(FellerProc[:,0]).plot()
a.plot()
plt.show()

'''
Plot the Volatility Smiles from the Milstein implementation
'''
plt.plot(K, VolQuarter, 'ro', label='T=0.25')
plt.plot(K, VolQuarterLower, 'r--')
plt.plot(K, VolQuarterUpper, 'r--')
plt.plot(K, VolYear, 'bo', label='T=1')
plt.plot(K, VolYearLower, 'b--')
plt.plot(K, VolYearUpper, 'b--')
plt.tight_layout()
plt.xlim([0.88,1.12])
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Estimated Implied Volatility via Milstein Discretization', y=1.02)
plt.legend()
plt.show()





'''
Estimate the IV Smile with 95% confidence bands using Mixing Method
'''
Prices, PricesUpper, PricesLower = [], [], [] ## Monte Carlo Option Prices
VolQuarter, VolQuarterUpper, VolQuarterLower = [], [], [] ## IV for T=0.25
VolYear, VolYearUpper, VolYearLower = [], [], [] ## IV for T=1

for t in T:
    FellerProc, A, B = MixingHeston(1, t, 0.3, 3, 0.4, 5, -0.5, 1000, 5000)
    for k in K:
        t, k = np.round(t,4), np.round(k,4)

        sigmaPrime = np.sqrt(B / t) + 0.0000000001
        rPrime = (A / t) + (B / (2*t))
        print(sigmaPrime)

        '''
        Compute Black Scholes-like formula analytically
        '''
        if k < 1:
            OptionPrice = BlackScholes(sigmaPrime, k, t, rPrime, 1, call=False) * \
                np.exp(rPrime*t)
        else:
            OptionPrice = BlackScholes(sigmaPrime, k, t, rPrime, 1, call=True) * \
                np.exp(rPrime*t)

        AnalyticalPrice = np.mean(OptionPrice)
        StdErr = np.std(OptionPrice)
        LowerLimit = AnalyticalPrice - (1.96 * (StdErr / np.sqrt(5000)))
        UpperLimit = AnalyticalPrice + (1.96 * (StdErr / np.sqrt(5000)))
        Prices.append(AnalyticalPrice)
        PricesLower.append(LowerLimit)
        PricesUpper.append(UpperLimit)

        if k < 1:
            IV = IVSolver(AnalyticalPrice, k, t, 0, 1, call=False)[0]
            IVLower = IVSolver(LowerLimit, k, t, 0, 1, call=False)[0]
            IVUpper = IVSolver(UpperLimit, k, t, 0, 1, call=False)[0]
        else:
            IV = IVSolver(AnalyticalPrice, k, t, 0, 1, call=True)[0]
            IVLower = IVSolver(LowerLimit, k, t, 0, 1, call=True)[0]
            IVUpper = IVSolver(UpperLimit, k, t, 0, 1, call=True)[0]
        if t == 0.25:
            VolQuarter.append(IV)
            VolQuarterLower.append(IVLower)
            VolQuarterUpper.append(IVUpper)
        elif t == 1:
            VolYear.append(IV)
            VolYearLower.append(IVLower)
            VolYearUpper.append(IVUpper)

        print((t, np.round(k,4), np.round(LowerLimit,4), np.round(AnalyticalPrice, 4), \
                                    np.round(UpperLimit, 4), IV))


'''
Plot the Volatility Smiles from the Mixing Method
'''
plt.plot(K, VolQuarter, 'ro', label='T=0.25')
plt.plot(K, VolQuarterLower, 'r--')
plt.plot(K, VolQuarterUpper, 'r--')
plt.plot(K, VolYear, 'bo', label='T=1')
plt.plot(K, VolYearLower, 'b--')
plt.plot(K, VolYearUpper, 'b--')
plt.tight_layout()
plt.xlim([0.88,1.12])
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Estimated Implied Volatility via Mixing Method', y=1.02)
plt.legend()
plt.show()