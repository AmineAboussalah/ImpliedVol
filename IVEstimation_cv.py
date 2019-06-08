import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import describe
#from HestonDiscretizations import EulerHeston, MilsteinHeston, MixingHeston
#from IVSolver import IVSolver


 font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
 rc('font', **font)
 rc('axes', labelweight='bold')

 K = np.arange(0.9, 1.1, 0.01) ## Strikes
 T = np.array([0.25, 1]) ## Maturities
 r = 0
 vol0 = 0.3**2
 theta = 0.4**2
 kappa = 3
 eta = 5
 rho = -0.5


 '''
 Estimate the IV Smile with 95% confidence bands using Milstein Discretization
 '''
 Prices, PricesUpper, PricesLower = [], [], [] ## Monte Carlo Option Prices
 VolQuarter, VolQuarterUpper, VolQuarterLower = [], [], [] ## IV for T=0.25
 VolYear, VolYearUpper, VolYearLower = [], [], [] ## IV for T=1

 for t in T:
     FellerProc, AssetProc, Z = MilsteinHeston(1, t, 0.3, 3, 0.4, 5, -0.5, 1000, 5000)
     
     dt = np.ones((len(FellerProc)-1, 5000)) * t / (len(FellerProc)-1)     
     #est values of control variates
     
     estPrice0 = np.mean(AssetProc[-1,])
     estPrice1 = np.mean(FellerProc[-1,])
     estPrice2 = np.mean(np.sum(FellerProc[:-1] * dt,0))
     estPrice3 = np.mean(np.sum(FellerProc[:-1]**2 * dt,0))
     est = [estPrice0,estPrice1, estPrice2, estPrice3]
     
     Y = np.array([AssetProc[-1,],FellerProc[-1,], np.sum(FellerProc[:-1] * dt,0), np.sum(FellerProc[:-1]**2 * dt,0)])
     Ycov = np.cov(Y)
     
     #true values of control variates
     ## vol0 = 0.09
     truePrice0 = np.exp(-0.5*(theta*t+(1/kappa)*(theta-vol0)*(np.exp(-kappa*t)-1)) +
                            (0.5)*(theta*t+(1/kappa)*(theta-vol0)*(np.exp(-kappa*t)-1)))
     
     truePrice2 = (1./kappa)*(1-np.exp(-kappa*t))*(vol0-theta) + (theta*t)
     gamma = (1./kappa) * (2.*kappa*theta + eta**2)
     term1 = (gamma/kappa) * (0.5 + 0.5*np.exp(-2.*kappa*t) - np.exp(-kappa*t)) * (vol0-theta)
     term2 = (gamma*theta*0.5) * (t + (0.5/kappa)*np.exp(-2.*kappa*t) - (0.5/kappa))
     term3 = (vol0**2 / (2. * kappa)) * (1 - np.exp(-2. * kappa * t))
     truePrice3 = term1 + term2 + term3
     truePrice1 = vol0 * np.exp(-kappa * t) + theta*(1-np.exp(-kappa*t))
     
     tru = [truePrice0,truePrice1, truePrice2, truePrice3]

     for k in K:
         t, k = np.round(t,4), np.round(k,4)
         if k < 1:
             Payoffs = np.exp(-r*t) * np.maximum(k - AssetProc[-1,:], 0)
         else:
             Payoffs = np.exp(-r*t) * np.maximum(AssetProc[-1,:] - k, 0)

#         MC = np.mean(Payoffs)
#         StdErr = np.std(Payoffs)
         
         #XY covariance matrix
         XYcov = np.cov(Payoffs,Y)[0,1:]
         #gamma_i's
         gam = np.linalg.solve(Ycov, XYcov)
         
         #new payoffs
         Payoffs2 = Payoffs + np.dot(gam, np.transpose(tru-np.transpose(Y)))
         MC = np.mean(Payoffs2)
         #MC = np.mean(Payoffs) + np.dot(gam, np.subtract(tru,est))
         
         StdErr = np.std(Payoffs2)
         #varghat = (1/5000) * (np.var(Payoffs)+ np.dot(np.dot(gam, Ycov), gam ) -2*np.dot(XYcov,gam))
         #StdErr = np.std(Payoffs + np.dot(gam, np.transpose(tru-np.transpose(Y))))
         
         
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
 a = pd.Series(FellerProc[:,0])
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
    
     dt = np.ones((len(FellerProc)-1, 5000)) * t / (len(FellerProc)-1)     
     #est values of control variates
     
     estPrice0 = np.mean(AssetProc[-1,])
     estPrice1 = np.mean(FellerProc[-1,])
     estPrice2 = np.mean(np.sum(FellerProc[:-1] * dt,0))
     estPrice3 = np.mean(np.sum(FellerProc[:-1]**2 * dt,0))
     est = [estPrice0,estPrice1, estPrice2, estPrice3]
     
     Y = np.array([AssetProc[-1,],FellerProc[-1,], np.sum(FellerProc[:-1] * dt,0), np.sum(FellerProc[:-1]**2 * dt,0)])
     Ycov = np.cov(Y)
     
     #true values of control variates
     ## vol0 = 0.09
     truePrice0 = np.exp(-0.5*(theta*t+(1/kappa)*(theta-vol0)*(np.exp(-kappa*t)-1)) +
                            (0.5)*(theta*t+(1/kappa)*(theta-vol0)*(np.exp(-kappa*t)-1)))
     
     truePrice2 = (1./kappa)*(1-np.exp(-kappa*t))*(vol0-theta) + (theta*t)
     gamma = (1./kappa) * (2.*kappa*theta + eta**2)
     term1 = (gamma/kappa) * (0.5 + 0.5*np.exp(-2.*kappa*t) - np.exp(-kappa*t)) * (vol0-theta)
     term2 = (gamma*theta*0.5) * (t + (0.5/kappa)*np.exp(-2.*kappa*t) - (0.5/kappa))
     term3 = (vol0**2 / (2. * kappa)) * (1 - np.exp(-2. * kappa * t))
     truePrice3 = term1 + term2 + term3
     truePrice1 = vol0 * np.exp(-kappa * t) + theta*(1-np.exp(-kappa*t))
     
     tru = [truePrice0,truePrice1, truePrice2, truePrice3]

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

        # MC = np.mean(Payoffs)
#       StdErr = np.std(Payoffs)
         
        #XY covariance matrix
        XYcov = np.cov(OptionPrice,Y)[0,1:]
        #gamma_i's
        gam = np.linalg.solve(Ycov, XYcov)
         
        #new payoffs
        OptionPrice2 = OptionPrice + np.dot(gam, np.transpose(tru-np.transpose(Y)))
        
        #varghat = (1/5000) * (np.var(Payoffs)+ np.dot(np.dot(gam, Ycov), gam ) -2*np.dot(XYcov,gam))
        #StdErr = np.std(Payoffs + np.dot(gam, np.transpose(tru-np.transpose(Y))))
        
        #AnalyticalPrice = np.mean(OptionPrice)
        #StdErr = np.std(OptionPrice)
        AnalyticalPrice = np.mean(OptionPrice2)
        StdErr = np.std(OptionPrice2)
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