import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

def IVSolver(truePrice, K, T, r, S, call=True):
    '''
    sigma - constant volatility under Black-BlackScholes
    K - Strike
    T = Time to Maturity
    r = Risk free rate
    S = Initial asset price
    truePrice: Option price to invert
    call: Call (True) or Put (False)
    '''

    def BlackScholes(sigma, K, T, r, S, call=call):
        d1 = (1/(sigma*np.sqrt(T)))*(np.log(S/K) + (r + ((sigma**2)/2))*T)
        d2 = d1 - sigma*np.sqrt(T)
        if call:
            return (norm.cdf(d1)*S - norm.cdf(d2)*K*np.exp(-r*T)) - truePrice
        else:
            return (norm.cdf(-d2)*K*np.exp(-r*T) - norm.cdf(-d1)*S) - truePrice

    return fsolve(BlackScholes, 0.3, args=(K, T, r, S, call))
