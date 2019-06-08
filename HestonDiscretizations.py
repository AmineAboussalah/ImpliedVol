import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def EulerHeston(S0, T, vol0, kappa, Rtheta, eta, rho, M, N):

    '''
    S0 - Initial asset price
    T - Time to option expiration
    vol0 - Initial volatility of the CIR volatility process
    kappa - Mean-reverting parameter
    Rtheta - Square-root of the volatility process mean
    eta - vol-vol
    rho - Correlation between the Wiener processes
    M - Number of discretized time intervals
    N - Number of Monte Carlo simulations
    '''

    theta = Rtheta**2
    dt = T / M
    sqrtdt = np.sqrt(T / M)


    Vt = np.ones((M+1,N)) * vol0**2
    Xt = np.ones((M+1,N)) * np.log(S0)
    Z = np.random.multivariate_normal([0,0], [[1,rho],[rho,1]], (M,N))

    for t in np.arange(1,M+1):
        Vt[t] = Vt[t-1] + kappa*(theta - Vt[t-1])*dt + \
                eta*np.sqrt(np.maximum(Vt[t-1],0))*sqrtdt*Z[t-1,:,0]
        Xt[t] = Xt[t-1] - 0.5*Vt[t-1]*dt + np.sqrt(np.maximum(Vt[t-1],0))*sqrtdt*Z[t-1,:,1]

    return Vt, np.exp(Xt)



def MilsteinHeston(S0, T, vol0, kappa, Rtheta, eta, rho, M, N):
    '''
    S0 - Initial asset price
    T - Time to option expiration
    vol0 - Initial volatility of the CIR volatility process
    kappa - Mean-reverting parameter
    Rtheta - Square-root of the volatility process mean
    eta - vol-vol
    rho - Correlation between the Wiener processes
    M - Number of discretized time intervals
    N - Number of Monte Carlo simulations
    '''

    theta = Rtheta**2
    dt = T / M
    sqrtdt = np.sqrt(T / M)

    Vt = np.ones((M+1,N)) * vol0**2
    Xt = np.ones((M+1,N)) * np.log(S0)
    Z = np.random.multivariate_normal([0,0], [[1,rho],[rho,1]], (M,N))

    for t in np.arange(1,M+1):
        Negative = (Vt[t-1] >= 0).astype('uint8')
        print(Negative)
        Vt[t] = Vt[t-1] + kappa*(theta - Vt[t-1])*dt + \
                eta*np.sqrt(np.maximum(Vt[t-1],0))*sqrtdt*Z[t-1,:,0] + \
                Negative*(1./4)*(eta**2)*((dt*(Z[t-1,:,0]**2))-dt)
        Xt[t] = Xt[t-1] + (-0.5*Vt[t-1]*dt) + np.sqrt(np.maximum(Vt[t-1],0))*sqrtdt*Z[t-1,:,1]
    return Vt, np.exp(Xt), Z


def MixingHeston(S0, T, vol0, kappa, Rtheta, eta, rho, M, N):
    '''
    S0 - Initial asset price
    T - Time to option expiration
    vol0 - Initial volatility of the CIR volatility process
    kappa - Mean-reverting parameter
    Rtheta - Square-root of the volatility process mean
    eta - vol-vol
    rho - Correlation between the Wiener processes
    M - Number of discretized time intervals
    N - Number of Monte Carlo simulations
    '''

    theta = Rtheta**2
    dt = T / M
    sqrtdt = np.sqrt(T / M)

    Vt = np.ones((M+1,N)) * vol0**2
    Z_V = np.random.normal(size=(M,N))
    Z_V_Perp = np.random.normal(size=N)

    '''
    Simulate N Feller Processes
    '''
    for t in np.arange(1,M+1):
        Negative = (Vt[t-1] >= 0).astype('uint8')
        print(Negative)
        Vt[t] = Vt[t-1] + kappa*(theta - Vt[t-1])*dt + \
                eta*np.sqrt(np.maximum(Vt[t-1],0))*sqrtdt*Z_V[t-1,:] + \
                Negative*(1./4)*(eta**2)*(dt*Z_V[t-1,:]**2-dt)

    '''
    Compute for Black-Scholes-like form of terminal asset price conditional on the filtration G
    '''
    alpha = np.sum(Vt[:-1] * dt, axis=0) ## Reimman Integral term
    beta = np.sum(np.sqrt(np.maximum(Vt[:-1],0) * dt) * Z_V, axis=0) ## Ito term measurable w.r.t. G

    A = np.log(S0) - 0.5*alpha + rho*beta
    B = (1 - rho**2)*np.maximum(alpha,0)

    return Vt, A, B
