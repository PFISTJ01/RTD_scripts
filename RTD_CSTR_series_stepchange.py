"""
@author: ARMSTC12
Cameron Armstrong
March 2022
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math


filename = input('Enter the filename: ')
tau_theo = input('Enter theoretical residence time (min): ')
tau_theo = float(tau_theo)
n_theo = input('Enter number of tanks: ')
n_theo = int(n_theo)

print(' \nCalculating....\n ')

data = pd.read_csv(filename,header=None)
data.columns = ['time','normC']

t_data = data.time[:]

def F_curve_series(t,N,tau):
    term = np.empty([N,len(t)])
    if N == 1:
        term[0,:] = 1
    elif N == 2:
        term[0,:] = 1
        term[1,:] = (N*(t/tau))
    else:
        term[0,:] = 1
        term[1,:] = (N*(t/tau))
        for i in range(2,N-1):
            i += 1
            term[i-1,:] = ((N*t/tau)**(i-1))/math.factorial(i-1)
    terms = np.sum(term[:],0)    
    RTD = 1-np.exp(-N*t/tau)*terms
    return RTD
    
#curve fitting for tau with fixed number of tanks
#returns optimal parameter fit value and covariance as arrays *popt & *pcov
#minimizes squared residuals of f(x,*popt)-y
#n will be popt[0] and tau is popt[1]
#std dev error is calculated as perr
n_fixed = n_theo    
popt, pcov = curve_fit(lambda t_data,tau: F_curve_series(t_data,n_fixed,tau),t_data,data.normC[:],method='lm')
perr = np.sqrt(np.diag(pcov))

#plots RTD data, fitted data, and theoretical ideal CSTR
plt.figure(1)
plt.plot(t_data,data.normC[:],label = 'RTD Experimental Data', marker="o")
plt.plot(t_data,F_curve_series(t_data,n_theo,*popt),label = 'Curve Fit: n = %s, tau = %s min'%(np.round(n_theo,0),np.round(*popt,1)))
plt.plot(t_data,F_curve_series(t_data,n_theo,tau_theo), label='theoretical residence time with n = %s tanks'%n_theo)
plt.ylim(0,1)
plt.xlabel('time (min)')
plt.ylabel('C/C0(-)')
plt.legend(fontsize=14,edgecolor='black')