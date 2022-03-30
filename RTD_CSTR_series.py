"""
@author: ARMSTC12
Cameron Armstrong
March 2022
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import math


filename = input('Enter the filename: ')
tau_theo = input('Enter theoretical residence time (min): ')
tau_theo = float(tau_theo)
n_theo = input('Enter number of tanks: ')
n_theo = int(n_theo)
mode = input('Enter RTD Mode pulse/step: ')

if mode == 'pulse':
    inlet = input('Is there an inlet curve? y/n ')
    if inlet == 'y':
        filename2 = input('Enter the inlet filename: ')
        inlet_data = pd.read_csv(filename2,header=None)
        inlet_data.columns = ['time','normC']
    else:
        filename2 = 'False'

print(' \nCalculating %s mode analysis....\n '%str(mode))

data = pd.read_csv(filename,header=None)
data.columns = ['time','normC']

t_data = data.time[:]
c_data = data.normC[:]

time_theo = np.linspace(0,t_data[len(t_data)-1],500)

if mode == 'pulse':
    c_integral = np.trapz(c_data,t_data)
    E_data = c_data/c_integral
    if filename2 != 'False':
        t_data2 = inlet_data.time[:]
        c_data2 = inlet_data.normC[:]
        c2_int = np.trapz(c_data2,t_data2)
        E_data2 = c_data2/c2_int
        
        E_data = signal.deconvolve(E_data,E_data2)
        E_int = np.trapz(E_data)
        E_data = E_data/E_int

def F_curve_series(t,N,tau):
    term = np.zeros([N,len(t)])
    if N == 1:
        term[0,:] = 1
    elif N == 2:
        term[0,:] = 1
        term[1,:] = (N*(t/tau))
    else:
        term[0,:] = 1
        term[1,:] = (N*(t/tau))
        for i in range(2,N):
            i += 1
            term[i-1,:] = ((N*t/tau)**(i-1))/math.factorial(i-1)
    terms = np.sum(term[:],0)    
    RTD = 1-np.exp(-N*t/tau)*terms
    return RTD

def E_curve_series(t,N,tau):
    if N == 1:
        term = 1/tau
    else:
        term = ((t)**(N-1))/math.factorial(N-1)/((tau/N)**N)
    RTD = np.exp(-t/(tau/N))*term
    return RTD

if mode == 'step':
    n_fixed = n_theo    
    popt, pcov = curve_fit(lambda t_data,tau: F_curve_series(t_data,n_fixed,tau),t_data,c_data,method='lm')
    #popt, pcov = curve_fit(F_curve_series,t_data,c_data,method='lm')
    perr = np.sqrt(np.diag(pcov))
    
    plt.figure(1)
    plt.plot(t_data,c_data,label = 'RTD Experimental Data', marker="o")
    plt.plot(t_data,F_curve_series(t_data,n_theo,*popt),label = 'Curve Fit: n = %s, tau = %s min'%(np.round(n_theo,0),np.round(*popt,2)))
    #plt.plot(t_data,F_curve_series(t_data,*popt),label = 'Curve Fit: n = %s, tau = %s min'%(np.round(popt[0],0),np.round(popt[1],1)))
    plt.plot(time_theo,F_curve_series(time_theo,n_theo,tau_theo), label='theoretical residence time %s min with n = %s tanks'%(tau_theo, n_theo))
    plt.ylim(0,1.05)
    plt.xlabel('time (min)')
    plt.ylabel('C/C0(-)')
    plt.legend(fontsize=14,edgecolor='black')

else:
    n_fixed = n_theo    
    popt, pcov = curve_fit(lambda t_data,tau: E_curve_series(t_data,n_fixed,tau),t_data,E_data,method='lm')
    perr = np.sqrt(np.diag(pcov))
    
    plt.figure(1)
    plt.plot(t_data,E_data,label = 'RTD Experimental Data', marker="o")
    plt.plot(t_data,E_curve_series(t_data,n_theo,*popt),label = 'Curve Fit: n = %s, tau = %s min'%(np.round(n_theo,0),np.round(*popt,1)))
    plt.plot(t_data,E_curve_series(t_data,n_theo,tau_theo), label='theoretical residence time with n = %s tanks'%n_theo)
    plt.xlabel('time (min)')
    plt.ylabel('C/C0(-)')
    plt.legend(fontsize=14,edgecolor='black')


#time_test = np.linspace(0,50,500)
#test = F_curve_series(time_test,1,10)
#plt.plot(time_test,test)

#N = 5
#t = time_theo
#tau = 4
#thet = t/tau
#test1 = 1-np.exp(-N*thet)*(1)
#test2 = 1-np.exp(-N*thet)*(1+N*thet)
#test3 = 1-np.exp(-N*thet)*(1+N*thet+(N*thet)**2/(math.factorial(2)))
#test4 = 1-np.exp(-N*thet)*(1+N*thet+(N*thet)**2/(math.factorial(2))+(N*thet)**3/(math.factorial(3)))
#test5 = 1-np.exp(-N*thet)*(1+N*thet+(N*thet)**2/(math.factorial(2))+(N*thet)**3/(math.factorial(3))+(N*thet)**4/(math.factorial(4)))
#func = F_curve_series(t,N,tau)
#plt.figure(1)
#plt.plot(t,test5,label = 'analytical')
#plt.plot(t,func,label = 'function')
#plt.legend()
