"""
@author: ARMSTC12
Cameron Armstrong
March 2022
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy as sp
from scipy import special
from scipy.optimize import curve_fit


filename = input('Enter the filename: ')

# pfr experimental parameters
Q = 8.3*10**-8                      #vol flow rate in m^3/s
ID = 1.5875/1000                    #inner diameter in m            
A = sp.pi*(ID/2)**2                 #cross-sectional area of reactor in m^2
u = Q/A                             #avg. velocity m/s
V = 1.0*10**-5                      #reactor volume in m^3
xl = V/A                            #tubing length m
DuL = .00001                        #initial guess for dispersion number
tau_theo = int(V/Q)                 #theoretical mean res. time (MRT) in s

data = pd.read_csv(filename,header=None)        
data.columns = ['time','normC']

t_data = data.time[:]

#f-curve (step-change) analytical solution for PFR
def f_curve(t,tau,DuL):
  return 0.5*(special.erf(((t/tau)-1)/(2*np.sqrt(DuL)))+special.erf((1)/(2*np.sqrt(DuL))))

#c-curve (pulse) analytical solution for PFR
def c_curve(t,tau,D):
  return (1/(2*np.sqrt(3.14*(DuL))))*np.exp(-(1-(t/tau))**2/(4*DuL))


#curve fitting for tau and DuL
#returns optimal parameter fit value and covariance as arrays *popt & *pcov
#minimizes squared residuals of f(x,*popt)-y
#tau will be popt[0] and DuL is popt[1]
#std dev error is calculated as perr
popt, pcov = curve_fit(f_curve,data.time[:],data.normC[:],method='lm')
perr = np.sqrt(np.diag(pcov))

#plots RTD data, fitted data, and theoretical ideal PFR against normalized time
plt.figure(1)
plt.plot(t_data/popt[0],data.normC[:],label = 'RTD Experimental Data', marker="o")
plt.plot(t_data/popt[0],f_curve(t_data,*popt),label = 'Curve Fit: D/uL = %s, D = %s'%(np.round(popt[1],4),np.round(popt[1]*u*xl,5)))
plt.plot(t_data/tau_theo,f_curve(t_data,tau_theo,DuL), label='theoretical residence time with ideal DuL')
output_text = '\n'.join(('tau theo = %.1f'%tau_theo,'tau = %.1f'%popt[0]))
textprops = dict(boxstyle='square',facecolor='none',edgecolor='black')
plt.xlim(0.5,1.5)
plt.legend(fontsize=14,edgecolor='black')
plt.text(.51, .75, output_text,fontsize=14,bbox=textprops)

#plots RTD data, fitted data, and theoretical ideal PFR againt time
plt.figure(2)
plt.plot(t_data,data.normC[:],label = 'RTD Experimental Data', marker="o")
plt.plot(t_data,f_curve(t_data,*popt),label = 'Curve Fit: D/uL = %s, D = %s'%(np.round(popt[1],4),np.round(popt[1]*u*xl,5)))
plt.plot(t_data,f_curve(t_data,tau_theo,DuL), label='theoretical residence time with ideal DuL')
output_text = '\n'.join(('tau theo = %.1f'%tau_theo,'tau = %.1f'%popt[0]))
textprops = dict(boxstyle='square',facecolor='none',edgecolor='black')
plt.xlim(0.5*tau_theo,1.5*tau_theo)
plt.legend(fontsize=14,edgecolor='black')
plt.text(61, .75, output_text,fontsize=14,bbox=textprops)