"""
@author: ARMSTC12
Cameron Armstrong
July 2022
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import special
import math

import warnings
warnings.filterwarnings("ignore")


filename = input('Enter the filename: ')

tau_theo = input('Enter theoretical residence time (min): ')
tau_theo = float(tau_theo)

Q = input('Enter flowrate (mL/min): ')
Q = float(Q)*1e-6/60

ID = input('Enter tubing inner diameter (mm): ')
ID = float(ID)
r = ID/2000
A = math.pi*r**2

u = Q/A 

V = input('Enter reactor volume (mL): ')
V = float(V)*1e-6

xl = V/A
nx = 1000
D = .002

print('''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ''')
print(' \nCarrying out %s RTD analysis....\n '%str(filename),
'\n*\n***\n*****')

data = pd.read_csv(filename,header=None)
data.columns = ['time','normC']

t_data = data.time[:]
c_data = data.normC[:]

theta_data = t_data/tau_theo

time_theo = np.linspace(0,t_data[len(t_data)-1],nx)
theta_theo = time_theo/tau_theo

def c_curve(time,tau,DuL):
  return (1/(2*np.sqrt(3.14*(DuL))))*np.exp(-(1-(time/tau))**2/(4*DuL))

def f_curve(time,tau,D):
  return 0.5*(special.erf(((time/tau)-1)/(2*np.sqrt(D/u/xl)))+special.erf((1)/(2*np.sqrt(D/u/xl))))

#C = c_curve(theta,DuL)
#F = f_curve(theta,DuL)
#plt.plot(theta,C)
#plt.plot(theta,F)

popt, pcov = curve_fit(f_curve,t_data,c_data,method='lm')
perr = np.sqrt(np.diag(pcov))

plt.figure(1)
plt.plot(t_data,c_data,label = 'RTD Experimental Data', marker="o")
plt.plot(t_data,f_curve(t_data,*popt),label = 'Curve Fit: tau = %s min, D/uL = %s '%(np.round(popt[0],2),np.round(popt[1],8)))
plt.plot(time_theo,f_curve(time_theo,tau_theo,(.01*u*xl)),label='theoretical residence time %s min with DuL = %s '%(tau_theo, np.round(0.01,5)))
plt.ylim(-.05,1.05)
plt.xlabel('time (min)')
plt.ylabel('C/C0(-)')
plt.legend(fontsize=14,edgecolor='black')

plt.figure(2)
E_grad = np.gradient(c_data[:],1)
C_calc = c_curve(t_data,popt[0],popt[1]/u/xl)
F_calc = f_curve(t_data,popt[0],popt[1])
E_calcf = np.gradient(F_calc[:])
E_calc = C_calc/np.trapz(C_calc)
plt.plot(t_data,E_grad, label='E - gradient method')
plt.plot(t_data,E_calc, label='E - analytical method')
plt.plot(t_data,E_calcf, label='E - analytical method step')
plt.legend(fontsize=14,edgecolor='black')
plt.xlim(9,10)

print(' \nD/uL = %s'%(np.round(popt[1],7)))
print(' \nmean residence time = %s min'%(np.round(popt[0],2)))

print('''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ''')
