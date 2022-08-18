# =============================================================================
# 
# Plug-Flow-Reactor Model
# Written by: Cameron Armstrong (2022)
# Coupled mass and energy equations for reacting flow
# Explicit FDM solver
#
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
import math

np.seterr(divide='ignore')

# =============================================================================
# Defining the PFR class
# =============================================================================

class PFR():
    
    def __init__(self,nx,dt,termcond,reltol):
        self.nx = nx
        self.dt = dt
        self.termcond = termcond
        self.reltol = reltol
        self.tolcheck = np.ones(self.nx)
        self.domain = np.array(self.nx)
        
    def reaction(self,Q1,Q2,C01,C02,Twall,Ea,k0):
        self.Q1 = Q1
        self.Q1 = (self.Q1*10**-6)/60
        self.Q2 = Q2
        self.Q2 = (self.Q2*10**-6)/60
        self.Q = self.Q1+self.Q2
        self.C01 = C01*self.Q1/self.Q
        self.C02 = C02*self.Q2/self.Q
        self.Twall = Twall+273.15
        self.Ea = Ea
        self.k0 = k0
        self.R = 8.314
        
    def reactor(self,V,ID,Dax):
        self.V = V*10**-6
        self.ID = ID/39.37
        self.Ac = math.pi*(self.ID/2)**2
        self.Dax = Dax
        
        self.L = self.V/self.Ac
        self.uax = self.Q/self.Ac
        self.DuL = self.Dax/self.uax/self.L
        self.Bo = 1/self.DuL
        
    def phys_props(self,k,Cp,rho,Nu,dHr):
        self.k = k
        self.Cp = Cp
        self.rho = rho
        
        self.Nu = Nu
        self.h = self.Nu*self.k/self.ID
        self.alpha = self.k/self.Cp/self.rho
        self.lam = self.alpha*self.dt/self.dx**2
        self.dHr = dHr
        
    def grid(self):
        self.domain = np.linspace(0,self.L,self.nx)
        self.dx = self.L/(self.nx-1)
        self.cfl = self.uax*self.dt/self.dx
        
        print('CFL Condition = %f'%self.cfl)
        
        self.C1 = np.ones(self.nx)
        self.C2 = np.ones(self.nx)
        self.prod = np.zeros(self.nx)
        
        self.C1_temp = np.ones(self.nx)
        self.C2_temp = np.ones(self.nx)
        self.prod_temp = np.zeros(self.nx)
        
        self.T = np.ones(self.nx)*self.Twall
        self.T_temp = self.T.copy()
        
        self.k_f = np.ones(self.nx)
        self.k_f[:] = self.k0*np.exp(-self.Ea/self.R/self.T[:])
        self.k_f_temp = self.k_f.copy()
    
    def check_yourself(self,old,new):
       return np.abs(((np.linalg.norm(old)-np.linalg.norm(new)))/np.linalg.norm(new))
            
    def C_UDS_solver(self,main,s1,s2,stoic):
        return main[1:-1] -(self.uax)*(self.dt/(self.dx))*(main[1:-1]-main[:-2]) \
                +self.Dax*self.dt/self.dx**2*(main[2:]-2*main[1:-1]+main[:-2])   \
                +stoic*self.k_f[1:-1]*s1[1:-1]*s2[1:-1]*self.dt
    
    def T_UDS_solver(self,T):
        return T[1:-1]-(self.uax*(self.dt/self.dx)*(T[1:-1]-T[:-2]))    \
                +self.lam*(T[2:]-2*T[1:-1]+T[:-2])                      \
                -self.h*self.ID*math.pi*(T[1:-1]-self.Twall)            \
                *self.dt/self.rho/self.Cp*self.L/self.V                 \
                -self.dHr*self.k*self.C1[1:-1]*self.C2[1:-1]            \
                /self.rho/self.Cp*self.dt

# =============================================================================
# Solving the system and plotting the result 
# =============================================================================

#Setup Parameters

nx = 100
dt = 10/nx
termcond = 1.0
reltol = 1e-8

# Reaction Parameters
Q1 = 2.5
Q2 = 2.5
C01 = 500
C02 = 400
Twall = 50
Ea = 7e4
k0 = 5e7

# Reactor Parameters
V = 10
ID = 0.0625
Dax = 0.005

# Phys Props
k= .2501 #W/(m*K)
rho = 786 #(kg/m^3)
Cp = 2200 #(J/kg/K)
Nu = 3.66
dHr = -5000

# Running the model

R1 = PFR(nx,dt,termcond,reltol)
R1.reaction(Q1,Q2,C01,C02,Twall,Ea,k0)
R1.reactor(V,ID,Dax)
R1.grid()
R1.phys_props(k,Cp,rho,Nu,dHr)

while R1.termcond >= R1.reltol: #time loop
        
        R1.termcond = R1.check_yourself(R1.tolcheck,R1.prod_temp)
        
        R1.T[0] = R1.Twall
        R1.T[R1.nx-1]=R1.T[R1.nx-2]
        
        R1.T_temp = R1.T.copy()
        
        R1.T[1:-1] = R1.T_UDS_solver(R1.T)
        
        R1.k_f[:] = R1.k0*np.exp(-R1.Ea/R1.R/R1.T[:])
        
        R1.C1_temp = R1.C1.copy()
        R1.C2_temp = R1.C2.copy()
        R1.prod_temp = R1.prod.copy()
        
        R1.C1[0] = R1.C01
        R1.C2[0] = R1.C02
        
        R1.C1[R1.nx-1] = R1.C1[R1.nx-2] 
        R1.C2[R1.nx-1] = R1.C2[R1.nx-2] 
        R1.prod[R1.nx-1] = R1.prod[R1.nx-2] 
        
        R1.C1[1:-1] = R1.C_UDS_solver(R1.C1_temp,R1.C1_temp,R1.C2_temp,-1)
        R1.C2[1:-1] = R1.C_UDS_solver(R1.C2_temp,R1.C1_temp,R1.C2_temp,-1)
        R1.prod[1:-1] = R1.C_UDS_solver(R1.prod_temp,R1.C1_temp,R1.C2_temp,+1)
        
        R1.tolcheck = R1.prod.copy()
        

plt.rcParams["font.family"] = "sans-serif"
plt.rc('axes',labelsize=18)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('figure',titlesize=10)
plt.rc('legend',fontsize=16)            

f, (ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=False)
ax1.scatter(R1.domain/R1.L,R1.prod,label='Product')
ax1.scatter(R1.domain/R1.L,R1.C1,label='Reagent1')
ax1.scatter(R1.domain/R1.L,R1.C2,label='Reagent2')
ax1.legend(loc=7)
ax1.set_ylabel(r'Molar Concentration $(\frac{mol}{m^3})$') 

ax2.scatter(R1.domain/R1.L,R1.T-273.15)
ax2.set_ylabel('Reactor Temperature ($^\circ$C)')
ax2.set_xlabel('Normalized Reactor Length (-)')


