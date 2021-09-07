# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:45:50 2021

@author: khana
"""

import numpy as np #base package imported for various functions

from scipy.integrate import odeint #integrate package imported as odeint function

import matplotlib.pyplot as plt #for plotting the numerical integration

import math

from drivingforce import Sum

from drivingforce import random_force

from drivingforce import Taucorr, timediff

from drivingforce import sum_array
#y'' + 3y' +2y = x + sin(x) , y(0)=1, y'(0)=0
#To turn this into system of 1st order ODEs, introduce new parameters as  y=a and y'=b
#then a'= b .......eqn1
#b' = 2a -b+x+cos(x) ............eqn2
#function dG_dx = (b, 2a-3b+x+sin(x))
#In array form G(a,b) = (G[0], G[1])

def dG_dt(G, t,om,gam,f0,omdrive,Sum):
    
    
    # equation = [G[1], -2*G[0] - 3*G[1] + x +  np.sin(x)] #for two first order ODE equations
    y=G[0]
    
    dqdt=G[1]
    
    d2qdt2 = -(om**2) * y - gam * dqdt #damped harmonic osscilator
  
    force_real = 0.0
    
    force_imag = 0.0
    
    d2qdt2_real= 0
    
    d2qdt2_imag= 0
    
    dqdt_real =0
    
    dqdt_imag =0

    #want to calculate f((t)
    for j in range(len(random_force)):
        
        tj =  random_force[j]['time_impulse']
        
        amp_real = random_force[j]['real']
        
        amp_imag = random_force[j]['imag']
        
        psi_j = math.exp(-(timediff - tj)^2/(2*taucorr^2))
        
        force_real += amp_real * psi_j
        
        force_imag += amp_imag * psi_j
        
    # drivingf= sum_array[-1]
    
    d2qdt2_real += force_real
    
    d2qdt2_imag += force_imag
    
   # d2ydt2 = d2ydt2 + drivingf
    
    # drivingf = sum_array
    # drivingf_array = []
    
    # for i in range(len(sum_array)):
    #     drivingf_array.append(sum_array[i]+ d2ydt2)
    
    # d2ydt2 = d2ydt2 +  drivingf
    
    equation = [dqdt, dq2dt2]
    
    return equation

Ginitial = [1, 0]  #Introduced initial values i.e a= y(0)=1 and b =y'(0)=0

#boundaries to integrate from and to

lowerb=2

upperb=100

timesteps = 0.1

tvalues = np.arange(lowerb, upperb, timesteps)

om=4

gam=0.3

f0=2

omdrive=3

Gvalues = odeint(dG_dt, Ginitial, tvalues, args=(om,gam,f0,omdrive,Sum)) #integrate a function dG_dx wrt x 
#This will return 2 values for the two 1st order ODEs function above

yvalues = Gvalues[:,0] #take all rows in the second entry of integrated Gvalues to store yvalue


plt.xlabel("t values")

plt.ylabel("y values/ y(t)")

plt.title("Solution to second order Differential Equation")

plt.plot(tvalues,yvalues, 'b-', linewidth=0.5, markersize=0.5)
