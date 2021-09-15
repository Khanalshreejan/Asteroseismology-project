# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:54:44 2021

@author: khana
"""

import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import math

from drivingforce import Taucorr, t2_array, f

from drivingforce import random_force






# function that returns dqdt_real and dqdt_imag
def dQ_dt(q, t, om, gam):
    
    q_real = q[0]
    
    q_imag = q[1]
    
    # print(q[0], q[1])
    
    # print(tvalues)
    
    dqdt_real = (q_imag * om) - (gam * q_real) 
    
    dqdt_imag = -(q_real * om) -(gam * q_imag) 
    
    
    
    
    force_real = 0.0
    
    force_imag = 0.0
    
    

    #want to calculate f((t)
    for j in range(len(random_force)):
        
        
        # time_impulses =[]
        
        # Cj_real =[]
        
        # Cj_imag =[]
        # time_impulse =[]
        
        # diff = np.asarray(t) - np.asarray(tj)
        time_impulses = random_force[j][0]
        
        timediff = t - time_impulses
        
        if abs(timediff) > (10*Taucorr) :
            
          continue;  
        # for i in range (len(random_force[j])):  

        # time_impulses = random_force[j][0] #first column passed as time_impulses
        
        # print(time_impulses)
        
        cj_real = random_force[j][1]  #second column passed as cj_real
        # print(cj_real)
        cj_imag = random_force[j][2]   #third column passed as cj_imag
        
          
        psi_j = math.exp(-(timediff)**2/(2*Taucorr**2))
        
       
        
        force_real += (cj_real * psi_j)
        
        force_imag += (cj_imag * psi_j)
        
        # print(cj_real, cj_imag, psi_j, force_real, force_imag)
    
    # time_impulses.append(time_impulse)
    
    
   
      
    dqdt_real += -(om * force_imag)
    
    dqdt_imag += (om * force_real )
    
    
    
    
  
    equation = (dqdt_real, dqdt_imag)
    
    
    return equation
 

q0 = [0.001, 0.0004]



lowerb = t2_array[0]

upperb = t2_array[-1]

timesteps = 0.1

tvalues = np.arange(lowerb, upperb, timesteps)

om=4

gam=0.3

# f0=2

# omdrive=3
Qvalues = odeint(dQ_dt, q0, tvalues, args =( om, gam))

Qvalues_real = Qvalues[:,0]

Qvalues_imag = Qvalues[:,1]


# plot results
plt.plot(tvalues, Qvalues_real ,'g--',linewidth=2, label="Real values of q")

plt.plot (tvalues, Qvalues_imag, 'b--', linewidth=2, label = "Imaginary values of q")

plt.xlabel('time')
plt.ylabel('q(t)')
plt.legend()
plt.show()