# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:54:44 2021

@author: khana
"""

import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import math

from drivingforce import timediff, Taucorr, Psi_j, t2_array

from drivingforce import random_force






# function that returns dqdt_real and dqdt_imag
def dQ_dt( q, t, om=4, gam=0.3):
    
    q_real = q[0]
    q_imag = q[1]
    dqdt_real = (q_imag * om) - (gam * q_real) 
    
    dqdt_imag = -(q_real * om) -(gam * q_imag) 
    
    
    
    
    force_real = 0.0
    
    force_imag = 0.0
    
    

    #want to calculate f((t)
    for j in range(len(random_force)):
        
        # tj =  random_force[j]['time_impulse']
        tj =0
        cj_real =0
        cj_imag =0
        if abs(t-tj) > 10*Taucorr :
          continue;  
        # cj_real = random_force[j]['real']
        
        # cj_imag = random_force[j]['imag']
        
        Psi_j = math.exp(-(t - tj)**2/(2*Taucorr**2))
        
        force_real += (cj_real * Psi_j)
        
        force_imag += (cj_imag * Psi_j)
        
   
    
    dqdt_real += -(om * force_imag)
    
    dqdt_imag += (om * force_real )
    
    equation =[dqdt_real, dqdt_imag]
    
    return dqdt_real, dqdt_imag

q0 = [0.001,0.00002]

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
plt.plot(tvalues, Qvalues_real ,'r-',linewidth=2)

plt.plot (tvalues, Qvalues_imag, 'b', linewidth=2)

plt.xlabel('time')
plt.ylabel('q(t)')
plt.legend()
plt.show()