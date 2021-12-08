# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:54:44 2021

@author: khana
"""

import numpy as np
import scipy.stats

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import math

from drivingforce import Taucorr, force_interp_real, force_interp_imag, t_impulse_array, delt
from drivingforce import f_interp_real, f_interp_imag, t_array_new

from drivingforce import random_force
#import pandas as pd
from scipy import stats
# function that returns dqdt_real and dqdt_imag
def dQ_dt(q, t, om, gam):
    #print(t)
    q_real = q[0]

    q_imag = q[1]

    force_real =0.0

    force_imag=0.0
    
    dqdt_real = (q_imag * om) - (gam * q_real)

    dqdt_imag = -(q_real * om) -(gam * q_imag)

    #for k in range(len(f_interp_real)):

    force_real =  force_interp_real(t)

    force_imag = force_interp_imag(t)

    #want to calculate f((t)

    #for j in range(len(random_force)):

            #time_impulses = random_force[j][0]

            #timediff = t - time_impulses

            #if abs(timediff) > (10*Taucorr) :

              #continue;

            #cj_real = random_force[j][1]  #second column passed as cj_real

            #cj_imag = random_force[j][2]   #third column passed as cj_imag

            # psi_j = math.exp(-(timediff)**2/(2*Taucorr**2))

    #force_real += (cj_real * f_interp_real[k])

   #force_imag += (cj_imag * f_interp_imag[k])

    dqdt_real += -(om * force_imag)

    dqdt_imag += (om * force_real )

    equation = (dqdt_real, dqdt_imag)

    return equation

q0 = [0.001, 0.0004]

lowerb = t_impulse_array[0]


upperb = t_impulse_array[-2]

print(lowerb, upperb)

timesteps = 0.01

tvalues = np.arange(lowerb, upperb, timesteps)
print(tvalues)
om=1

gam=0.01
print('le', len(delt))
Qvalues = odeint(dQ_dt, q0, tvalues, args =( om, gam))

Qvalues_real = Qvalues[:,0]

Qvalues_imag = Qvalues[:,1]

#size, scale = 1000, 10

Energy = (Qvalues_real**2 + Qvalues_imag**2)

#Emod.plot.hist(grid=True, bins=20, rwidth=0.9, color='blue')

#plt.title('Energy mode distribution')

#plt.xlabel('Counts')

#plt.ylabel('Energy mode')

#plt.grid(axis='x', alpha=0.75)
print(t_impulse_array[-2])
print(len(t_impulse_array), len(delt))
print(min(Energy))

print(max(Energy))

delt_E = (max(Energy) - min(Energy))/10

bins=50

fig, axs = plt.subplots(4,1)

# plt.sca(axs[0])
# w = np.linspace(0,5, len(Energy))
# P_w = (Energy/2)* (gam/((w-om)**2 + gam**2))
# plt.plot(w, P_w)
# plt.plot(t_array_new, f_interp_real, 'g-', t_array_new, f_interp_imag, 'b-' )
#
# plt.legend(['f_interp_real', 'f_interp_imag'], loc='best')
#
# plt.xlabel('linearly spaced time')
#
# plt.ylabel('Driven force')




#plt.hist(Emod, bins)
#plt.xlabel('Energy ')
#plt.ylabel('Frequency')
#plt.show()

plt.sca(axs[1])

#plt.plot(tvalues, Qvalues_real ,'g-',linewidth=2, label="Real values of q")

#plt.plot (tvalues, Qvalues_imag, 'y-', linewidth=2, label = "Imaginary values of q")
plt.plot (tvalues, Energy, 'b-', linewidth=2, label = "Energy")
plt.xlabel('time')

plt.ylabel('Energy(t)')
#plt.legend(['Q_real', 'Q_imag', 'Energy'], loc='best')

plt.sca(axs[2])

plt.hist(Energy, bins)
E_avg = np.average(Energy)
plt.plot( Energy, 4000*np.exp(-Energy/ E_avg))

#calculate average energy
plt.xlabel('Energy')

plt.ylabel('Frequency')


plt.sca(axs[3])


plt.plot( Energy, 5000*np.exp(-Energy/ E_avg), 'g-', linewidth='2')
plt.hist(Energy, bins)
plt.xlabel('Energy')
from matplotlib import pyplot
pyplot.yscale('log')
plt.ylabel('frequency')

plt.show()


