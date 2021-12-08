# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 23:35:13 2021

@author: khana
"""
#%%
from typing import Union, Any

import numpy as np

import matplotlib.pyplot as plt

import math

import sys

from scipy.interpolate import interp1d

omega= 1

mu = (10*omega)/(2*math.pi) #10 impulses per oscillation period

inv_mu = 1/mu

t1_array =[0]

t_impulse_array =[]

# t_end = 30* ((2*math.pi)/omega)
sample=500
delt = np.random.exponential(inv_mu,sample)
#delt: Union[Union[int, float, complex], Any] = np.random.exponential(inv_mu,1000)
t1 = 0

impulse_index =[0]

Cj_real_array=[]

Cj_imag_array=[]



for i in range(len(delt)):
    #loop runs for every index i of list delt
    
    #t2_array and t1_array lengths are dependent of length of delt, so use i to access their elements each time 
    #the loop runs
    
    # if t2_array[i]< t_end:
        
        # t2_array.append(delt[i] + t1_array[i]) #t2 = delt + t1 
        t_impulse_array.append(delt[i]+ t1)
        
        #t_impulse_j = delt[i] + t1
        
        impulse_index.append(len(t_impulse_array))
        
        t1 = t_impulse_array[-1]
        # t1_array.append(t2_array[len(t2_array)-1])   #t1 = previous t2 as last element of t2 before the next loop iterates
        
        # pden_array.append(mu * math.exp(-mu * delt[i]))  #probability density equation
        
        # Sum_array.append(sum(pden_array))  #just to see how the sum looks? looks exponential in increment
        phi = np.random.uniform(0, 2 * math.pi)

        cos_phi = math.cos(phi)

        sin_phi = math.sin(phi)

        cos_theta = np.random.uniform(-1, 1)

        plm_cos_theta = cos_theta

        E_lin = 1

        Eavg = E_lin

        epsilon = np.random.exponential (Eavg)

        Ij = 2 *(epsilon/Eavg)

        cj_real = Ij * plm_cos_theta * cos_phi

        cj_imag = Ij * plm_cos_theta * sin_phi
        
        Cj_real_array.append(cj_real)
        
        Cj_imag_array.append(cj_imag)
        
    
        
t_end= t_impulse_array[-1]

timediff= 0

sum_array =[]

Psi_j_array=[]

Psi_j=0

Taucorr =1*(1/omega)

t_array= np.arange(0, t_end, 0.1)

random_force =[]

force_real_data =[]

force_imag_data =[]

t_impulse_j=0.0

for i in range (len(t_array)):
    
    Sum =0

    force_real_at_t = 0.0

    force_imag_at_t =0.0

    for j in range (len(t_impulse_array)):
        
        #t_impulse_j  = t_impulse_array
        
        timediff = t_array[i] - t_impulse_array[j]

        Psi_j = math.exp(-(timediff**2)/(2*(Taucorr**2)))
        
        Psi_j_array.append(Psi_j)
        
      
        #print(cj_real)
    #comlex_amplitude calculation    

        
        #random_force_j = {"time_impulse" :t2_array[j],"real": cj_real,"imag" : cj_imag}
        #impulse_j = [t_impulse_array[j], cj_real, cj_imag]
     
        # print (t2_array[j], cj_real, cj_imag)
        # print (Ij, plm_cos_theta, cos_phi, sin_phi)
        # sys.exit()
        #random_force.append(random_force_j)

        #Sum = Sum + Psi_j

        force_real_at_t +=  Cj_real_array[j] * Psi_j

        force_imag_at_t += Cj_imag_array[j] * Psi_j

    force_real_data.append(force_real_at_t)

    force_imag_data.append(force_imag_at_t)

#sum_array.append(Sum)
    
#t_array_new = np.arange(0, t_end,0.1)

force_interp_real= interp1d(t_array, force_real_data, kind='linear')

force_interp_imag: interp1d = interp1d (t_array, force_imag_data, kind ='linear')
print(t_array[0], t_array[-1])
#def convert(f_interp_real):

    #return tuple(f_interp_real)
t_array_new = np.arange(0, t_end,0.1)
f_interp_real = force_interp_real(t_array_new)
#
# #def convert(f_interp_imag):
#
#     #return tuple(f_interp_imag)
#
# #for i in range(len(f_interp_real)):
#
#     #interp_forces = [f_interp_real[i], force_interp_imag[i]]
#
f_interp_imag = force_interp_imag(t_array_new)
#
# #plt.plot( t_impulse_array, impulse_index,  'r-', linewidth=1)
#plt.plot(t_array,  force_real_data,  'r.', t_array, force_imag_data, 'y.')
plt.plot(t_array_new, f_interp_real, 'g-', t_array_new, f_interp_imag, 'b-' )

plt.legend(['f_interp_real', 'f_interp_imag', 'linear'], loc='best')
#
plt.xlabel('linearly spaced time')
#
plt.ylabel('Driven force')
#
plt.show()




# for i in range(len(t_array)):
    
#     timediff = t_array[i] - t2_array[i]
    
    # Psi_j_array = math.exp(-(timediff**2)/(2*(Taucorr**2)))
    # Psi_j_array.append(Psi_j) 
    
    # Sum = Sum + Psi_j
    
    # sum_array.append(Sum)
    
# plt.plot(t_array, sum_array, 'r-', linewidth =2)    
# plt.show()
    
    
    
        
    
    
    
# fig, ax_left = plt.subplots()  

# ax_right = ax_left.twinx()

# ax_left.plot(impulse_index, t2_array, 'r.', linewidth=2, markersize=0.5)

# ax_right.plot(t_array, sum_array, 'b.',  linewidth=2, markersize=0.5)

# # plt.plot(t2_array, pden_array, 'g-', linewidth =2, label='mu = 5')

# # plt.plot (t2_array, Sum_array,'y-', linewidth =2, label='Sum of forces')

# ax_left.set_ylabel('impulse at corresponding time', color = 'r')

# ax_left.set_xlabel('time in minutes')

# ax_right.set_ylabel('Sum of the impulses', color = 'b')

# plt.legend()

# plt.title("impulse generated at random time intervals")

# plt.show()

# import numpy as np

# import matplotlib.pyplot as plt

# import math

# mu = 5

# inv_mu =1/mu

# t1_array =[0]

# # Sum_array = [0]

# # pden_array = [0]

# t2_array =[0]

# delt = np.random.exponential(inv_mu,100)

# t1 = 0

# impulse_index =[0]

# for i in range(len(delt)):
#     #loop runs for every index i of list delt
    
#     #t2_array and t1_array lengths are dependent of length of delt, so use i to access their elements each time 
#     #the loop runs
#      if t2_array[i]< 500:
        
#         # t2_array.append(delt[i] + t1_array[i]) #t2 = delt + t1 
#         t2_array.append(delt[i]+ t1)
        
#         impulse_index.append(len(t2_array))
        
#         t1 = t2_array[-1]
#         # t1_array.append(t2_array[len(t2_array)-1])   #t1 = previous t2 as last element of t2 before the next loop iterat
        
# # plt.plot(impulse_index,t2_array, 'b-', linewidth =1, markersize=0.5) 

# timediff= 0

# Sum =0

# sum_array =[]

# Taucorr =inv_mu

# t_array= np.linspace(0, 100, 100 )

# for i in range(len(t_array)):
    
#     timediff = t_array[i] - t2_array[i]
    
#     Psi = math.exp(-(timediff**2)/(2*(Taucorr**2)))
    
#     Sum = Sum + Psi
    
#     sum_array.append(Sum)
    
    
# # plt.plot(t_array, sum_array, 'r-', linewidth =2)    
# # plt.show()
# fig, ax_left = plt.subplots()

# ax_right = ax_left.twinx()

# ax_left.plot(impulse_index, t2_array, 'r.', linewidth=2, markersize=0.5)

# ax_right.plot(t_array, sum_array, 'b.',  linewidth=2, markersize=0.5)

# ax_left.set_ylabel('t2', color = 'r')

# ax_left.set_xlabel('impulse index and time')

# ax_right.set_ylabel('Sum of Si_j', color = 'b')

# # plt.legend()

# plt.title("Impulse Index and Si_j plot")

# # plt.show()