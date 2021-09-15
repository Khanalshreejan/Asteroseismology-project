# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:30:36 2021

@author: khana
"""

import numpy as np

import matplotlib.pyplot as plt

import math
omega= 1

mu = (10*omega)/(2*math.pi) #10 impulses per oscillation period

inv_mu =1/mu

t1_array =[0]

# Sum_array = []

# pden_array = [0]

t2_array =[0]

# t_end = 30* ((2*math.pi)/omega)


delt = np.random.exponential(inv_mu,500)

t1 = 0

impulse_index =[0]

for i in range(len(delt)):
    #loop runs for every index i of list delt
    
    #t2_array and t1_array lengths are dependent of length of delt, so use i to access their elements each time 
    #the loop runs
    
    # if t2_array[i]< t_end:
        
        # t2_array.append(delt[i] + t1_array[i]) #t2 = delt + t1 
        t2_array.append(delt[i]+ t1)
        
        impulse_index.append(len(t2_array))
        
        t1 = t2_array[-1]
        # t1_array.append(t2_array[len(t2_array)-1])   #t1 = previous t2 as last element of t2 before the next loop iterates
        
        # pden_array.append(mu * math.exp(-mu * delt[i]))  #probability density equation
        
        # Sum_array.append(sum(pden_array))  #just to see how the sum looks? looks exponential in increment
        
t_end= t2_array[-1]
# x_array=[np.linspace(0,len(t2_array)-1,1)]
# print("Sum =", sum(pden_array))

  #plot both pden and sum into the same graph but with different scales 
# plt.plot(impulse_index,t2_array, 'b-', linewidth =1, markersize=0.5)   
# plt.show() 
# fig, ax_left = plt.subplots()

# t=0
timediff= 0


Sum =0

sum_array =[]

Psi_j_array=[]

Psi_j=0

Taucorr = (1/omega)

t_array= np.linspace(0, t_end, 1000)


for i in range (len(t_array)):
    
    Sum =0
    
    for j in range (len(t2_array)):
        
        timediff = t_array[i] -t2_array[j]
        
        Psi_j = math.exp(-(timediff**2)/(2*(Taucorr**2)))
        
        Psi_j_array.append(Psi_j) 
    
        Sum = Sum + Psi_j
    
    sum_array.append(Sum)
        
    # sum_array[i] = Psi_j_array[j] + sum_array[i]
        
    # sum_array.append(Sum)    

plt.plot(t_array, sum_array, 'r-', linewidth =2)    
# plt.plot(impulse_index, t2_array, 'b.', markersize=0.5)
# plt.show()
      
    
    
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

    