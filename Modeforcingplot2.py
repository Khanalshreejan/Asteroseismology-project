

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 03:41:30 2021

@author: khana
"""

import numpy as np

import matplotlib.pyplot as plt

import math



# def Modeforcing(cj, Si):
#     equation1 = sum(eq1*eq2)
    



# # def Gaussian(t1_array,t2_array, Tco,freqmax,forcing,cj,k,Ej_array,Ea):
    
#     return equation1

alpha =1

Modeforcing_array=[0]

Ej_array=[0]

Ej_array = np.random.exponential(1,301)
    
Ea_array =np.mean(Ej_array)
    
Ya = np.random.random(301)
    
freqmax = np.random.random(301)

# foa_array = np.random.random(301)

Tco_array = []

for f in freqmax:
    Tco_array.append(1/(2*math.pi*f)) 
    #Equation for corelation time
    


mu = 5.0

inv_mu =1/mu

t1_array =[0]

Sum_array = [0]

pden_array = [0]

t2_array =[0]

cj_array = [0]

product_array=[0]

si_array = [0]

Modeforcing_array2 =[0]

ModeforceSum_array=[0]


delt = np.random.exponential(inv_mu,300)

for i in range(len(delt)):  #loop runs for every index i of list delt
    
    #t2_array and t1_array lengths are dependent of length of delt, so use i to access their elements each time 
    #the loop runs
    
    if t2_array[i]< 500:
        
        t2_array.append(delt[i] + t1_array[i]) #t2 = delt + t1 
        
        t1_array.append(t2_array[len(t2_array)-1])  #t1 = previous t2 as last element of t2 before the next loop iterates
        
        pden_array.append(mu * math.exp(-mu * delt[i])) 

# print(len(Ej_array),Ea_array)

for i in range(len(delt)):
    
          if freqmax[i] < 25 and Ya[i] <15:
              
            cj_array.append(((Ej_array[i]/Ea_array)**alpha)* Ya[i]) 
            #Equation for complex_amplitude
            
            si_array.append(math.exp(-((t1_array[i]-t2_array[i])**2)/(2*(Tco_array[i]**2))))
            #Equation for Gaussian time dependence
            
            
            product_array.append(cj_array[i] * si_array[i])
           
            Modeforcing_array2.append(product_array[i] + Modeforcing_array[i])
            
            # ModeforceSum_array.append(product_array[i] + Modeforcing_array2[i])
           
            Modeforcing_array.append(Modeforcing_array2[len(Modeforcing_array2)-1]) 
            #Modeforcing1 = previous Modeforcing2 as the last element of Modeforcing2 before the next loop iterates
            
            
fig, ax_left = plt.subplots()

ax_right = ax_left.twinx()

ax_left.plot(t2_array, Modeforcing_array2, color='red', linewidth=2)

ax_right.plot(t2_array, pden_array, color='blue',  linewidth=2)

ax_left.set_ylabel('Modeforcing', color = 'r')

ax_left.set_xlabel('time in minutes')

ax_right.set_ylabel('Impulses at radnom time interval ', color = 'b')

plt.title("Mode forcing plot")

plt.show()
