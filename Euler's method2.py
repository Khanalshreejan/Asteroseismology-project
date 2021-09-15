# -*- coding: utf-8 -*-
"""


@author: khana
"""
import matplotlib.pyplot as plt

import numpy as np

def f(x,y):
    
    dydx=  np.sin(3*x)
    
    return dydx
x=1

y=0

x_array=[1]

y_array=[0]

xn=20

n=200

stepsize=(xn-x)/n

while x < xn:
    y= y + stepsize*f(x,y)
    
    x= x + stepsize
    
    x_array.append(x)
    
    y_array.append(y)
    
    # print(x,y)
    
plt.plot(x_array,y_array, 'r-', linewidth=2, label='Eulers method')

plt.xlabel("x value")

plt.ylabel("y value")

plt.legend()

plt.title("Solving ODE using Euler's method")



#Odeint
import numpy as np #base package imported for various functions
from scipy.integrate import odeint #integrate package imported as odeint function
import matplotlib.pyplot as plt #for plotting the numerical integration


def model(y,x):
 
  dydx = np.sin(3*x)#to input variables y and x and return a slope dy/dx. Can also change the function for dxdt accordingly
 
  return dydx


y0 = 0 #provide initial value

timesteps=100

x = np.linspace(1,20,timesteps) #define range of x as from start to end and 30 divisions
#linspace funtion from numpy to create linearly spaced points

y = odeint(model,y0,x)



  #put three different arguments
#model to return derivative values at different x and t as dxdt
#x0 as initial condition for different states
#t to lay out evenly distributed time points


# plt.plot(x,abs(y_array)/y, 'b--', label='m=-0.2')
# print(x,(y-y_array))
plt.plot(x,y, 'b-', label='Odeint')

plt.xlabel('x value')
plt.legend()

plt.ylabel('y(x)')

plt.show()


