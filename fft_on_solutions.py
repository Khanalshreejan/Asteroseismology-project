import numpy as np

import matplotlib.pyplot as plt

from first_order_approach import Qvalues_real, Qvalues_imag, Energy,om, gam
from drivingforce import t_impulse_array

plt.rcParams['figure.figsize'] =[10,5]

plt.rcParams.update({'font.size': 18})

#create a signal with two frequencies
N=len(Qvalues_real)
print('len' ,len(Qvalues_real))
omega= 1

t_final =t_impulse_array[-2]

t_initial = t_impulse_array[0]
#dt = len(Qvalues_real) #samples per second
dt = (t_final - t_initial)/N

t=np.linspace(t_initial,t_final,N) #linearly spaced data set that goes from 0 to N with dt samples per second

#print(len(Qvalues_real))
f1 = Qvalues_real

f2 = Qvalues_imag

#f2 = 3.0 * np.cos(2*omega*t)
#f_Sum=Qvalues_real+Qvalues_imag #Sum of two frequencies

#f_clean = f_Sum

#f_noisy = f_Sum + 2.5 * np.random.randn(len(t))  #Gaussian noise

# plt.plot(t, f_noisy, color='red', linewidth=1.5, label= 'Noisy')
#
# plt.plot(t, f_clean, color='black', linewidth=1, label='Clean')
#
# plt.xlim(t[0], t[-1])
#
# plt.legend()
#
# plt.show()

#compute fast fourrier transform
N1 = len(t)

fhat1 = np.fft.fft(f1, N1) #fft of the f_Sum

fhat2 = np.fft.fft(f2, N1)



Power_spec_den1 = fhat1 * np.conj(fhat1) /N1 #Power density per frequency

#Suppose q = a+ ib , then q_conjug = a-ib, then q * q_conjug = a^2 + b^2  have units of Power

Power_spec_den2 = fhat2 * np.conj(fhat2) /N1



freq = (1/(dt*N1)) * np.arange(N1) #build a vector of all frequencies (from low to high frequencies)

L = np.arange(1, np.floor(N1/2), dtype ='int') #create an array of only first half of the values, So from 1 to n/2

fig, axs = plt.subplots(3,1)

plt.sca(axs[0])
w = np.linspace(0,5, len(Energy))
P_w = 6000*(gam/((w-om)**2 + gam**2))
plt.plot(w, P_w)
#plt.plot(freq[L]*2*np.pi, Power_spec_den2[L],color='black', linewidth= 1.5, label='Power Spectral Density_Energy')

plt.xlim(freq[L[0]], freq[L[-1]])
#plt.plot(t, Qvalues_real, label='Q_real')

#plt.plot(t, Qvalues_imag, label='Q_imag')

# plt.xlim(t[0], t[-1])
plt.xlabel('Frequency Hz')
plt.ylabel('power spectrum')
plt.legend(loc= 'best')

plt.sca(axs[1])

plt.plot(t, Qvalues_real, color='y', linewidth =1.5,)

plt.plot(t, Qvalues_imag, color='r', linewidth =1.5,)


#plt.plot(t, f_noisy, color='r', linewidth =1.5, label='Noisy')

plt.xlim(t[0], t[-1])

plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('Amplitude q(t)')
# indices = Power_spec_den >100 #find frequencies with larger power
#
# Power_spec_den_clean = Power_spec_den * indices  #zero out all other frequencies with smaller power
#
# fhat = indices*fhat #zero out smaller fourrier coefficients
#
# f_filter = np.fft.ifft(fhat) #inverse fft for filtered time signal

plt.sca(axs[2])
#w = np.linspace(0,5, len(Energy))
#P_w = 1000*(gam/((w-om)**2 + gam**2))
#plt.plot(w, P_w)
plt.plot(freq[L]*2*np.pi, Power_spec_den1[L],color='g', linewidth= 1.5, ) # Power spectrum i.e magnitude of fhat ^2 vs the frequencies

#plt.plot(freq[L]*2*np.pi, Power_spec_den2[L],color='b', linewidth= 1.5, )

plt.xlim(freq[L][0], freq[L][-1])
plt.xlabel('frequency Hz')
plt.ylabel('Power Spectrum')
plt.legend()

plt.show()





