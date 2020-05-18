'''
DFT VS FFT
======================================================================
Author : Rounak Chatterjee
Date : 04/05/2020
======================================================================
This program computes the DFT of n numbers using the oot DFT algorithm and using the
FFT algorithm from numpy as compares the time difference between the two.

To fo the DFT we use the standard DFT alorithm for n points as:

If {f_p} are a set of n discrete points, then the the discrete fourier transform of these n points are given by
another set of n points {dft_f_q}, such that

dft_f_q = (1/sqrt(n))*Sum(p = 0 to n-1){exp(-j2*Pi*p*q/n)*f_p}

the fast fourier transform of th n points is calculated using the numpy fft algorithm.

We do this for n points from 1 to n and vary n to get an estimate of the time taken.
'''

import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
import time as t

def dft(x):
	ft_x = np.zeros(len(x),dtype = 'complex')
	
	for q in range(len(x)):
		for p in range(len(x)):
			ft_x[q] = ft_x[q] + np.exp(-1.0j*2.0*np.pi*p*q/len(x))*x[p]
		ft_x[q] = (1.0/np.sqrt(len(x)))*ft_x[q]
	return ft_x
n = np.arange(4,101,1)
t_dft = np.zeros(len(n),dtype = np.float64)
t_fft = np.zeros(len(n),dtype = np.float64)
for i in range(len(n)):
	x = np.linspace(1.0,n[i],n[i])
	tm = t.time()
	dft(x)
	t_dft[i] = t.time()-tm
	tm = t.time()
	ft.fft(x)
	t_fft[i] = t.time()-tm
plt.plot(n,t_dft,'ro',marker = 'd',color = 'blue',label = "DFT time")
plt.plot(n,t_fft,'ro',marker = 'x',color = 'red',label = "FFT time")
plt.title("Comparison of time taken by DFT and FFT",size = 15)
plt.xlabel("Number of points n")
plt.ylabel("Time taken(in s)")
plt.legend()
plt.grid()
plt.show()




