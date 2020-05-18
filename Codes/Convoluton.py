'''
CONVOLUTION USING DFT
================================================================
Author: Rounak Chatterjee
Date : 9/05/2020
================================================================

This program computes the convolution of a box function with it self
using discrete convolution of the function computed by using discrete 
fourier transform and to get rid of the non-periodic nature of the function
we zero pad it.

if h(x) is convolution of two functions f(x) and  g(x)

then from the theory of fourier transform we can write

FT(h) = sqrt(2*Pi)*FT(f)*FT(g)

where FT implies the fourier transform of the function.

thus if we define discrete convolution of the function h as
h_q = Sum(r = 0 to n){f_r*g_(q-r)}

where g(x) is considered a periodic function, then we can write this 
discrete convolution formula in terms of DFTs as

h_q = d*sqrt(n)*IDFT(DFT{f(x_p)}*DFT{g(x_p)})_q

where d = sampling length and n is number of samples while IDFT is the
inverse discrete fourier transform.

For our example since the box function is not periodic, we zero pad it.

obviously while obtaining the output, the first half of the IDFT will be our 
requisite output.
'''


import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
from scipy import signal as s 

def f(x): #The rectangular pulse function
	out = np.zeros(len(x),dtype = np.float64)
	for i in range(len(x)):
		if(-1.0<x[i] and x[i]<1.0):
			out[i] = 1.0
	return out


x_min = -2.0 # the effective range of function necessary
x_max = 2.0
n = 512 # number of points(will be doubled while zero padding)
x = np.linspace(x_min,x_max,n)
d = (x_max-x_min)/n 
'''
even though the sampling rate is computed using original n
it will not affect anything when zero padded (that is size doubled minus 1.)
'''
f_x  = np.zeros(2*n-1,dtype = np.float64)
g_x  = np.zeros(2*n-1,dtype = np.float64)
h = np.zeros(2*n-1,dtype = np.float64)
h2 = np.zeros(2*n-1,dtype = np.float64)
f_x[0:n] = f(x)
g_x[0:n] = f(x)
'''
When doing IDFT, since the points are divided into 2n-2 divisions, so we modify d as d/2, while the total number of points is 2n-1
So this normalisation factor is used.

The Actual Answer of the convolution lie in the middle points data set, and we smear this data on our range of x_min to x_max to get our convolution
result.
'''
h = (d/2.0)*np.sqrt(2*n-1)*(ft.ifft(ft.fft(f_x,norm = 'ortho')*ft.fft(g_x,norm = 'ortho'),norm = 'ortho'))
plt.title("DFT CONVOLUTION",size = 15)
plt.plot(x,f_x[0:n],'--',color = 'blue',label = "Function")
# Doing the split in the plotting
plt.plot(np.linspace(x_min,x_max,n-1),h[(n-1)-n//2+1:(n-1)+n//2].real,color = 'red',label = "DFT convolve")
plt.xlabel("x",size = 13)
plt.ylabel("f(x),h(x)",size = 13)
plt.legend()
plt.grid()
plt.show()
