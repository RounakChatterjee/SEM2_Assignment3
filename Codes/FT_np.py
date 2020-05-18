'''
FOURIER TRANSFORM OF SINC(X)
===================================================================
Author : Rounak Chatterjee
Date : 30/04/2020
===================================================================
This program computes the Fourier Transfor of sinc(x) using numpy.fft
package.

To do this we first invoke the computation of fourier transfor of a function 
using DFT

If {f(x_p)} are the samples of the function at sample points {x_p}
where if x_min and x_max are the two end points, then

d = (x_min-x_max)/(n-1), where n is number of sampled points , hence

x_p = x_min + p.d

thus 

f_ft(k_q) = sqrt(n/2*pi)*d*exp(-2*pi*i*k_q*x_min)*DFT[{f(x_p)}]_q

where DFT[{f(x_p)}]_q is the qth component of the DFT of the sampled points
{f(x_p)} and we have considered the frequencies as k_q = q/(nd), in the same
convention as numpy does to make easy computation. thus the original frequency
components are k_q = 2*Pi*q/(nd), where q varies from 0 to (n-1)

we will use the normalized fft along with this scheme to compute the FT
of the given function

Since the FFT algorithm is fastest with 2^m points, we made sure that we 
choose number of points in that way.

the function box(k) is the analytical fourier transform of the sinc(x) function
'''

import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
def f(x): #Given sinc(x) function, can be converted into anythin generic 
	if(x.any() == 0.0):
		return 1.0
	else:
		return (np.sin(x)/x)
def box(k):
	return 0.5*np.sqrt(np.pi/2.0)*(np.sign(k+1)-np.sign(k-1))
n = 1024 # Sampling points		
x_min = -50.0
x_max = 50.0
x = np.linspace(x_min,x_max,n) 
d = (x_max-x_min)/(n-1)
dft_f = ft.fft(f(x),norm = 'ortho')
k =  ft.fftfreq(n,d = d)
k = 2*np.pi*k
phi = np.exp(-1.0j*k*x_min)
ft_f = np.sqrt(n/(2.0*np.pi))*d*phi*dft_f

# reordering arrays for plotting
ft_f = ft.fftshift(ft_f)
k = ft.fftshift(k)


fig = plt.figure(constrained_layout = True)
spec = fig.add_gridspec(1,2)
fig.suptitle("Fourier transform of sinc(x) function",size = 15)
p1 = fig.add_subplot(spec[0,0])
p1.set_title("Configaration space",size = 14)
p1.plot(x,f(x))
p1.set_xlabel("x",size = 13)
p1.set_ylabel("sinc(x)",size = 13)
p1.grid()
p2 = fig.add_subplot(spec[0,1])
p2.set_title("Fourier Space",size = 14)
p2.set_xlabel("frequency modes (k)",size = 13)
p2.set_ylabel("FT(sinc(x))",size = 13)
p2.set_xlim([-5.0,5.0])
p2.plot(k,ft_f.real,'.-',color='blue',label = "Numerical")
p2.plot(k,box(k),color = '#00FF00',label = "Analytical",lw = 3)
p2.legend()
p2.grid()

plt.show()
