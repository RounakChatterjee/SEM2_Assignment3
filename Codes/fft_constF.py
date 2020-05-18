'''
FOURIER TRANSFORM OF CONST FUNCTION
=======================================================================================
Author : Rounak Chatterjee
Date : 04/05/2020
=======================================================================================
This Module computes the fourier transfor of a constant function using numpy's
FFT module

we know that to do FT with FFT we do the following:

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
of the given function.

Now ideally a constant function must yield a dirac delta function, we can find that as we increase the x range 
pf constant function, in the fourier space it keeps on increasing in magnitude.
 to depict it we have plotted three functions 
'''

import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
def FT_const(x_min,x_max):
	n = 512
	ft_f = np.zeros(n,dtype = 'complex')
	d = (x_max-x_min)/(n-1)
	#Even though the effective portion of th function extends from -inf to inf, but still we restrict our self to a finite region for convergence.
	x = np.linspace(x_min,x_max,n)
	k = 2*np.pi*ft.fftshift(ft.fftfreq(n,d = d))
	f = np.linspace(1.0,1.0,len(x))
	ft_f = np.sqrt(n/(2*np.pi))*d*np.exp(-1.0j*k*x_min)*ft.fftshift(ft.fft(f,norm='ortho'))
	return [k,ft_f]

ft1 = FT_const(-1000.0,1000)
ft2 = FT_const(-1.0e5,1.0e5)
ft3 = FT_const(-1.0e7,1.0e7)
ft4 = FT_const(-1.0e8,1.0e8)

fig = plt.figure(constrained_layout = True)
spec = fig.add_gridspec(2,3)
fig.suptitle("Fourier transform of constant function f(x) = 1.0",size = 15)
p1 = fig.add_subplot(spec[:,0])
p1.set_title("Configaration space",size = 14)
p1.plot(np.linspace(-1.0e8,1.0e8,2),np.linspace(1.0,1.0,2))
p1.set_xlabel("x",size = 13)
p1.set_ylabel("f(x) = 1",size = 13)
p1.grid()

p2 = fig.add_subplot(spec[0,1])
p2.set_title("Fourier Space",size = 12)
p2.set_xlabel("frequency modes (k)")
p2.set_ylabel("FT(f(x) = 1.0)")
p2.plot(ft1[0],ft1[1],color = 'blue',label = "x$_{min}$ = -1000.0, x$_{max}$ = 1000.0")
p2.legend()
p2.grid()

p3 = fig.add_subplot(spec[0,2])
p3.set_title("Fourier Space",size = 12)
p3.set_xlabel("frequency modes (k)")
p3.set_ylabel("FT(f(x) = 1.0)")
p3.plot(ft2[0],ft2[1],color = 'red',label = "x$_{min}$ = -1.0 X 10$^5$, x$_{max}$ = 1.0 X 10$^5$")
p3.legend()
p3.grid()

p4 = fig.add_subplot(spec[1,1])
p4.set_title("Fourier Space",size = 12)
p4.set_xlabel("frequency modes (k)")
p4.set_ylabel("FT(f(x) = 1.0)")
p4.plot(ft3[0],ft3[1],color = 'green',label = "x$_{min}$ = -1.0 X 10$^7$, x$_{max}$ = 1.0 X 10$^7$")
p4.legend()
p4.grid()

p5 = fig.add_subplot(spec[1,2])
p5.set_title("Fourier Space",size = 12)
p5.set_xlabel("frequency modes (k)")
p5.set_ylabel("FT(f(x) = 1.0)")
p5.plot(ft4[0],ft4[1],color = 'black',label = "x$_{min}$ = -1.0 X 10$^8$, x$_{max}$ = 1.0 X 10$^8$")
p5.legend()
p5.grid()



plt.show()

	
