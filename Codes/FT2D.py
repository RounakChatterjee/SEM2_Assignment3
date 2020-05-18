'''
FT OF 2D GAUSSIAN
======================================================================
Author: Rounak Chatterjee
Date : 04/05/2020
=======================================================================
This progra computes the 2D fourier transform of a Gaussian function.
In this case we can modify the algorithm as :

If {f(x_p,y_q)} are the samples of the function at sample points {x_p,y_q}
where if [x_min,x_max] and [y_min,y_max] are the two ranges, then

dx = (x_max-x_min)/(nx-1) and dy = (y_max-y_min)/(ny-1), where nx*ny is number of sampled points , hence

x_p = x_min + p.dx
y_q = y_min + q.dy


thus 

f_ft(kx_r,ky_s) = (nx*ny)/2*pi*dx*dy*exp(-2*pi*i*kx_r*x_min)exp(-2*pi*i*ky_s*y_min)*DFT[{f(x_p,y_q)}]_rs

where DFT[{f(x_p,y_q)}]_rs is the qth component of the DFT of the sampled points
{f(x_p)} and we have considered the frequencies as kx_r = r/(nxdx),  ky_s = s/(ndy), in the same
convention as numpy does to make easy computation. thus the original frequency
components are kx_r = 2*Pi*r/(nd),ky_s = 2*Pi*s/(nydy), where r,s varies from 0 to (nx-1) and 0 to (ny-1)
'''
import numpy as np
import numpy.fft as ft
import matplotlib as mlab
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def Gaussian(x,y):
	return np.exp(-(x**2.0+y**2.0))

def FT_Gaussian(kx,ky):
	return 0.5*np.exp(-(kx**2.0+ky**2.0)*0.25)
x_min = -10.0
x_max = 10.0
y_min = -10.0
y_max = 10.0
nx = 512
ny = 512

dx = (x_max - x_min)/(nx-1)
dy = (y_max - y_min)/(ny-1)

X,Y = np.meshgrid(np.linspace(x_min,x_max,nx),np.linspace(y_min,y_max,ny))
f = Gaussian(X,Y)
kx = 2.0*np.pi*ft.fftshift(ft.fftfreq(nx,d = dx))
ky = 2.0*np.pi*ft.fftshift(ft.fftfreq(ny,d = dy))
K_x,K_y = np.meshgrid(kx,ky)
dft_f = ft.fftshift(ft.fft2(f,norm = 'ortho'))
ft_f = (np.sqrt(nx*ny)/(2*np.pi))*dx*dy*np.exp(-1.0j*x_min*K_x)*np.exp(-1.0j*y_min*K_y)*dft_f


fig = plt.figure(constrained_layout = True)
spec = fig.add_gridspec(2,2)
plt_3d_f = fig.add_subplot(spec[:,0],projection = '3d')
plt_3d_f.set_title("2D Gaussian")
surf = plt_3d_f.plot_surface(X,Y,f,cmap = cm.coolwarm)
plt.colorbar(surf, shrink=0.6, aspect=30)
plt_3d_f.set_xlabel("x")
plt_3d_f.set_ylabel("y")
plt_3d_f.set_zlabel("exp($x^2+y^2$)")


plt_3d_ft = fig.add_subplot(spec[0,1],projection = '3d')
plt_3d_ft.set_title("Numerical Fourier Transform of 2D Gaussian")
plt_3d_ft.plot_surface(K_x,K_y,ft_f.real,cmap = cm.coolwarm)
plt_3d_ft.set_xlabel("k$_x$",size = 13)
plt_3d_ft.set_ylabel("k$_y$",size = 13)
plt_3d_ft.set_zlabel("Fourier Transform")

plt_3d_ft2 = fig.add_subplot(spec[1,1],projection = '3d')
plt_3d_ft2.set_title("Analytical Fourier Transform of 2D Gaussian")
plt_3d_ft2.plot_surface(K_x,K_y,FT_Gaussian(K_x,K_y),cmap = cm.coolwarm)
plt_3d_ft2.set_xlabel("k$_x$",size = 13)
plt_3d_ft2.set_ylabel("k$_y$",size = 13)
plt_3d_ft2.set_zlabel("Fourier Transform")
plt.show()
                     