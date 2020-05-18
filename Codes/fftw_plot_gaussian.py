'''
This Program plots the data from the file created by the C program
named fftw_gaussian.c and compares with analytical form
'''
import numpy as np
import matplotlib.pyplot as plt

def G(k):
	return np.exp(-0.25*k**2.0)/np.sqrt(2.0)

x_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fftw_data_2.txt",dtype = np.float64,comments = '#',usecols = 0)
f_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fftw_data_2.txt",dtype = np.float64,comments = '#',usecols = 1)
k_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fftw_data_2.txt",dtype = np.float64,comments = '#',usecols = 2)
ft_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fftw_data_2.txt",dtype = np.float64,comments = '#',usecols = 3)

fig = plt.figure(constrained_layout = True)
spec = fig.add_gridspec(1,2)
fig.suptitle("Fourier transform of Gaussian function using GSL",size = 15)
p1 = fig.add_subplot(spec[0,0])
p1.set_title("Configaration space",size = 14)
p1.plot(x_data,f_data)
p1.set_xlabel("x",size = 13)
p1.set_ylabel("exp(-x$^2$)",size = 13)
p1.grid()
p2 = fig.add_subplot(spec[0,1])
p2.set_title("Fourier Space",size = 14)
p2.set_xlabel("frequency modes (k)",size = 13)
p2.set_ylabel("FT(exp(-x$^2$))",size = 13)
p2.set_xlim([-15.0,15.0])
p2.plot(np.fft.fftshift(k_data),np.fft.fftshift(ft_data),'o',color='blue',label = "Numerical")
p2.plot(np.linspace(-15.0,15.0,1000),G(np.linspace(-15.0,15.0,1000)),color = 'green',label = 'Analytical')
p2.legend()
p2.grid()
plt.show()