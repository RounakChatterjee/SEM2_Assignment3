'''
This function plots the data from the data file created by the c program
named fft_gsl.c and also compares with analytical result
'''
import numpy as np
import matplotlib.pyplot as plt
def box(k):
	return 0.5*np.sqrt(np.pi/2.0)*(np.sign(k+1)-np.sign(k-1))

x_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fft_gsl_data.txt",dtype = np.float64,comments = '#',usecols = 0)
f_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fft_gsl_data.txt",dtype = np.float64,comments = '#',usecols = 1)
k_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fft_gsl_data.txt",dtype = np.float64,comments = '#',usecols = 2)
ft_data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fft_gsl_data.txt",dtype = np.float64,comments = '#',usecols = 3)

fig = plt.figure(constrained_layout = True)
spec = fig.add_gridspec(1,2)
fig.suptitle("Fourier transform of sinc(x) function using GSL",size = 15)
p1 = fig.add_subplot(spec[0,0])
p1.set_title("Configaration space",size = 14)
p1.plot(x_data,f_data)
p1.set_xlabel("x",size = 13)
p1.set_ylabel("sinc(x)",size = 13)
p1.grid()
p2 = fig.add_subplot(spec[0,1])
p2.set_title("Fourier Space",size = 14)
p2.set_xlabel("frequency modes (k)",size = 13)
p2.set_ylabel("FT(sinc(x))",size = 13)
p2.set_xlim([-5.0,5.0])
p2.plot(np.fft.fftshift(k_data),np.fft.fftshift(ft_data),'.-',color='blue',label = "Numerical")
p2.plot(np.linspace(-5.0,5.0,500),box(np.linspace(-5.0,5.0,500)),color = '#00FF00',label = "Analytical",lw = 2)
p2.legend()
p2.grid()

plt.show()