'''
PERIODOGRAM
===============================================================
Author :Rounak Chatterjee
Date : 18/05/2020
==============================================================
This Program computes the Periodogram for the Given Noise Data.
This is done in two methods,the first method is a direct computation 
of the periodogram using discrete fourier transform of the data.
if {x_p} is a data set, then the periodogram value for a frequency point
{k_q} is Pn(k_q) = 1/n |dft({x_p})_q|^2.0

In the next method, we bin the data in a 10 data window and compute the average periodogram
of the data in this window, this is the simpliest bartlett method that we will use 
to compute the periodogram
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
data = np.loadtxt("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/Noise_data.txt",dtype = np.float64)
x = np.linspace(0.0,len(data),len(data),dtype = np.float64)
# Computing Normal Periodogram
def calc_pow_spec(data,binned = False):
	if(binned == False ):
		k = np.fft.fftshift(np.fft.fftfreq(len(data)))
		dft = np.fft.fftshift(np.fft.fft(data,norm = 'ortho'))
		spec = (1.0/len(data))*np.absolute(dft)**2.0
		return [k,spec]
	else:
		dft = np.fft.fftshift(np.fft.fft(data,norm = 'ortho'))
		val = np.mean(np.absolute(dft)**2.0)
		return val
pow_spec = calc_pow_spec(data)
# performing Accuracy test using Scipy
sci_pow_spec = sps.periodogram(data,scaling = 'spectrum',return_onesided = False)

# Binned Power Spectrum

data_bin = np.zeros(shape = (52,10),dtype = np.float64)
binned_pow_spec = np.zeros(52,dtype = np.float64)

# We took the length as 52 since there are 512 data points
k = 0
for i in range(len(data_bin)):
	for j in range(10):
		data_bin[i][j] = data[k]
		k = k+1
		if(k == len(data)):
			break
for i in range(len(data_bin)):
	binned_pow_spec[i] = calc_pow_spec(data_bin[i],binned = True)

fig = plt.figure(constrained_layout  = True)
spec = fig.add_gridspec(2,2)
d_ft = fig.add_subplot(spec[0,1])
d_ft.set_title("DFT of Data",size = 13)
dft = np.fft.fftshift(np.fft.fft(data,norm = 'ortho'))
k = np.fft.fftshift(np.fft.fftfreq(len(data)))
d_ft.plot(k,dft,color = 'green',label = 'DFT')
d_ft.set_xlabel("frequencies")
d_ft.set_ylabel("DFT of Data")
d_ft.legend()
d_ft.grid()

dat = fig.add_subplot(spec[0,0])
dat.set_title("Data",size = 13)
dat.scatter(x,data)
dat.set_xlabel("Data Label")
dat.set_ylabel("Experimental Data")
dat.grid()

ft = fig.add_subplot(spec[1,0])
ft.set_title("Normal periodogram",size = 13)
ft.plot(pow_spec[0],pow_spec[1],color = 'red',label = 'DFT Computed')
ft.scatter(sci_pow_spec[0],sci_pow_spec[1],color = 'green',label = 'Scipy Computed')
ft.set_xlabel("frequencies(Hz)")
ft.set_ylabel("Periodogram")
ft.legend()
ft.grid()

bin_ft = fig.add_subplot(spec[1,1])
bin_ft.set_title("Binned Periodogram(Bartlett)",size = 13)
bin_ft.stem(np.linspace(0.0,len(binned_pow_spec),len(binned_pow_spec)),binned_pow_spec,markerfmt = ('o','black'),linefmt = ('-','red'),label = "10 point bins")
bin_ft.set_xlabel("frequency bins")
bin_ft.set_ylabel("average periodogram")
bin_ft.legend()
bin_ft.grid()

plt.show()

