/*
FOURIER TRANSFORM OF 1D GAUSSIAN USING FFTW
======================================================================
Author: Rounak Chatterjee
Date : 02/05/2020
=====================================================================

This program uses the same principle as the python version, i.e using the scheme

f_ft(k_q) = sqrt(n/2*pi)*d*exp(-2*pi*i*k_q*x_min)*DFT[{f(x_p)}]_q

where DFT[{f(x_p)}]_q is the qth component of the DFT of the sampled points
{f(x_p)}

to compute the fourier transform of the function f(x)

as we described earlier
{f(x_p)} are the samples of the function at sample points {x_p}
where if x_min and x_max are the two end points, then

d = (x_min-x_max)/(n-1), where n is number of sampled points , hence

x_p = x_min + p.d

*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<fftw3.h>


double f(double x)
{
  return exp(-(x*x));
}


int main()
{
	int n = 512; // Number of sample points
	float x_min = 	-5.0, x_max = 5.0,d = 0.0, *k_arr, *x_arr; // declaring x_min, x_max and delta(d)
	fftw_complex *in, *out, *ft_factors,*prod;
	FILE *ft_data;
	fftw_plan p;
	x_arr =  calloc(n,sizeof(float));
	k_arr =  calloc(n,sizeof(float));
	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n);
	ft_factors = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n);
	prod = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n);
	p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	//Change file path name here
	ft_data = fopen("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fftw_2_data.txt","w");

	d = (x_max - x_min)/(n-1);
	for(int i = 0;i<n;i++)
	{
		x_arr[i] = x_min + i*d;
		 if(i<n/2)
        k_arr[i] = 2*3.142*(i/(n*d));
      else
        k_arr[i] = 2*3.142*((i-n)/(n*d));

		in[i][0] = f(x_arr[i]);
		in[i][1] = 0.0;
		ft_factors[i][0] = cos(k_arr[i]*x_min);
		ft_factors[i][1] = -sin(k_arr[i]*x_min);
	}
	fftw_execute(p);
	printf("DFT Printing\n");
	for(int i = 0;i<n;i++)
	{
		printf("%f 	%f\n",out[i][0],out[i][1]);
	}
	//Normalizing
	for(int i = 0;i<n;i++)
	{
		out[i][0] = (1.0/sqrt(n))*out[i][0];
		out[i][1] = (1.0/sqrt(n))*out[i][1];
	}
	// Complex multiplication by factors
	for(int i =0;i<n;i++)
	{
		prod[i][0] = ft_factors[i][0]*out[i][0] - ft_factors[i][1]*out[i][1];
    	prod[i][1] = ft_factors[i][0]*out[i][1] + ft_factors[i][1]*out[i][0];
	}
	// Constructing FT 
	for(int i = 0;i<n;i++)
	{
		out[i][0] = sqrt(n/(2*3.142))*d*prod[i][0];
		out[i][1] = sqrt(n/(2*3.142))*d*prod[i][1];
	}
	fprintf(ft_data,"#X_val f(x)  k_val   FT(f(x))\n");
for(int i = 0;i<n;i++)
{
  if(i == 0)
    {printf("Creating file\n");}
  fprintf(ft_data,"%f %f %f  %f\n",x_arr[i],in[i][0],k_arr[i],out[i][0]);
} 
fclose(ft_data);

}
