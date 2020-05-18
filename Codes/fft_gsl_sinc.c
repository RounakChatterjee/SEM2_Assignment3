/*
FFT USING GSL
===================================================================
Author : Rounak Chatterjee
Date : 03/05/2020
===================================================================
This program finds the fourier transform of sinc(x) function 
using GSL library functions.

We'll be using the same concepts here as the previous two programs,that is

If {f(x_p)} are the samples of the function at sample points {x_p}
where if x_min and x_max are the two end points, then

d = (x_min-x_max)/(n-1), where n is number of sampled points , hence

x_p = x_min + p.d

thus 

f_ft(k_q) = sqrt(n/2*pi)*d*exp(-2*pi*i*k_q*x_min)*DFT[{f(x_p)}]_q

where DFT[{f(x_p)}]_q is the qth component of the DFT of the sampled points
{f(x_p)}, the frequency elements are corrensponding to q/n, which we can convert
to corresponding frequencies k_q = 2*Pi*q/n*d  
*/

#include <stdio.h>
#include<stdlib.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

// These two macro functions help to extract the real and imaginary parts from a 1D array
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

double f(double x)
{
  if(x == 0.0)
    {return 1.0;}
  else
    {return (sin(x)/x);}
}


int main()
{
  int n = 1024; // Sampling points    
  double x_min = -50.0;
  double x_max = 50.0;
  double x_arr[n],k_arr[n],fact_q[2*n],f_data[n],ft_f_data[2*n];
  int i;
  double p[2*n];
  FILE *ft_data;
  double d = (x_max-x_min)/(n-1);
  ft_data = fopen("C:/Users/ROUNAK/Desktop/study/Numerical Assignment Codes/Assignment III/fft_gsl_data.txt","w");

  for ( i = 0; i < n; i++)
    {
      x_arr[i] = x_min + i*d;
      if(i<n/2)
        k_arr[i] = 2*3.142*(i/(n*d));
      else
        k_arr[i] = 2*3.142*((i-n)/(n*d));
      f_data[i] = f(x_arr[i]);
      REAL(ft_f_data,i) = f(x_arr[i]); 
      IMAG(ft_f_data,i) = 0.0;
      REAL(fact_q,i) = cos(k_arr[i]*x_min) ;
      IMAG(fact_q,i) = -sin(k_arr[i]*x_min);
    }
//Finding FFT
  gsl_fft_complex_radix2_transform(ft_f_data, 1, n,+1);

// Normalizing
for(i = 0;i<n;i++)
{
  if(i == 0)
  {printf("I'm Normalizing\n");}
  
  REAL(ft_f_data,i) = (1.0/sqrt(n))*REAL(ft_f_data,i);
  IMAG(ft_f_data,i) = (1.0/sqrt(n))*IMAG(ft_f_data,i);
 
}

  for(int i = 0;i<n;i++)
  {
    REAL(p,i) = REAL(fact_q,i)*REAL(ft_f_data,i) - IMAG(fact_q,i)*IMAG(ft_f_data,i);
    IMAG(p,i) = REAL(fact_q,i)*IMAG(ft_f_data,i) + IMAG(fact_q,i)*REAL(ft_f_data,i);
   
  }
  printf("Product Done\n");
//Since we only need the real part we multiply the factor sqrt(n/2*Pi) with real part only
for(i = 0;i<n;i++)
{
  REAL(p,i) = d*sqrt(n/(2*3.142))*REAL(p,i);
  printf("%d %f %f\n", i,k_arr[i],REAL(p,i));
} 

fprintf(ft_data,"#X_val f(x)  k_val   FT(f(x))\n");
for(i = 0;i<n;i++)
{
  if(i == 0)
    {printf("Creating file\n");}
  fprintf(ft_data,"%f %f %f  %f\n",x_arr[i],f_data[i],k_arr[i],REAL(p,i));
} 
fclose(ft_data);
  return 0;
}