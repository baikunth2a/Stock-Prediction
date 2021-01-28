import numpy as np
import pandas as pd
import pywt
import copy



def WT(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False, name='close'):

	 # Decomposing 
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levelsï¼ŒcD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function 


    # Denoising
    # Soft Threshold Processing Method
    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficientsï¼Œand no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        va=np.var(cD)
        Tr =va*np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold

    # Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])
    
    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]
            
    if plot:     
        denoised_index = np.sum(coeff, axis=0)   
        d = pd.DataFrame({name: index_list, 'denoised': denoised_index})
        d.plot(figsize=(10,10),subplots=(2,1))
        d.plot(figsize=(10,5))
   
    return coeff

def snr(x,y):
  a=sum(x*x)/sum ((x-y)*(x-y))
  return (10*np.log10(a))



