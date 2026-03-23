import pywt
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def waveletscaleaogram(sig:np.ndarray,fs:float,waveletname:str='morl',fmin:float=0.1,fmax:float=2000,dfreq:float=100)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    "cwt wavelet power"
    print(sig.shape)
    scales=np.arange(1,1024)
    dt=1/fs
    freq=pywt.scale2frequency(waveletname,scales)/dt
    print(f"Min freq: {np.min(freq)}, Max freq: {np.max(freq)}") 
    frequencies = np.arange(fmin,fmax+1,dfreq)/fs
    scales = pywt.frequency2scale(waveletname, frequencies)
    scales=np.ceil(scales)
    print(f"Min scale: {np.min(scales)}, Max scale: {np.max(scales)}") 
    
    [coefficients, frequencies] = pywt.cwt(sig, scales, waveletname, sampling_period=dt)
    power=np.square(np.abs(coefficients))
    # print(power.shape)
    Sp=np.sum(power,axis=1)
    # print(Sp.shape)
    return Sp,frequencies,power
