from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
def findpeaks(sig:np.ndarray,fs:float,low:float=85,high:float=99)->list:
    "find peaks  use hilbert transform"
    fn=0.5*fs
    fcut=45/fn
    b,a=signal.butter(1,fcut)
    y=signal.filtfilt(b,a,sig)
    # y[y<=0]=0
    
    hs=signal.hilbert(y)
    power=np.real(hs*hs.conj())
    power=np.clip(power, a_min=None,a_max=np.percentile(power,high))
    t=np.linspace(0,len(sig)/fs,len(sig))
    rpeaks, _ = signal.find_peaks(power,distance=150,height=(np.percentile(power,low),np.amax(power)))
    print('Средняя длительность,с')
    peaksR=np.asarray(rpeaks/fs,dtype=float)
    print(np.mean(np.diff(peaksR)))
    print()
    print('Медиана длительность,с')
    print(np.median(np.diff(peaksR)))
    print()
    print('Средняя частота,Гц')
    peaksR=np.asarray(rpeaks/fs,dtype=float)
    print(len(peaksR)/np.ptp(t))
    print()
    print('Пульс')
    print(60*len(peaksR)/np.ptp(t))
    print()
  
    return rpeaks