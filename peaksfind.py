
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def peaksfind(sig:np.ndarray,fs:float):
    "find peaks signal averged ecg"
    
    y=sig.copy()
    y[y<=0]=0
    rpeaks=np.argmax(y)
    y=sig.copy()
    y[y>=0]=0
    speaks=np.argmin(y)
    speaks = np.array([speaks], dtype=int) 
    rpeaks = np.array([rpeaks], dtype=int) 
    widthsR,width_heights,left_ipsR, right_ipsR=signal.peak_widths(sig, rpeaks)
    widthsS,width_heights,left_ipsS, right_ipsS=signal.peak_widths(sig, speaks)
    q_indices = []
   
    
    for i, r_idx in enumerate(rpeaks):
       
        left_bound = int(np.floor(left_ipsR[i]))
        right_bound = int(np.ceil(right_ipsR[i]))
        
        # === Q-волна: минимум СЛЕВА от R ===
        q_start = max(0, left_bound)
        q_end = r_idx
        
        if q_end > q_start + np.median(widthsR/2):  # Мин. окно 5 отсчётов
            q_window = sig[q_start:q_end]
            q_local_idx = np.argmin(q_window)
            q_global_idx = q_start + q_local_idx
            q_indices.append(q_global_idx)

    if len(q_indices) > 0:
        qpeaks = np.array(q_indices, dtype=int) 
        widthsQ, width_heights, left_ipsQ, right_ipsQ = signal.peak_widths(sig, qpeaks)
        qrson = np.ceil(left_ipsQ).astype(int)
    else:
        # Если Q не найден, возвращаем пустые массивы, чтобы код не упал
        qpeaks = np.array([], dtype=int)
        qrson = np.array([], dtype=int)
        
   
    qrsoff=np.ceil(right_ipsS)
    
    qrsoff=np.asarray(qrsoff,dtype=int)
    return rpeaks,qpeaks,speaks,qrson,qrsoff
