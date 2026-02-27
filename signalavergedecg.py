import biosppy
import numpy as np
def signalavergedecg(sig:np.ndarray,fs:float,rpeaks:[np.ndarray,list],before:float=0.06,after:float=0.10,k:int=100)->np.ndarray:
    "signal averged ecg"

    templates,rpeaks=biosppy.signals.ecg.extract_heartbeats(signal=sig, rpeaks=rpeaks, sampling_rate=fs,before=before, after=after)
    # print(templates.shape)
    print(f'Количество кардиоциклов ={templates.shape[0]}')
    if templates.shape[1]>=k:
        saecg=np.mean(templates,axis=0)
        print(saecg.shape)
        print(f'Длительность={len(saecg)/fs}')
        print(f'Отношение {(np.std(sig)/np.std(saecg))/np.sqrt(templates.shape[1])}')

    return saecg