import numpy as np

def time_slice(sig:np.ndarray, start_sec:float, end_sec:float, fs:float)->np.ndarray:
    "time slise"
    start = np.round(fs * start_sec).astype(int)

    if start >= len(sig):
        print("Начальное смещение превышает длительность сигнала")
        return []

    end = np.round(fs * end_sec).astype(int)

    if end < len(sig):
        return sig[start:end]
    else:
        print("Временное окно выходит за пределы длительности сигнала")
        return sig[start:]