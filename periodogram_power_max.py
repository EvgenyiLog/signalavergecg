import numpy as np
from scipy import signal
from typing import Tuple

def periodogram_power_max(
    sig: np.ndarray,
    fs: float,
    fmin: float = 300.0,
    fmax: float = 2000.0
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Periodogram power integrated over [fmin, fmax] using trapezoidal rule.
    
    Returns
    -------
    
    max_frequency : float
        Frequency of maximum PSD within [fmin, fmax] [Hz].
    max_power_density : float
        Maximum PSD value within [fmin, fmax] [units²/Hz].
    
    """
    print(f"Signal shape: {sig.shape}, fs={fs} Hz")
    
    # Периодограмма: плотность мощности [единица²/Гц]
    freq, Pxx = signal.periodogram(sig, fs=fs, window='hann', scaling='density')
    
    # Фильтрация по диапазону [fmin, fmax]
    mask = (freq >= fmin) & (freq <= fmax)
    freq_band = freq[mask]
    Pxx_band = Pxx[mask]
    
    
    
    # Максимальная частота и максимальная СПМ
    if len(Pxx_band) > 0:
        max_idx = np.argmax(Pxx_band)
        max_frequency = freq_band[max_idx]
        max_power_density = Pxx_band[max_idx]
    else:
        max_frequency = np.nan
        max_power_density = np.nan
    
    return max_frequency, max_power_density