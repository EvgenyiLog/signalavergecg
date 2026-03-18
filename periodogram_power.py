import numpy as np
from scipy import signal
from typing import Tuple, Optional


import numpy as np
from scipy import signal
from typing import Tuple


def periodogram_power(
    sig: np.ndarray,
    fs: float,
    fmin: float = 300.0,
    fmax: float = 2000.0
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Periodogram power integrated over [fmin, fmax] using trapezoidal rule.
    
    Returns
    -------
    total_power : float
        Integrated power in [fmin, fmax] band [μV² or arb. units²].
        Computed via np.trapz(Pxx, freq) — no interpolation.
    frequencies : np.ndarray
        Frequency bins within [fmin, fmax] (original FFT grid).
    power_density : np.ndarray
        PSD values [units²/Hz] corresponding to `frequencies`.
    """
    print(f"Signal shape: {sig.shape}, fs={fs} Hz")
    
    # Периодограмма: плотность мощности [единица²/Гц]
    freq, Pxx = signal.periodogram(sig, fs=fs, window='hann', scaling='density')
    
    # Фильтрация по диапазону [fmin, fmax]
    mask = (freq >= fmin) & (freq <= fmax)
    freq_band = freq[mask]
    Pxx_band = Pxx[mask]
    
    if len(freq_band) < 2:
        raise ValueError(
            f"Too few frequency bins in [{fmin}, {fmax}] Hz. "
            f"Check fs={fs} and signal length (min N ≈ fs/fmin)."
        )
    
    # Интегральная мощность методом трапеций (без интерполяции!)
    total_power = np.trapezoid(Pxx_band, freq_band)
    
    print(f"Band: [{fmin}, {fmax}] Hz | Bins: {len(freq_band)} | "
          f"Total power: {total_power:.4g}")
    
    return total_power, freq, Pxx

