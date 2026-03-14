import pywt
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def calculate_band_power_ratio_2d(power: np.ndarray, frequencies: np.ndarray, 
                                   time_axis: np.ndarray,
                                   band_min: float, band_max: float,
                                   dfreq: float, dt: float) -> float:
    """Интегрирование по обеим осям методом трапеций"""
    
    # Маска по частотам
    mask = (frequencies >= band_min) & (frequencies <= band_max)
    
    # 1. Интеграл по времени для каждой частоты (внутри полосы)
    power_in_band = power[mask, :]  # (n_band_freqs, n_time)
    band_power_freq = np.trapz(power_in_band, dx=dt, axis=1)  # интеграл по времени
    
    # 2. Интеграл по частоте
    band_power = np.trapz(band_power_freq, dx=dfreq)  # интеграл по частоте
    
    # 3. Общая мощность (по всем частотам)
    total_power_freq = np.trapz(power, dx=dt, axis=1)
    total_power = np.trapz(total_power_freq, dx=dfreq)
    
    return band_power / total_power if total_power > 0 else 0.0


def calculate_band_power_ratio_robust(Sp: np.ndarray, frequencies: np.ndarray, 
                                       band_min: float, band_max: float,
                                       dfreq: float = None) -> float:
    """
    Расчет доли мощности с учетом численного интегрирования.
    Для равномерной сетки частот dfreq можно не указывать — результат для ratio будет тем же.
    """
    mask = (frequencies >= band_min) & (frequencies <= band_max)
    
    if dfreq is not None:
        # Метод трапеций (более точно, особенно при неравномерной сетке)
        band_power = np.trapz(Sp[mask], dx=dfreq)
        total_power = np.trapz(Sp, dx=dfreq)
    else:
        # Простая сумма (достаточно для относительных величин)
        band_power = np.sum(Sp[mask])
        total_power = np.sum(Sp)
    
    return band_power / total_power if total_power > 0 else 0.0