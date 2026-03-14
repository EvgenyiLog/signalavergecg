import pywt
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional



def calculate_band_power_ratio(Sp: np.ndarray, frequencies: np.ndarray, 
                               band_min: float, band_max: float) -> float:
    """
    Вычисляет долю мощности в заданном частотном диапазоне.
    
    Args:
        Sp: Глобальный спектр (мощность на каждой частоте)
        frequencies: Соответствующие частоты (в Гц)
        band_min: Нижняя граница диапазона (Гц)
        band_max: Верхняя граница диапазона (Гц)
        
    Returns:
        ratio: Доля мощности (от 0 до 1)
    """
    # Находим индексы частот, попадающих в диапазон
    mask = (frequencies >= band_min) & (frequencies <= band_max)
    
    # Интеграл (сумма) мощности в диапазоне
    band_power = np.sum(Sp[mask])
    
    # Общий интеграл (сумма) мощности
    total_power = np.sum(Sp)
    
    if total_power == 0:
        return 0.0
        
    ratio = band_power / total_power
    return ratio
