import numpy as np
from scipy.signal import hilbert
from typing import Union, List

def phase_instability(signal: Union[List[float], np.ndarray]) -> float:
    """
    Вычисляет 50-й перцентиль аналитической фазы сигнала.
    
    Parameters
    ----------
    signal : Union[List[float], np.ndarray]
        Входной сигнал (временной ряд)
        
    Returns
    -------
    float
        50-й перцентиль фазы (медиана)
        
    Example
    -------
    >>> signal = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> phase_median = phase_percentile_50(signal)
    >>> print(phase_median)
    """
    # Преобразуем в numpy массив если нужно
    signal = np.asarray(signal)
    
    # Вычисляем аналитический сигнал через преобразование Гильберта
    analytic_signal = hilbert(signal)
    
    # Получаем фазу (угол) аналитического сигнала
    phase = np.angle(analytic_signal)
    
    # Вычисляем 50-й перцентиль фазы
    percentile_50 = np.percentile(phase, 50)
    
    return percentile_50


