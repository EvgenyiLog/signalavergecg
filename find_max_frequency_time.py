import numpy as np

def find_max_frequency_time(power: np.ndarray, frequencies: np.ndarray, 
                           band_min: float, band_max: float,time_axis: np.ndarray=None,fs:float=6250) -> tuple:
    """Находит частоту и время максимума мощности в заданной полосе частот
    
    Args:
        power: 2D массив мощности [частота, время]
        frequencies: 1D массив частот
        time_axis: 1D массив времени
        band_min: нижняя граница полосы частот
        band_max: верхняя граница полосы частот
    
    Returns:
        tuple: (частота_максимума, время_максимума)
    """
    # Маска по частотам
    mask = (frequencies >= band_min) & (frequencies <= band_max)
    if time_axis is None:
        time_axis=np.linspace(0,power.shape[1]/fs,power.shape[1]) 
    # Проверка, что есть частоты в маске
    if not np.any(mask):
        return np.nan, np.nan
    
    # Берем только данные в полосе частот
    power_masked = power[mask, :]
    
    # Находим индекс глобального максимума
    max_flat_idx = np.argmax(power_masked)
    
    # Преобразуем в индексы частоты (в маске) и времени
    freq_idx_in_mask, time_idx = np.unravel_index(max_flat_idx, power_masked.shape)
    
    # Получаем реальный индекс частоты
    freq_indices = np.where(mask)[0]
    actual_freq_idx = freq_indices[freq_idx_in_mask]
    
    # Получаем значения частоты и времени
    max_frequency = frequencies[actual_freq_idx]
    max_time = time_axis[time_idx]
    
    return max_frequency, max_time


