from typing import List, Union, Dict
import numpy as np
from scipy import stats
from compute_late_potentials_from_avg import bandpass_saecg

def calculate_skew_kurtosis(
    data: Union[List[float], np.ndarray],fs:int=6250
) -> Dict[str, float]:
    """
    Рассчитывает асимметрию (skewness) и эксцесс (kurtosis) для набора данных.

    Параметры
    ----------
    data : Union[List[float], np.ndarray]
        Одномерный массив или список числовых данных (выборка).

    Возвращает
    ----------
    Dict[str, float]
        Словарь, содержащий:
        - 'skewness': Коэффициент асимметрии.
        - 'kurtosis': Коэффициент эксцесса (избыточный, excess kurtosis).
        - 'mean': Среднее значение.
        - 'std': Стандартное отклонение.

    Интерпретация результатов
    -------------------------
    Skewness (Асимметрия):
        - > 0 : Правосторонняя асимметрия (хвост справа длиннее).
        - < 0 : Левосторонняя асимметрия (хвост слева длиннее).
        - = 0 : Симметричное распределение.

    Kurtosis (Эксцесс):
        - > 0 : Островершинное распределение (тяжелые хвосты, больше выбросов, чем у нормального).
        - < 0 : Плосковершинное распределение (легкие хвосты).
        - = 0 : Нормальное распределение (гауссово).
        *Примечание: Здесь используется 'excess' kurtosis (Kurtosis - 3).*
    """
    
    # Конвертируем входные данные в numpy массив для производительности
    arr = np.asarray(data)

    if arr.ndim != 1:
        raise ValueError("Входные данные должны быть одномерным массивом.")
    
    if len(arr) < 3:
        raise ValueError("Для расчета статистик требуется минимум 3 элемента данных.")
    
    arr=bandpass_saecg(arr,fs)
    # Расчет статистик с помощью scipy
    skew_val = stats.skew(arr)
    kurt_val = stats.kurtosis(arr, fisher=True) # fisher=True возвращает избыточный эксцесс
    
    return {
        "skewness": float(skew_val),
        "kurtosis": float(kurt_val),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr))
    }

