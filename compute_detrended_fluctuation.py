import numpy as np
from typing import Union, Optional
from antropy import detrended_fluctuation

def compute_detrended_fluctuation(
    data: Union[np.ndarray, list]
    ) -> float:
    """
    Вычисляет показатель детрендированного флуктуационного анализа (DFA) для временного ряда.

    DFA используется для обнаружения долговременной корреляции и самоподобия 
    в нестационарных сигналах.

    Parameters
    ----------
    data : Union[np.ndarray, list]
        Входной массив данных (временной ряд). Может быть списком или NumPy массивом.
    

    Returns
    -------
    float
        Показатель Херста (Hurst exponent) или показатель масштабирования DFA.
        Значение интерпретируется следующим образом:
        - H < 0.5: антиперсистентность (отрицательная корреляция)
        - H = 0.5: случайное блуждание (отсутствие корреляции)
        - H > 0.5: персистентность (положительная корреляция)

    Raises
    ------
    ValueError
        Если входной массив пуст или содержит недостаточно данных.
    TypeError
        Если тип входных данных не поддерживается.

    Examples
    --------
    >>> import numpy as np
    >>> # Генерация случайного временного ряда
    >>> x = np.random.randn(1000)
    >>> # Вычисление DFA
    >>> hurst = compute_detrended_fluctuation(x)
    >>> print(f"Показатель Херста: {hurst:.3f}")
    
    >>> # С персистентным рядом
    >>> persistent = np.cumsum(np.random.randn(1000))
    >>> hurst_persistent = compute_detrended_fluctuation(persistent)
    >>> print(f"Показатель Херста (персистентный): {hurst_persistent:.3f}")
    """
    # Проверка входных данных
    if data is None or len(data) == 0:
        raise ValueError("Входной массив не может быть пустым")
    
    # Преобразование в numpy массив, если передан список
    if isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"Входные данные должны быть списком или numpy массивом, получен {type(data)}")
    
    # Проверка на достаточное количество точек
    if len(data) < 10:
        raise ValueError(f"Недостаточно данных для DFA. Минимум 10 точек, получено {len(data)}")
    
    # Проверка на NaN или inf значения
    if not np.isfinite(data).all():
        raise ValueError("Входные данные содержат NaN или бесконечные значения")
    
    # Вычисление DFA с использованием библиотеки antropy
    try:
        dfa_result = detrended_fluctuation(data)
        return float(dfa_result)
    except Exception as e:
        raise RuntimeError(f"Ошибка при вычислении DFA: {str(e)}")





