from typing import Tuple, Dict, Optional
import numpy as np
from scipy.stats import mannwhitneyu


def compare_signals_mannwhitney(
    saecgnorm: np.ndarray,
    saecgpat: np.ndarray,
    alternative: str = 'two-sided',
    lpvalue:float=0.05
) -> Dict[str, Optional[float]]:
    """
    Выполняет статистическое сравнение двух сигналов (SAECG) с использованием 
    U-критерия Манна-Уитни.

    Критерий Манна-Уитни — это непараметрический тест, который проверяет, 
    принадлежат ли две независимые выборки к одной генеральной совокупности.
    Он не требует нормального распределения данных.

    Параметры
    ---------
    saecgnorm : np.ndarray
        Первый сигнал (например, нормализованный усредненный ЭКГ).
        Одномерный массив числовых данных.
    saecgpat : np.ndarray
        Второй сигнал (например, патологический усредненный ЭКГ).
        Одномерный массив числовых данных.
    alternative : str, optional
        Тип альтернативной гипотезы (по умолчанию 'two-sided'):
        - 'two-sided' : распределения различаются (двусторонний тест)
        - 'less'      : saecgnorm статистически меньше saecgpat
        - 'greater'   : saecgnorm статистически больше saecgpat

    Возвращаемые значения
    ---------------------
    Dict[str, Optional[float]]
        Словарь с результатами теста:
        - 'statistic' : float, значение U-статистики
        - 'pvalue'    : float, p-значение (вероятность ошибки)
        - 'significant' : bool, True если p < 0.05 (статистически значимо)
        - 'mean_norm' : float, среднее значение saecgnorm
        - 'mean_pat'  : float, среднее значение saecgpat
        - 'median_norm' : float, медиана saecgnorm
        - 'median_pat'  : float, медиана saecgpat
        - 'n_norm'    : int, количество точек в saecgnorm
        - 'n_pat'     : int, количество точек в saecgpat
        В случае ошибки все значения будут None.

    Raises
    ------
    ValueError
        Если массивы пустые или имеют недопустимую размерность.
    TypeError
        Если входные данные не являются массивами numpy.

    Пример
    ------
    >>> import numpy as np
    >>> norm = np.array([1.2, 1.5, 1.3, 1.4])
    >>> pat = np.array([2.1, 2.3, 2.0, 2.2])
    >>> result = compare_signals_mannwhitney(norm, pat)
    >>> print(f"p-value: {result['pvalue']:.4f}")
    >>> print(f"Значимо: {result['significant']}")
    """
    try:
        # Валидация входных данных
        if not isinstance(saecgnorm, np.ndarray) or not isinstance(saecgpat, np.ndarray):
            raise TypeError("Оба аргумента должны быть массивами numpy.ndarray")
        
        if saecgnorm.ndim != 1 or saecgpat.ndim != 1:
            raise ValueError("Массивы должны быть одномерными")
        
        if len(saecgnorm) == 0 or len(saecgpat) == 0:
            raise ValueError("Массивы не должны быть пустыми")
        
        # Проверка допустимых значений alternative
        valid_alternatives = ['two-sided', 'less', 'greater']
        if alternative not in valid_alternatives:
            raise ValueError(f"alternative должен быть одним из: {valid_alternatives}")
        
        # Выполнение теста Манна-Уитни
        statistic, pvalue = mannwhitneyu(
            saecgnorm, 
            saecgpat, 
            alternative=alternative,
            nan_policy='omit'  # Игнорировать NaN значения
        )
        if pvalue<lpvalue:
            print('Сильные различия')
        # Формирование результата
        result = {
            'statistic': float(statistic),
            'pvalue': float(pvalue),
            'significant': bool(pvalue < 0.05),
            'mean_norm': float(np.mean(saecgnorm)),
            'mean_pat': float(np.mean(saecgpat)),
            'median_norm': float(np.median(saecgnorm)),
            'median_pat': float(np.median(saecgpat)),
            'n_norm': int(len(saecgnorm)),
            'n_pat': int(len(saecgpat))
        }
        
        return result
    
    except (TypeError, ValueError) as e:
        # Возврат словаря с None в случае ошибки
        return {
            'statistic': None,
            'pvalue': None,
            'significant': None,
            'mean_norm': None,
            'mean_pat': None,
            'median_norm': None,
            'median_pat': None,
            'n_norm': None,
            'n_pat': None,
            'error': str(e)
        }

