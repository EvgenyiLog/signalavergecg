from pyentrp import entropy as ent
import numpy as np
from typing import List, Union, Tuple

from compute_late_potentials_from_avg import bandpass_saecg

def calculate_entropies(
    data: Union[List[float], np.ndarray],
    sample_dim: int = 2,
    sample_r_factor: float = 0.2,
    perm_order: int = 3,
    perm_delay: int = 1,
    fs:int=6250
) -> Tuple[float, float, float]:
    """
    Вычисляет три типа энтропии для временного ряда: Шеннона, семпл-энтропию и permutation-энтропию.
    
    Parameters
    ----------
    data : Union[List[float], np.ndarray]
        Входной временной ряд (список или массив чисел)
    sample_dim : int, optional
        Размерность вложения для семпл-энтропии (параметр m), по умолчанию 2
    sample_r_factor : float, optional
        Коэффициент для вычисления радиуса толерантности (r = factor * std(data)), 
        по умолчанию 0.2
    perm_order : int, optional
        Порядок (размерность вложения) для permutation-энтропии, по умолчанию 3
    perm_delay : int, optional
        Задержка для permutation-энтропии, по умолчанию 1
    
    Returns
    -------
    Tuple[float, float, float]
        Кортеж из трех значений энтропии в порядке:
        (shannon_entropy, sample_entropy, permutation_entropy)
    
    Examples
    --------
    >>> ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
    >>> shannon, sample, perm = calculate_three_entropies(ts)
    >>> print(f"Shannon: {shannon:.4f}, Sample: {sample:.4f}, Perm: {perm:.4f}")
    
    Notes
    -----
    - Энтропия Шеннона измеряет неопределенность/информационное содержание распределения
    - Семпл-энтропия оценивает сложность и предсказуемость временного ряда
    - Permutation-энтропия отражает сложность динамической системы через порядок паттернов
    """
    # Преобразуем входные данные в numpy массив, если необходимо
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    data=bandpass_saecg(data,fs)
    shannon_entropy = ent.shannon_entropy(data)
    
    # 2. Семпл-энтропия
    std_data = np.std(data)
    r_value = sample_r_factor * std_data
    sample_entropy = ent.sample_entropy(data, sample_dim, r_value)
    
    # 3. Permutation-энтропия
    permutation_entropy = ent.permutation_entropy(data, perm_order, perm_delay)
    
    return shannon_entropy, np.mean(sample_entropy), permutation_entropy




