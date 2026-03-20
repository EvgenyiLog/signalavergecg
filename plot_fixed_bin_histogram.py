import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal
import os


def plot_fixed_bin_histogram(
    freq: np.ndarray,
    Pxx: np.ndarray,
    bin_width: float = 50.0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    normalize: Literal['total', 'bin_width', 'both', None] = 'total',
    save_path: Optional[str] = None,
    dpi: int = 300,
    title: str = "Power Spectrum Histogram",
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строит и сохраняет гистограмму мощности с фиксированными бинами.
    Нормировка — обязательная (по умолчанию: по общей мощности).
    
    Parameters
    ----------
    freq : np.ndarray
        Массив частот от periodogram.
    Pxx : np.ndarray
        Массив PSD значений [единица²/Гц].
    bin_width : float
        Ширина бина в Гц (по умолчанию 50).
    fmin, fmax : float, optional
        Диапазон частот для анализа.
    normalize : {'total', 'bin_width', 'both', None}
        Тип нормировки (обязателен, по умолчанию 'total'):
        - 'total' : делит на сумму всех мощностей (получаем доли/проценты)
        - 'bin_width' : делит на ширину бина (получаем плотность мощности)
        - 'both' : делит и на сумму, и на ширину бина
        - None : без нормировки (не рекомендуется)
    save_path : str, optional
        Путь для сохранения изображения (.png, .pdf, .svg).
    dpi : int
        Разрешение сохраняемого изображения.
    title : str
        Заголовок графика.
    ylabel : str, optional
        Подпись оси Y (авто-генерация, если None).
    figsize : tuple
        Размер фигуры в дюймах.
    **kwargs : dict
        Дополнительные аргументы для plt.bar (color, alpha, edgecolor и т.д.)
    
    Returns
    -------
    bin_centers : np.ndarray
        Центры частотных бинов.
    bin_powers : np.ndarray
        Нормированные значения мощности для каждого бина.
    """
    # === Валидация нормировки ===
    if normalize not in ['total', 'bin_width', 'both', None]:
        raise ValueError(f"normalize must be one of ['total', 'bin_width', 'both', None], got '{normalize}'")
    
    # === Диапазон частот ===
    if fmin is None:
        fmin = freq.min()
    if fmax is None:
        fmax = freq.max()
    
    # === Создание бинов ===
    bin_edges = np.arange(fmin, fmax + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_powers = []
    
    for i in range(len(bin_edges) - 1):
        f_start, f_end = bin_edges[i], bin_edges[i + 1]
        mask = (freq >= f_start) & (freq < f_end)
        f_sub, p_sub = freq[mask], Pxx[mask]
        
        if len(f_sub) < 2:
            power = np.mean(p_sub) * bin_width if len(p_sub) > 0 else 0.0
        else:
            power = np.trapezoid(p_sub, f_sub)  # Интегрируем мощность
        bin_powers.append(power)
    
    bin_powers = np.array(bin_powers)
    
    # === НОРМИРОВКА (обязательная) ===
    if normalize == 'total':
        total = np.sum(bin_powers)
        if total > 0:
            bin_powers = bin_powers / total  # Доли от общей мощности (0..1)
        ylabel_auto = "Relative Power (fraction of total)"
        
    elif normalize == 'bin_width':
        bin_powers = bin_powers / bin_width  # Плотность мощности [единица²/Гц]
        ylabel_auto = f"Power Density (units²/Hz) per {bin_width} Hz bin"
        
    elif normalize == 'both':
        total = np.sum(bin_powers)
        if total > 0:
            bin_powers = bin_powers / total / bin_width
        ylabel_auto = f"Normalized Density (1/Hz)"
        
    else:  # normalize is None
        ylabel_auto = f"Absolute Power (units²) per {bin_width} Hz bin"
    
    if ylabel is None:
        ylabel = ylabel_auto
    
    # === Построение графика ===
    fig, ax = plt.subplots(figsize=figsize)
    
    # Параметры столбцов с возможностью переопределения через kwargs
    bar_params = {
        'width': bin_width,
        'edgecolor': 'black',
        'alpha': 0.85,
        'align': 'center',
        'color': '#2E86AB',
        **kwargs
    }
    ax.bar(bin_centers, bin_powers, **bar_params)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Подписи значений (если бинов не слишком много)
    if len(bin_centers) <= 40:
        for x, v in zip(bin_centers, bin_powers):
            if v > np.max(bin_powers) * 0.05:  # подписывать только значимые
                ax.text(x, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    
    # === Сохранение изображения ===
    if save_path:
        # Создаём директорию, если нужно
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ График сохранён: {save_path} (dpi={dpi})")
    
        # Показать график (можно отключить, если работаете в headless-режиме)
    
        plt.close(fig)  # Освобождаем память
    
    return bin_centers, bin_powers