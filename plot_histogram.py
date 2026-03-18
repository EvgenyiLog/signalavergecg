import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal
import matplotlib.axes


def plot_histogram(
    data: np.ndarray,
    bins: int = 50,
    kde: bool = True,
    stat: Literal['count', 'frequency', 'density', 'probability'] = 'density',
    title: str = 'Histogram',
    xlabel: str = 'Value',
    ylabel: str = 'Density',
    figsize: Tuple[float, float] = (8, 4),
    color: str = '#2E86AB',
    alpha: float = 0.6,
    show_grid: bool = True,
    save_path: Optional[str] = None
) -> Tuple[ np.ndarray, np.ndarray]:
    """
    Seaborn histogram with KDE overlay.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (1D array).
    bins : int
        Number of histogram bins.
    kde : bool
        If True, overlay Kernel Density Estimate curve.
    stat : {'count', 'frequency', 'density', 'probability'}
        What statistic to compute:
        - 'count': number of observations in each bin
        - 'frequency': count / total count
        - 'density': normalize so area under histogram = 1
        - 'probability': proportion of observations in each bin
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    color : str
        Bar color (hex, name, or RGB).
    alpha : float
        Bar transparency (0–1).
    show_grid : bool
        If True, show grid on y-axis.
    save_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
   
    bin_edges : np.ndarray
        Histogram bin edges (length = bins + 1).
    bin_heights : np.ndarray
        Histogram bar heights (length = bins).
    """
    print(f"Data shape: {data.shape}, range: [{np.min(data):.4g}, {np.max(data):.4g}]")
    
    # Настройка стиля seaborn
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Гистограмма через seaborn (возвращает Axes)
    sns.histplot(
        data=data,
        bins=bins,
        kde=kde,
        stat=stat,
        color=color,
        alpha=alpha,
        ax=ax,
        line_kws={'linewidth': 2, 'color': '#E63946'} if kde else None
    )
    
    # Оформление
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    if show_grid:
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Сохранение (если указан путь)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to: {save_path}")
    
    # Извлекаем данные гистограммы для возврата (через matplotlib)
    # Примечание: sns.histplot не возвращает n/bins напрямую, используем np.histogram
    bin_heights, bin_edges = np.histogram(data, bins=bins)
    
    # Нормализуем heights в соответствии со stat
    if stat == 'density':
        bin_widths = np.diff(bin_edges)
        bin_heights = bin_heights / (len(data) * bin_widths)
    elif stat == 'frequency':
        bin_heights = bin_heights / len(data)
    elif stat == 'probability':
        bin_heights = bin_heights / len(data)
    # 'count' — оставляем как есть
    
    print(f"Bins: {bins}, stat='{stat}', KDE: {kde}")
    
    return  bin_edges, bin_heights