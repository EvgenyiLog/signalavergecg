import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
def apply_mannwhitney_to_all(
    df_results: pd.DataFrame,
    alternative: str = 'two-sided',
    lpvalue: float = 0.05
) -> pd.DataFrame:
    """
    Применяет критерий Манна-Уитни к каждой крысе (rat_number) в DataFrame.
    Сравнивает парные колонки norm_* и pat_*.

    Параметры
    ----------
    df_results : pd.DataFrame
        DataFrame с данными, где строки — это крысы, а колонки — метрики 
        с префиксами 'norm_' и 'pat_'.
    alternative : str
        Тип альтернативной гипотезы для mannwhitneyu.
    lpvalue : float
        Порог значимости.

    Возвращает
    ----------
    pd.DataFrame
        Таблица результатов, где индекс — rat_number, а колонки содержат 
        статистику теста для каждой метрики.
    """
    # 1. Находим парные метрики
    norm_cols = [c for c in df_results.columns if c.startswith('norm_')]
    pat_cols = [c for c in df_results.columns if c.startswith('pat_')]
    
    norm_metrics = [c.replace('norm_', '') for c in norm_cols]
    pat_metrics = [c.replace('pat_', '') for c in pat_cols]
    
    # Оставляем только общие метрики
    common_metrics = list(set(norm_metrics) & set(pat_metrics))
    common_metrics.sort()
    
    print(f"Найдено метрик для сравнения: {len(common_metrics)}")
    
    results = []
    
    for metric in common_metrics:
        norm_col = f'norm_{metric}'
        pat_col = f'pat_{metric}'
        
        # Собираем все значения по всем крысам, отбрасывая NA
        norm_vals = df_results[norm_col].dropna().values
        pat_vals = df_results[pat_col].dropna().values
        
        # Проверяем, достаточно ли данных (хотя бы 2 в каждой группе)
        if len(norm_vals) < 2 or len(pat_vals) < 2:
            print(f"Предупреждение: для метрики {metric} недостаточно данных "
                  f"(norm={len(norm_vals)}, pat={len(pat_vals)}). Пропускаем.")
            continue

    # Выполняем тест Манна-Уитни
        try:
            u_stat, p_value = mannwhitneyu(norm_vals, pat_vals, alternative=alternative)
        except Exception as e:
            print(f"Ошибка для метрики {metric}: {e}")
            continue
        
        # Вычисляем разность (патология - норма) для интерпретации
        median_diff = np.median(pat_vals) - np.median(norm_vals)
        mean_diff = np.mean(pat_vals) - np.mean(norm_vals)
        
        results.append({
            'metric': metric,
            'n_norm': len(norm_vals),
            'n_pat': len(pat_vals),
            'median_norm': np.median(norm_vals),
            'median_pat': np.median(pat_vals),
            'median_diff': median_diff,
            'mean_norm': np.mean(norm_vals),
            'mean_pat': np.mean(pat_vals),
            'mean_diff': mean_diff,
            'statistic': u_stat,
            'pvalue': p_value,
            'significant': p_value < lpvalue
        })
    
    # Создаём DataFrame с результатами
    df_stats = pd.DataFrame(results)
    
    return df_stats