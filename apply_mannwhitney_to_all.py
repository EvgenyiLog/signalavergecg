import numpy as np
import pandas as pd
from compare_signals_mannwhitney import compare_signals_mannwhitney
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
    
    # 2. Словарь для сбора результатов
    all_results = []
    
    # 3. Итерируемся по каждой крысе
    for idx, row in df_results.iterrows():
        rat_id = row.get('rat_number', idx)
        rat_results = {'rat_number': rat_id}
        
        for metric in common_metrics:
            norm_col = f'norm_{metric}'
            pat_col = f'pat_{metric}'
            
            val_norm = row.get(norm_col)
            val_pat = row.get(pat_col)
            
            # Проверка на наличие данных
            if pd.isna(val_norm) or pd.isna(val_pat):
                rat_results[f'{metric}_pvalue'] = None
                rat_results[f'{metric}_significant'] = None
                continue
            
            # Для Манна-Уитни нужны массивы. 
            # Если у вас в ячейке одно число, оборачиваем в массив.
            # Если там уже массив/список — оставляем как есть.
            arr_norm = np.array([val_norm]) if np.isscalar(val_norm) else np.array(val_norm)
            arr_pat = np.array([val_pat]) if np.isscalar(val_pat) else np.array(val_pat)
            
            # Запуск теста
            test_res = compare_signals_mannwhitney(
                arr_norm, 
                arr_pat, 
                alternative=alternative,
                lpvalue=lpvalue
            )
            
            # Сохраняем ключевые результаты
            rat_results[f'{metric}_pvalue'] = test_res.get('pvalue')
            rat_results[f'{metric}_significant'] = test_res.get('significant')
            rat_results[f'{metric}_statistic'] = test_res.get('statistic')
            
            # Опционально: сохраняем разность средних для понимания направления
            if test_res.get('mean_norm') is not None:
                rat_results[f'{metric}_diff'] = test_res.get('mean_pat') - test_res.get('mean_norm')
        
        all_results.append(rat_results)
    
    # 4. Создаем итоговый DataFrame
    df_stats = pd.DataFrame(all_results)
    
    # Сортируем колонки: сначала rat_number, потом по алфавиту
    cols = ['rat_number'] + sorted([c for c in df_stats.columns if c != 'rat_number'])
    
    return df_stats[cols]