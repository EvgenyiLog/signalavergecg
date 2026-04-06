import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, 
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def prepare_rf_data(df_results: pd.DataFrame, use_differences: bool = False) -> tuple:
    if df_results.empty:
        raise ValueError("df_results is empty!")

    df_results = df_results.copy()
    df_results.columns = df_results.columns.str.strip()
    
    print(f"\n=== prepare_rf_data ===")
    print(f"Входные данные: {df_results.shape}, NaN: {df_results.isna().sum().sum()}")
    
    # Поиск парных метрик
    norm_cols = [c for c in df_results.columns if c.startswith('norm_')]
    pat_cols = [c for c in df_results.columns if c.startswith('pat_')]
    
    norm_metrics = [c.replace('norm_', '') for c in norm_cols]
    pat_metrics = [c.replace('pat_', '') for c in pat_cols]
    common_metrics = sorted(list(set(norm_metrics) & set(pat_metrics)))
    
    print(f"Парных метрик: {len(common_metrics)}")
    
    # === ИСПРАВЛЕНО: Одна строка = одно измерение (все метрики вместе) ===
    rows = []
    for idx, row in df_results.iterrows():
        rat_id = row.get('rat_number')
        
        # Строка для НОРМЫ (все метрики в одной строке)
        row_norm = {'rat_number': rat_id, 'metric_source': 'norm'}
        for metric in common_metrics:
            col_name = f'norm_{metric}'
            row_norm[metric] = row.get(col_name)
        rows.append(row_norm)
        
        # Строка для ПАТОЛОГИИ (все метрики в одной строке)
        row_pat = {'rat_number': rat_id, 'metric_source': 'pat'}
        for metric in common_metrics:
            col_name = f'pat_{metric}'
            row_pat[metric] = row.get(col_name)
        rows.append(row_pat)
    # =======================================================================
    
    df_long = pd.DataFrame(rows)
    print(f"df_long shape: {df_long.shape}")
    print(f"NaN в df_long: {df_long[common_metrics].isna().sum().sum()}")
    
    df_long['target'] = (df_long['metric_source'] == 'pat').astype(int)
    
    feature_cols = common_metrics
    X = df_long[feature_cols].copy()
    y = df_long['target'].copy()
    
    # Фильтр NaN
    mask = X.notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"Подготовлено примеров: {len(X)} (norm: {(y==0).sum()}, pat: {(y==1).sum()})")
    print(f"=========================\n")
    
    if len(X) == 0:
        print("❌ Все данные отфильтрованы! Проверьте NaN в df_long выше.")
    
    return X, y, feature_cols

def train_rf_and_plot_roc(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    n_splits: int = 2,
    n_estimators: int = 100,
    random_state: int = 42,
    plot_feature_importance: bool = True,
    save_path: str = None
) -> dict:
    """
    Обучает Random Forest и строит ROC-кривую с доверительным интервалом.
    
    Возвращает словарь с результатами и метриками.
    """
    
    # Масштабирование признаков (рекомендуется для интерпретации важности)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Инициализация модели
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight='balanced',  # Важно при дисбалансе классов
        random_state=random_state,
        n_jobs=-1
    )
    
    # Кросс-валидация для получения надежных предсказаний
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Получаем предсказанные вероятности для каждого примера (out-of-fold)
    y_prob_cv = cross_val_predict(rf, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    
    # Расчет ROC-кривой по кросс-валидированным предсказаниям
    fpr, tpr, thresholds = roc_curve(y, y_prob_cv)
    roc_auc = auc(fpr, tpr)
    
    # Доверительный интервал для AUC (bootstrap)
    n_bootstraps = 1000
    bootstrapped_scores = []
    rng = np.random.RandomState(random_state)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y[indices], y_prob_cv[indices])
        bootstrapped_scores.append(auc(fpr_b, tpr_b))
    
    sorted_scores = np.sort(bootstrapped_scores)
    auc_ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    auc_ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    # Оптимальный порог по индексу Юдена
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    sensitivity = tpr[youden_idx]
    specificity = 1 - fpr[youden_idx]
    
    # Финальное обучение на всех данных для feature importance
    rf.fit(X_scaled, y)
    
    # Важность признаков
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Матрица ошибок (по предсказаниям с оптимальным порогом)
    y_pred = (y_prob_cv >= optimal_threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    
    # Отчет о классификации
    report = classification_report(y, y_pred, output_dict=True)
    
    # === Построение графиков ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ROC-кривая
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC кривая (AUC = {roc_auc:.3f} [{auc_ci_lower:.3f}–{auc_ci_upper:.3f}])')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Случайный классификатор')
    ax.scatter(fpr[youden_idx], tpr[youden_idx], color='red', s=100, zorder=5,
               label=f'Опт. порог = {optimal_threshold:.3f}')
    ax.fill_between(fpr, 
                    np.maximum(0, tpr - 2*np.std(bootstrapped_scores)), 
                    np.minimum(1, tpr + 2*np.std(bootstrapped_scores)), 
                    color='orange', alpha=0.2, label='95% CI (approx)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    ax.set_title('ROC-кривая: Random Forest', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # 2. Важность признаков
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = ax.barh([feature_names[i] for i in indices[:10]][::-1], 
                   importances[indices][:10][::-1], 
                   color=colors[:10][::-1], edgecolor='black')
    ax.set_xlabel('Важность признака', fontsize=10)
    ax.set_title('Топ-10 наиболее важных признаков', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    # Добавляем значения на столбцы
    for bar, val in zip(bars, importances[indices][:10][::-1]):
        ax.text(val * 1.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=8)
    
    # 3. Матрица ошибок
    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Norm (0)', 'Pat (1)'],
                yticklabels=['Norm (0)', 'Pat (1)'], ax=ax)
    ax.set_xlabel('Предсказано', fontsize=10)
    ax.set_ylabel('Фактически', fontsize=10)
    ax.set_title(f'Матрица ошибок (порог = {optimal_threshold:.3f})', fontsize=11)
    
    # 4. Precision-Recall кривая
    ax = axes[1, 1]
    precision, recall, pr_thresh = precision_recall_curve(y, y_prob_cv)
    avg_precision = average_precision_score(y, y_prob_cv)
    ax.plot(recall, precision, color='blue', lw=2, 
            label=f'PR кривая (AP = {avg_precision:.3f})')
    ax.set_xlabel('Recall (Чувствительность)', fontsize=10)
    ax.set_ylabel('Precision (Точность)', fontsize=10)
    ax.set_title('Precision-Recall кривая', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # === Возврат результатов ===
    results = {
        'model': rf,
        'scaler': scaler,
        'roc_auc': roc_auc,
        'auc_ci_95': (auc_ci_lower, auc_ci_upper),
        'optimal_threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr,
        'feature_importances': dict(zip([feature_names[i] for i in indices], importances[indices])),
        'confusion_matrix': cm,
        'classification_report': report,
        'y_prob_cv': y_prob_cv,
        'y_true': y.values,
        'y_pred': y_pred
    }
    
    return results



def compute_and_plot_individual_roc_auc(X: pd.DataFrame, y: pd.Series, feature_names: list,
                                         save_plots_dir: str = None) -> pd.DataFrame:
    """
    Вычисляет ROC AUC для каждого признака, строит отдельные графики и сохраняет параметры в DataFrame.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Матрица признаков
    y : pd.Series
        Целевая переменная (0 - норма, 1 - патология)
    feature_names : list
        Список названий признаков
    save_plots_dir : str, optional
        Директория для сохранения графиков (если None - графики не сохраняются)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame с параметрами ROC AUC для каждого признака
    """
    results_list = []
    
    for i, feature in enumerate(feature_names):
        # Убираем NaN
        mask = X[feature].notna()
        if mask.sum() == 0:
            continue
            
        y_col = y[mask]
        X_col = X[feature][mask].values
        
        # Проверяем наличие обоих классов
        if len(np.unique(y_col)) < 2:
            print(f"⚠️ Признак '{feature}': только один класс, пропускаем")
            results_list.append({
                'feature': feature,
                'roc_auc': np.nan,
                'optimal_threshold': np.nan,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'youden_index': np.nan
            })
            continue
        
        # Вычисляем ROC кривую
        fpr, tpr, thresholds = roc_curve(y_col, X_col)
        roc_auc = auc(fpr, tpr)
        
        # Индекс Юдена для оптимального порога
        youden_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_idx]
        sensitivity = tpr[youden_idx]
        specificity = 1 - fpr[youden_idx]
        youden_index = sensitivity + specificity - 1
        
        # Сохраняем параметры
        results_list.append({
            'feature': feature,
            'roc_auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'youden_index': youden_index
        })
        
        # Строим график для каждого признака
        plt.figure(figsize=(15, 7))
        
        # ROC кривая
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', 
                label='Random classifier')
        
        # Отмечаем оптимальную точку
        plt.scatter(fpr[youden_idx], tpr[youden_idx], color='red', s=100, zorder=5,
                   label=f'Optimal threshold = {optimal_threshold:.3f}\n'
                         f'Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}')
        
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        plt.title(f'ROC Curve for {feature}', fontsize=13, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем или показываем
        if save_plots_dir:
            import os
            os.makedirs(save_plots_dir, exist_ok=True)
            safe_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
            plt.savefig(f'{save_plots_dir}/roc_{safe_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values('roc_auc', ascending=False).reset_index(drop=True)
    
    # Выводим статистику
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ROC AUC ДЛЯ КАЖДОГО ПРИЗНАКА")
    print("="*60)
    print(df_results.to_string(index=False))
    print("\n" + "="*60)
    print(f"Лучший признак: {df_results.iloc[0]['feature']} (AUC = {df_results.iloc[0]['roc_auc']:.4f})")
    print(f"Худший признак: {df_results.iloc[-1]['feature']} (AUC = {df_results.iloc[-1]['roc_auc']:.4f})")
    print(f"Средний AUC: {df_results['roc_auc'].mean():.4f} ± {df_results['roc_auc'].std():.4f}")
    print("="*60)
    
    return df_results


