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



def prepare_rf_data(
    df_results: pd.DataFrame,
    use_differences: bool = False  # Если True — используем разницу pat - norm как признак

) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Подготавливает данные для Random Forest из DataFrame с парными метриками.
    
    Параметры:
    ----------
    df_results : pd.DataFrame
        Исходный DataFrame (строка = крыса, колонки = norm_*/pat_*).
    use_differences : bool
        Если True — создаем признаки как разницу (pat - norm) для каждой крысы.
        Если False — преобразуем в длинный формат (каждое измерение — отдельная строка).
    
    Возвращает:
    -----------
    X : pd.DataFrame — матрица признаков
    y : pd.Series — целевая переменная (0=norm, 1=pat)
    feature_names : list — имена признаков
    """
    # Находим парные метрики
    norm_cols = [c for c in df_results.columns if c.startswith('norm_')]
    pat_cols = [c for c in df_results.columns if c.startswith('pat_')]
    
    norm_metrics = [c.replace('norm_', '') for c in norm_cols]
    pat_metrics = [c.replace('pat_', '') for c in pat_cols]
    common_metrics = sorted(list(set(norm_metrics) & set(pat_metrics)))
    
    print(f"Найдено парных метрик: {len(common_metrics)}")
    
    if use_differences:
        # Вариант 1: Разница между pat и norm как признак (одна строка на крысу)
        # Подходит если вы хотите классифицировать крыс по степени изменений
        rows = []
        for _, row in df_results.iterrows():
            rat_features = {}
            rat_features['rat_number'] = row.get('rat_number')
            
            for metric in common_metrics:
                val_norm = row.get(f'norm_{metric}')
                val_pat = row.get(f'pat_{metric}')
                
                if pd.notna(val_norm) and pd.notna(val_pat):
                    # Разница как признак
                    rat_features[f'{metric}_diff'] = val_pat - val_norm
                    # Можно добавить и абсолютные значения
                    rat_features[f'{metric}_norm'] = val_norm
                    rat_features[f'{metric}_pat'] = val_pat
            
            if len([k for k in rat_features.keys() if k not in ['rat_number']]) > 0:
                # Целевая переменная: 1 = есть патология (для этой крысы)
                rows.append(rat_features)
        
        df_features = pd.DataFrame(rows)
        if df_features.empty:
            raise ValueError("Нет данных для обучения после фильтрации")
            
        # В этом режиме мы классифицируем не отдельные измерения, а крыс
        # Поэтому y = 1 для всех (все крысы имеют патологию в эксперименте)
        # Это не подходит для бинарной классификации norm/pat!
        # Поэтому лучше использовать Вариант 2 (ниже)
        print("⚠️ Режим use_differences не подходит для norm/pat классификации!")
        print("   Используйте use_differences=False для long-format данных")
        return None, None, None
        
    else:
        # Вариант 2: Длинный формат (каждое измерение — отдельный пример)
        # Это правильный подход для классификации norm vs pat
        rows = []
        for _, row in df_results.iterrows():
            rat_id = row.get('rat_number')
            
            for metric in common_metrics:
                val_norm = row.get(f'norm_{metric}')
                val_pat = row.get(f'pat_{metric}')
                
                # Норма (класс 0)
                if pd.notna(val_norm):
                    row_data = {'rat_number': rat_id, 'metric_source': 'norm'}
                    row_data[metric] = val_norm
                    rows.append(row_data)
                
                # Патология (класс 1)
                if pd.notna(val_pat):
                    row_data = {'rat_number': rat_id, 'metric_source': 'pat'}
                    row_data[metric] = val_pat
                    rows.append(row_data)
        
        df_long = pd.DataFrame(rows)
        
        # Создаем целевую переменную
        df_long['target'] = (df_long['metric_source'] == 'pat').astype(int)
        
        # Признаки — это сами метрики
        feature_cols = common_metrics
        X = df_long[feature_cols].copy()
        y = df_long['target'].copy()
        
        # Удаляем строки с пропусками в признаках
        mask = X.notna().all(axis=1)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        print(f"Подготовлено примеров: {len(X)} (norm: {(y==0).sum()}, pat: {(y==1).sum()})")
        
        return X, y, feature_cols


def train_rf_and_plot_roc(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    n_splits: int = 5,
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
        min_samples_split=5,
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
        plt.savefig(f"{save_path}_rf_analysis.png", dpi=300, bbox_inches='tight')
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