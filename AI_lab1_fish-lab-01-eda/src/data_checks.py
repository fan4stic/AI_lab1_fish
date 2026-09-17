import pandas as pd
import numpy as np

def check_structure_and_duplicates(df: pd.DataFrame) -> dict:
    #Проверяет форму таблицы, типы данных, дубликаты и уникальные поля-утечки
    results = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_leakage_columns": []
    }
    
    #Поиск потенциальных уникальных идентификаторов/утечек (nunique == len(df))
    for col in df.columns:
        if df[col].nunique(dropna=False) == len(df):
            results["unique_leakage_columns"].append(col)
            
    return results

def check_missing_and_invalid(df: pd.DataFrame) -> dict:
    #Проверяет наличие пропусков (NaN), бесконечностей (Inf) и нефизичных значений (<=0)
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    missing_sum = df.isna().sum().to_dict()
    inf_sum = {col: int(np.isinf(df[col]).sum()) for col in num_cols}
    
    #Нереальные значения (масса, длинны, высота и ширина должны быть > 0)
    non_positive = {col: int((df[col] <= 0).sum()) for col in num_cols}
    
    return {
        "missing_values": missing_sum,
        "infinity_values": inf_sum,
        "non_positive_values": non_positive
    }

def check_geometry_consistency(df: pd.DataFrame) -> dict:
    #Проверяет геометрическую согласованность: Length1 <= Length2 <= Length3
    violations = df[
        (df['Length1'] > df['Length2']) | 
        (df['Length2'] > df['Length3'])
    ]
    return {
        "geometry_violations_count": len(violations),
        "violation_indices": violations.index.tolist()
    }

def check_class_imbalance(df: pd.DataFrame, target_col: str = 'Species') -> pd.DataFrame:
    #Анализирует распределение категорий, выявляя редкие классы и дисбаланс
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    imbalance_df = pd.DataFrame({
        'Count': counts,
        'Percentage (%)': percentages.round(2)
    })
    return imbalance_df

def check_near_duplicates(df: pd.DataFrame, threshold: float = 1e-4) -> int:
    #Находит практически идентичные строки по числовым признакам
    num_df = df.select_dtypes(include=[np.number])
    #Нормализуем данные для честного сравнения расстояний
    normalized_df = (num_df - num_df.min()) / (num_df.max() - num_df.min() + 1e-9)
    
    #Вычисляем матрицу расстояний между объектами
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(normalized_df.values, metric='euclidean'))
    
    #Зануляем диагональ
    np.fill_diagonal(distances, np.inf)
    
    # Считаем пары с расстоянием меньше порога
    near_dup_pairs = np.sum(distances < threshold) // 2
    return int(near_dup_pairs)