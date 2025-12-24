import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                   MaxAbsScaler, Normalizer, QuantileTransformer, 
                                   PowerTransformer)

def apply_log_transform(df, columns, scaler=None):
    """
    Applies natural logarithm transformation to specified columns.
    Adds 1 to handle zero values (log1p).
    """
    df_new = df.copy()
    for col in columns:
        df_new[col] = np.log1p(df_new[col])
    return df_new

def apply_standard_scaler(df, columns, scaler=None):
    """
    Standardizes features. 
    If scaler is None, fits and transforms. 
    If scaler is provided, only transforms.
    """
    df_new = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df_new[columns] = scaler.fit_transform(df_new[columns])
        return df_new, scaler  # Hem yeni df'i hem de eğitilmiş nesneyi döndürür
    else:
        df_new[columns] = scaler.transform(df_new[columns])
        return df_new  # Sadece yeni df'i döndürür

def apply_minmax_scaler(df, columns,scaler = None, feature_range=(0, 1)):
    """
    Transforms features by scaling each feature to a given range, typically [0, 1].
    """
    df_new = df.copy()

    if scaler is None:

        scaler = MinMaxScaler(feature_range=feature_range)
        df_new[columns] = scaler.fit_transform(df_new[columns])
        return df_new, scaler  # Hem yeni df'i hem de eğitilmiş nesneyi döndürür
    else:

        df_new[columns] = scaler.transform(df_new[columns])
        return df_new

def apply_robust_scaler(df, columns, scaler=None):
    """
    Scales features using statistics that are robust to outliers (Median and IQR).
    """
    df_new = df.copy()

    if scaler is None:
        scaler = RobustScaler()
        df_new[columns] = scaler.fit_transform(df_new[columns])
        return df_new, scaler  # Hem yeni df'i hem de eğitilmiş nesneyi döndürür
    else:
        df_new[columns] = scaler.transform(df_new[columns])
        return df_new

def apply_maxabs_scaler(df, columns, scaler=None):
    """
    Scales each feature by its maximum absolute value. Ideal for sparse data.
    """
    df_new = df.copy()
    
    if scaler is None:
        scaler = MaxAbsScaler()
        df_new[columns] = scaler.fit_transform(df_new[columns])
        return df_new, scaler
    else:
        df_new[columns] = scaler.transform(df_new[columns])
        return df_new

def apply_l2_normalizer(df, columns, scaler=None):
    """
    Scales individual samples to have unit L2 norm (Euclidean distance).
    """
    df_new = df.copy()

    if scaler is None:
        scaler = Normalizer(norm='l2')
        df_new[columns] = scaler.fit_transform(df_new[columns])
        return df_new, scaler  # Hem yeni df'i hem de eğitilmiş nesneyi döndürür
    else:
        df_new[columns] = scaler.transform(df_new[columns])
        return df_new

def apply_quantile_transform(df, columns, scaler = None ,output_distribution='normal'):
    """
    Transforms features using quantiles information to follow a Normal or Uniform distribution.
    """
    df_new = df.copy()

    if scaler is None:
        scaler = QuantileTransformer(output_distribution = output_distribution)
        df_new[columns] = scaler.fit_transform(df_new[columns])
        return df_new, scaler  # Hem yeni df'i hem de eğitilmiş nesneyi döndürür
    else:
        df_new[columns] = scaler.transform(df_new[columns])
        return df_new

def apply_power_transform(df, columns, scaler = None , method='yeo-johnson'):
    """
    Applies a power transform to make data more Gaussian-like (Yeo-Johnson or Box-Cox).
    """
    df_new = df.copy()

    if scaler is None:
        scaler = PowerTransformer(method=method)
        df_new[columns] = scaler.fit_transform(df_new[columns])
        return df_new, scaler  # Hem yeni df'i hem de eğitilmiş nesneyi döndürür
    else:
        df_new[columns] = scaler.transform(df_new[columns])
        return df_new







if __name__ == "__main__":
    
    data = {'val': [10, 20, 30, 40, 50, 1000]} 
    train_df = pd.DataFrame(data)
    test_df = pd.DataFrame({'val': [15, 25]})
    
    # 1. Train üzerinde eğit ve dönüştür
    train_scaled, my_scaler = apply_standard_scaler(train_df, ['val'])
    print("\nEğitilmiş Train Verisi:\n", train_scaled)

    # 2. Test üzerinde sadece uygula
    test_scaled = apply_standard_scaler(test_df, ['val'], scaler=my_scaler)
    print("\nTrain'e göre dönüştürülmüş Test Verisi:\n", test_scaled)
    
    print("\nİşlem Başarılı! Scaler nesnesi saklandı ve test setine uygulandı.")

