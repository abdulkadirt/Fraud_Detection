import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler , MaxAbsScaler, QuantileTransformer , Normalizer , PowerTransformer 

def apply_log_transform(df, columns):
    """
    Applies natural logarithm transformation to specified columns.
    Adds 1 to handle zero values (log1p).
    """
    df_new = df.copy()
    for col in columns:
        df_new[col] = np.log1p(df_new[col])
    return df_new

def apply_standard_scaler(df, columns):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """
    df_new = df.copy()
    scaler = StandardScaler()
    df_new[columns] = scaler.fit_transform(df_new[columns])
    return df_new

def apply_minmax_scaler(df, columns, feature_range=(0, 1)):
    """
    Transforms features by scaling each feature to a given range, typically [0, 1].
    """
    df_new = df.copy()
    scaler = MinMaxScaler(feature_range=feature_range)
    df_new[columns] = scaler.fit_transform(df_new[columns])
    return df_new

def apply_robust_scaler(df, columns):
    """
    Scales features using statistics that are robust to outliers (Median and IQR).
    """
    df_new = df.copy()
    scaler = RobustScaler()
    df_new[columns] = scaler.fit_transform(df_new[columns])
    return df_new

def apply_maxabs_scaler(df, columns):
    """
    Scales each feature by its maximum absolute value. Ideal for sparse data.
    """
    df_new = df.copy()
    scaler = MaxAbsScaler()
    df_new[columns] = scaler.fit_transform(df_new[columns])
    return df_new

def apply_l2_normalizer(df, columns):
    """
    Scales individual samples to have unit L2 norm (Euclidean distance).
    """
    df_new = df.copy()
    scaler = Normalizer(norm='l2')
    df_new[columns] = scaler.fit_transform(df_new[columns])
    return df_new

def apply_quantile_transform(df, columns, output_distribution='normal'):
    """
    Transforms features using quantiles information to follow a Normal or Uniform distribution.
    """
    df_new = df.copy()
    scaler = QuantileTransformer(output_distribution=output_distribution)
    df_new[columns] = scaler.fit_transform(df_new[columns])
    return df_new

def apply_power_transform(df, columns, method='yeo-johnson'):
    """
    Applies a power transform to make data more Gaussian-like (Yeo-Johnson or Box-Cox).
    """
    df_new = df.copy()
    scaler = PowerTransformer(method=method)
    df_new[columns] = scaler.fit_transform(df_new[columns])
    return df_new


