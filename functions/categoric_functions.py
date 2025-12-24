import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import chi2_contingency
from itertools import combinations
import warnings
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


def top_missing_cols(df, n=10, thresh=80):
    """
    Return columns with missing values percentage greater than threshold.

    Args:
        df (DataFrame): Input dataframe
        n (int): Number of top columns to return
        thresh (int): Missing percentage threshold

    Returns:
        DataFrame: Columns with missing percentages
    """
    missing_pct = (df.isnull().sum() / df.shape[0]) * 100
    missing_df = missing_pct.reset_index()
    missing_df.columns = ['col', 'missing_percent']
    missing_df = missing_df.sort_values(by=['missing_percent'], ascending=False).reset_index(drop=True)

    print(f'There are {df.isnull().any().sum()} columns with missing values.')
    print(
        f'There are {missing_df[missing_df["missing_percent"] > thresh].shape[0]} columns with missing percent > {thresh}%')

    if n:
        return missing_df.head(n)
    else:
        return missing_df


def plot_categorical_analysis(df, column, target='isFraud'):
    """
    Plot category distribution (pie chart) and fraud rates (stacked bar chart).

    Args:
        df (DataFrame): Input dataframe
        column (str): Column name to analyze
        target (str): Target variable name
    """
    sns.set_style('whitegrid')
    pastel_colors = sns.color_palette('pastel')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=120)

    counts = df[column].value_counts()

    axes[0].pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=pastel_colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 11}
    )
    axes[0].set_title(f'{column} Overall Distribution', fontsize=15, fontweight='bold')

    crosstab = pd.crosstab(df[column], df[target], normalize='index') * 100
    bar_colors = [pastel_colors[0], pastel_colors[3]]

    crosstab.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        color=bar_colors,
        width=0.7,
        edgecolor='white'
    )

    axes[1].set_title(f'{column} Fraud vs Normal Distribution', fontsize=15, fontweight='bold')
    axes[1].set_ylabel('Rate (%)', fontsize=12)
    axes[1].set_xlabel(column, fontsize=12)
    axes[1].legend(title='Status', labels=['Normal', 'Fraud'], loc='upper right', frameon=True)
    axes[1].tick_params(axis='x', rotation=0)

    for c in axes[1].containers:
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 2 else '' for v in c]
        axes[1].bar_label(c, labels=labels, label_type='center', fontsize=10, color='black', weight='bold')

    plt.tight_layout()
    plt.show()

def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by downcasting numeric columns to appropriate data types.

    Args:
        df (DataFrame): Input dataframe
        verbose (bool): Print memory reduction information

    Returns:
        DataFrame: Memory-optimized dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def cat_binary_test(df, feature_list, target='isFraud', alpha=0.05, min_category_size=30):
    """
    Perform statistical hypothesis testing for categorical features with binary target.

    Args:
        df (DataFrame): Input dataframe
        feature_list (list): List of features to test
        target (str): Binary target variable name
        alpha (float): Significance level
        min_category_size (int): Minimum sample size per category

    Returns:
        DataFrame: Test results with statistical significance and practical importance
    """
    results = []

    for feature in feature_list:
        if feature not in df.columns:
            continue

        rates = df.groupby(feature)[target].mean()
        counts = df.groupby(feature)[target].count()

        if len(rates) < 2:
            continue

        rate_diff = rates.max() - rates.min()
        overall_rate = df[target].mean()

        contingency = pd.crosstab(df[feature], df[target])
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        warning = f"Low sample size ({counts.min()})" if counts.min() < min_category_size else ""

        stat_sig = p_value < alpha
        strong_practical = rate_diff >= 0.02
        medium_practical = rate_diff >= 0.01

        if stat_sig and strong_practical:
            decision = "Strong Relation"
            keep = True
        elif stat_sig and medium_practical:
            decision = "Enough Relation"
            keep = True
        elif stat_sig:
            decision = "Poor but Meaningful"
            keep = False
        else:
            decision = "Not Related"
            keep = False

        results.append({
            'Feature': feature,
            'P_Value': round(p_value, 6),
            'Fraud_Rate_Min': round(rates.min(), 4),
            'Fraud_Rate_Max': round(rates.max(), 4),
            'Rate_Diff': round(rate_diff, 4),
            'Overall_Fraud_Rate': round(overall_rate, 4),
            'N_Categories': len(rates),
            'Min_Category_Size': int(counts.min()),
            'Decision': decision,
            'Keep': keep,
            'Warning': warning
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Rate_Diff', ascending=False).reset_index(drop=True)

    return results_df


# Encoding 
def apply_label_encoding(df, columns, encoder_dict=None):
    """Apply Label Encoding to specified columns."""
    df_encoded = df.copy()
    is_training = encoder_dict is None
    if is_training:
        encoder_dict = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if is_training:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoder_dict[col] = le
        else:
            le = encoder_dict[col]
            known = df[col].astype(str).isin(le.classes_)
            df_encoded.loc[known, col] = le.transform(df.loc[known, col].astype(str))
            df_encoded.loc[~known, col] = -1
    
    return df_encoded, encoder_dict

def apply_frequency_encoding(df, columns, freq_dict=None, normalize=False):
    """Replace categories with their frequency counts."""
    df_encoded = df.copy()
    is_training = freq_dict is None
    if is_training:
        freq_dict = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if is_training:
            counts = df[col].value_counts(normalize=normalize)
            freq_dict[col] = {
                'map': counts.to_dict(),
                'default': counts.min() if len(counts) > 0 else 0
            }
        
        df_encoded[col] = df[col].map(freq_dict[col]['map']).fillna(freq_dict[col]['default'])
    
    return df_encoded, freq_dict

# feature engineering functions

# 1. Rare category encoding (reusable) --> car3 ve card5 te kullanılacak.
def encode_rare_categories(df, columns, thresh = 200, rare_maps=None):
    """
    Replace rare categories with 'Others'.
    
    Args:
        columns: dict {col: threshold} or list
        rare_maps: dict {col: [rare_values]} for test set
    """
    is_train = rare_maps is None
    if is_train:
        rare_maps = {}
    
    if isinstance(columns, list):
        columns = {col: 200 for col in columns}
    
    for col, thresh in columns.items():
        if col not in df.columns:
            continue
            
        if is_train:
            counts = df[col].value_counts()
            rare_maps[col] = counts[counts < thresh].index.tolist()
        
        df.loc[df[col].isin(rare_maps[col]), col] = 'Others'
    
    return df, rare_maps

def clean_email_domains(df):
    """
    Group and clean P_emaildomain and R_emaildomain columns into major providers.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Dataframe with cleaned email domain columns
    """
    emails = {
        'gmail': 'google', 'gmail.com': 'google', 'googlemail.com': 'google',
        'hotmail.com': 'microsoft', 'outlook.com': 'microsoft', 'msn.com': 'microsoft',
        'live.com': 'microsoft', 'hotmail.co.uk': 'microsoft', 'hotmail.de': 'microsoft',
        'hotmail.es': 'microsoft', 'live.com.mx': 'microsoft',
        'yahoo.com': 'yahoo', 'ymail.com': 'yahoo', 'rocketmail.com': 'yahoo',
        'yahoo.com.mx': 'yahoo', 'yahoo.co.uk': 'yahoo', 'yahoo.co.jp': 'yahoo',
        'yahoo.de': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo',
        'icloud.com': 'apple', 'me.com': 'apple', 'mac.com': 'apple',
        'aol.com': 'aol', 'aim.com': 'aol',
        'anonymous.com': 'anonymous',
        'protonmail.com': 'protonmail'
    }

    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        df[c + '_bin'] = df[c + '_bin'].fillna('Others')
        df.loc[df[c].isnull(), c + '_bin'] = 'Missing'

    return df


def create_email_match(df):
    """
    Analyze match status between purchaser and recipient email domains.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Dataframe with email match analysis
    """
    mask_both_exist = (df['P_emaildomain'].notnull()) & (df['R_emaildomain'].notnull())
    df['email_match'] = 'Unknown/Missing'

    df.loc[mask_both_exist, 'email_match'] = np.where(
        df.loc[mask_both_exist, 'P_emaildomain'] == df.loc[mask_both_exist, 'R_emaildomain'],
        'Match',
        'Different'
    )

    return df


def consolidate_device_info(df):
    """
    Consolidate id_30 (OS) and DeviceInfo columns into main device categories.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Dataframe with consolidated device information
    """
    if 'id_30' in df.columns:
        df['id_30'] = df['id_30'].astype(str).str.lower()
        df['OS_type'] = 'Others'

        df.loc[df['id_30'].str.contains('windows', na=False), 'OS_type'] = 'Windows'
        df.loc[df['id_30'].str.contains('ios', na=False), 'OS_type'] = 'iOS'
        df.loc[df['id_30'].str.contains('mac', na=False), 'OS_type'] = 'Mac'
        df.loc[df['id_30'].str.contains('android', na=False), 'OS_type'] = 'Android'
        df.loc[df['id_30'].str.contains('linux', na=False), 'OS_type'] = 'Linux'
        df.loc[df['id_30'] == 'nan', 'OS_type'] = 'Missing'

    if 'DeviceInfo' in df.columns:
        df['DeviceInfo'] = df['DeviceInfo'].astype(str).str.lower()
        df['Device_name'] = 'Others'

        df.loc[df['DeviceInfo'].str.contains('windows|trident|rv:', na=False), 'Device_name'] = 'Windows PC'
        df.loc[df['DeviceInfo'].str.contains('ios', na=False), 'Device_name'] = 'Apple Device'
        df.loc[df['DeviceInfo'].str.contains('macos|mac', na=False), 'Device_name'] = 'Mac'
        df.loc[df['DeviceInfo'].str.contains('samsung|sm-|gt-', na=False), 'Device_name'] = 'Samsung'
        df.loc[df['DeviceInfo'].str.contains('huawei|ale-|hi6210', na=False), 'Device_name'] = 'Huawei'
        df.loc[df['DeviceInfo'].str.contains('lg|lg-', na=False), 'Device_name'] = 'LG'
        df.loc[df['DeviceInfo'].str.contains('moto', na=False), 'Device_name'] = 'Motorola'
        df.loc[df['DeviceInfo'].str.contains('redmi|mi ', na=False), 'Device_name'] = 'Xiaomi'
        df.loc[df['DeviceInfo'] == 'nan', 'Device_name'] = 'Missing'

    return df


#  Screen resolution features
def extract_screen_features(df):
    """Parse id_33 into width, height, pixels, aspect ratio."""
    if 'id_33' not in df.columns:
        return df
    
    split = df['id_33'].astype(str).str.split('x', expand=True)
    if split.shape[1] != 2:
        return df
    
    df['screen_width'] = pd.to_numeric(split[0], errors='coerce')
    df['screen_height'] = pd.to_numeric(split[1], errors='coerce')
    df['total_pixels'] = df['screen_width'] * df['screen_height']
    df['aspect_ratio'] = (df['screen_width'] / df['screen_height']).round(2)
    
    return df


def bivariate_comb_risk(df, feature1, feature2, target='isFraud', min_samples=30, show_bar_chart=False):
    """
    Performs a detailed analysis of two specified features and draws a heatmap.

    Parameters:
    -----------
    df : DataFrame
        Dataset to analyze
    feature1, feature2 : str
        Features to analyze
    target : str, default='isFraud'
        Target variable
    min_samples : int, default=30
        Minimum sample size filter
    show_bar_chart : bool, default=False
        Show/hide bar chart

    Returns:
    --------
    DataFrame
        Top 10 risky combinations with statistics
    """
   
    df_viz = df.copy()
    df_viz[feature1] = df_viz[feature1].astype(str).fillna('Missing')
    df_viz[feature2] = df_viz[feature2].astype(str).fillna('Missing')

    
    group = df_viz.groupby([feature1, feature2])[target].agg(['sum', 'count', 'mean']).reset_index()
    group.columns = [feature1, feature2, 'fraud_count', 'total_count', 'fraud_rate']
    group['fraud_rate'] = group['fraud_rate'] * 100

    # Minimum sample filter
    group = group[group['total_count'] >= min_samples].sort_values(by='fraud_rate', ascending=False)

    # Use ALL valid categories for heatmap
    pivot_table = group.pivot(index=feature1, columns=feature2, values='fraud_rate')

    # --- Dynamic Sizing ---
    n_rows = len(pivot_table.index)
    n_cols = len(pivot_table.columns)

    if show_bar_chart:
        # Two plots side by side
        fig_width = max(12, n_cols * 0.8) + 8
        fig_height = max(8, n_rows * 0.5)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        heatmap_ax = axes[0]
        bar_ax = axes[1]
    else:
        # Only heatmap
        fig_width = max(12, n_cols * 0.8)
        fig_height = max(8, n_rows * 0.5)
        fig, heatmap_ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # --- HEATMAP ---
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        center=5,
        ax=heatmap_ax,
        linewidths=0.5,
        cbar_kws={'label': 'Fraud Rate (%)'}
    )
    heatmap_ax.set_title(f'Fraud Heatmap: {feature1} vs {feature2} (All Categories)',
                         fontsize=14, fontweight='bold')
    heatmap_ax.tick_params(axis='x', rotation=45, labelsize=9)
    heatmap_ax.tick_params(axis='y', labelsize=9)

    # --- BAR CHART (Optional) ---
    if show_bar_chart:
        top_risky = group.head(15).sort_values(by='fraud_rate', ascending=True)
        labels = top_risky[feature1].astype(str) + " + " + top_risky[feature2].astype(str)

        bars = bar_ax.barh(
            range(len(top_risky)),
            top_risky['fraud_rate'],
            color=sns.color_palette("Reds", len(top_risky))
        )
        bar_ax.set_yticks(range(len(top_risky)))
        bar_ax.set_yticklabels(labels, fontsize=9)
        bar_ax.set_xlabel('Fraud Rate (%)', fontsize=11)
        bar_ax.set_title(f'Top 15 Highest Risk Combinations\n({feature1} & {feature2})',
                         fontsize=14, fontweight='bold')

        # Annotate bar ends with counts and fraud rates
        for bar, count, fraud_rate in zip(bars, top_risky['total_count'], top_risky['fraud_rate']):
            bar_ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height()/2,
                f'N={count}\n{fraud_rate:.1f}%',
                va='center',
                fontsize=8,
                color='black'
            )

    plt.tight_layout()
    plt.show()

    return group.head(10)


def scan_all_bivariate_combinations(df, feature_list, target='isFraud',
                                    min_samples=50, top_n=50):
    """
    Scan all pairwise combinations of features and return top riskiest subcategories.

    Args:
        df (DataFrame): Input dataframe
        feature_list (list): List of categorical features to test
        target (str): Target variable name
        min_samples (int): Minimum sample size for combination
        top_n (int): Return top N riskiest combinations

    Returns:
        DataFrame: Top riskiest combinations with fraud rates and sample counts
    """
    df_work = df.copy()
    df_work[target] = df_work[target].astype('int32')

    for feat in feature_list:
        if feat in df_work.columns:
            df_work[feat] = df_work[feat].astype(str)

    results = []
    total_pairs = len(list(combinations(feature_list, 2)))

    print(f"Scanning {total_pairs} feature pairs...")

    for idx, (f1, f2) in enumerate(combinations(feature_list, 2), 1):
        if f1 not in df_work.columns or f2 not in df_work.columns:
            continue

        try:
            combo_stats = df_work.groupby([f1, f2], as_index=False).agg({
                target: ['sum', 'count', 'mean']
            })

            combo_stats.columns = [f1, f2, 'fraud_count', 'total_count', 'fraud_rate']
            combo_stats['fraud_rate'] = combo_stats['fraud_rate'] * 100

            combo_stats = combo_stats[combo_stats['total_count'] >= min_samples]

            if len(combo_stats) == 0:
                continue

            riskiest = combo_stats.nlargest(1, 'fraud_rate')

            if len(riskiest) > 0:
                row = riskiest.iloc[0]
                results.append({
                    'feature1': f1,
                    'feature2': f2,
                    'subcat1': str(row[f1]),
                    'subcat2': str(row[f2]),
                    'fraud_rate': row['fraud_rate'],
                    'sample_count': int(row['total_count']),
                    'fraud_count': int(row['fraud_count'])
                })
        except Exception as e:
            print(f" Error with {f1} x {f2}: {str(e)}")
            continue

        if idx % 100 == 0:
            print(f"Progress: {idx}/{total_pairs} pairs processed...")

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No valid combinations found!")
        return results_df

    results_df = results_df.sort_values('fraud_rate', ascending=False).reset_index(drop=True)
    results_df['combination'] = (results_df['feature1'] + ' x ' + results_df['feature2'] +
                                 ': ' + results_df['subcat1'] + ' + ' + results_df['subcat2'])

    print(f"\nAnalysis complete! Found {len(results_df)} valid combinations.")
    print(f"Top fraud rate: {results_df.iloc[0]['fraud_rate']:.1f}%")

    return results_df.head(top_n)

def create_interaction_features(df, interactions, prefix='inter'):
    """
    Create interaction features from predefined column pairs.
    Safe for train/test - no data leakage.
    
    Args:
        df: DataFrame
        interactions: list of tuples [(col1, col2), ...]
        prefix: prefix for new feature names
    
    Returns:
        df with new interaction columns
    """
    df = df.copy()
    
    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            new_name = f"{prefix}_{col1}_x_{col2}"
            df[new_name] = (
                df[col1].astype(str).fillna('missing') + '_' + 
                df[col2].astype(str).fillna('missing')
            )
    
    return df


if __name__ == '__main__':
    print("This file contains categorical analysis functions.")



