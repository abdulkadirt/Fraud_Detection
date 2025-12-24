
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kstest , ks_2samp
import warnings

warnings.filterwarnings('ignore')


def get_numerical_summary(df, target='isFraud', exclude_cols=None):
    """
    Generate comprehensive summary statistics for numerical features.
    
    Purpose: Quick overview of all numerical features with fraud/non-fraud comparison
    
    Args:
        df (DataFrame): Input dataframe
        target (str): Binary target variable
        exclude_cols (list): Columns to exclude (e.g., ['TransactionID'])
    
    Returns:
        DataFrame: Summary statistics with fraud rate correlations
    """
    if exclude_cols is None:
        exclude_cols = ['TransactionID', target]
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    summary_list = []
    
    for col in numerical_cols:
        # Basic statistics
        col_data = df[col].dropna()
        
        # Fraud vs Non-Fraud comparison
        fraud_mean = df[df[target] == 1][col].mean()
        normal_mean = df[df[target] == 0][col].mean()
        
        # Correlation with target
        corr_with_target = df[[col, target]].corr().iloc[0, 1]
        
        summary_list.append({
            'Feature': col,
            'Missing_Rate': df[col].isnull().mean() * 100,
            'Mean': col_data.mean(),
            'Std': col_data.std(),
            'Min': col_data.min(),
            'Max': col_data.max(),
            'Fraud_Mean': fraud_mean,
            'Normal_Mean': normal_mean,
            'Mean_Diff': abs(fraud_mean - normal_mean),
            'Corr_with_Target': corr_with_target,
            'Unique_Values': df[col].nunique()
        })
    
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('Mean_Diff', ascending=False).reset_index(drop=True)
    
    print(f"Total numerical features: {len(numerical_cols)}")
    print(f"Features with >90% missing: {(summary_df['Missing_Rate'] > 90).sum()}")
    print(f"Features with strong correlation (|r| > 0.5): {(abs(summary_df['Corr_with_Target']) > 0.5).sum()}")
    
    return summary_df

def test_feature_discrimination(df, columns, target='isFraud', test='ks', 
                                min_samples=30, alpha=0.05):
    """
    Test if features can discriminate between fraud and normal transactions.
    
    Purpose: Identify features with statistically different distributions.
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): Numerical columns to test
        target (str): Binary target variable
        test (str): 'ks' (Kolmogorov-Smirnov) or 'mw' (Mann-Whitney U)
        min_samples (int): Minimum samples required in each class
        alpha (float): Significance level (default 0.05)
    
    Returns:
        DataFrame: Test results sorted by discriminative power
    
    Output Columns:
        - Feature: Feature name
        - Test_Stat: KS statistic (D) or MW U-statistic
        - P_Value: Statistical significance
        - Significance: ***, **, *, ns (visual indicator)
        - n_fraud / n_normal: Sample sizes
        - Unique_Ratio_Fraud / Normal: Ratio of unique values (tie detection)
        - Decision: Keep/Drop recommendation
    
    Statistical Tests Explained:
        
        KS Test (Kolmogorov-Smirnov):
        - Measures maximum vertical distance between CDFs
        - D ∈ [0, 1]: 0 = identical, 1 = completely different
        - Non-parametric (no distribution assumption)
        
        Mann-Whitney U Test:
        - Tests if one distribution is stochastically greater
        - Sensitive to median differences
        - Robust to outliers
    
    Interpretation Guide:
        
        KS Statistic (D):
        - D < 0.1:  Weak discrimination (likely noise)
        - 0.1-0.3:  Moderate discrimination
        - D > 0.3:  Strong discrimination (keep feature!)
        
        P-Value:
        - p < 0.001: Very strong evidence (***) 
        - p < 0.01:  Strong evidence (**)
        - p < 0.05:  Moderate evidence (*)
        - p >= 0.05: Not significant (ns) → Drop feature
    """
    
    fraud = df[df[target] == 1]
    normal = df[df[target] == 0]
    
    results = []
    
    for col in columns:
        if col not in df.columns or col == target:
            continue
        
        # Extract values
        fraud_vals = fraud[col].dropna()
        normal_vals = normal[col].dropna()
        
        n_fraud = len(fraud_vals)
        n_normal = len(normal_vals)
        
        # Skip if insufficient samples
        if n_fraud < min_samples or n_normal < min_samples:
            continue
        
        # Statistical test
        try:
            if test == 'ks':
                stat, p = ks_2samp(fraud_vals, normal_vals)
            elif test == 'mw':
                stat, p = mannwhitneyu(fraud_vals, normal_vals, alternative='two-sided')
            else:
                raise ValueError("test must be 'ks' or 'mw'")
        except Exception as e:
            continue
        
        # Unique ratio analysis (for tie detection)
        unique_ratio_fraud = fraud_vals.nunique() / n_fraud
        unique_ratio_normal = normal_vals.nunique() / n_normal
        
        # Significance marking --> gerekli mi gerçekten bir düşünmek lazım.
        if p < 0.001:
            significance = '***'
        elif p < 0.01:
            significance = '**'
        elif p < alpha:
            significance = '*'
        else:
            significance = 'ns'
        
        results.append({
            'Feature': col,
            'Test_Stat': round(stat, 4),
            'P_Value': round(p, 6),
            'Significance': significance,
            'n_fraud': n_fraud,
            'n_normal': n_normal,
            'Unique_Ratio_Fraud': round(unique_ratio_fraud, 3),
            'Unique_Ratio_Normal': round(unique_ratio_normal, 3),
        })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print(" No features passed minimum sample threshold")
        return results_df
    
    # Sort by test statistic  --> p-val a göre de sıralanabilir.
    ascending = True if test == 'mw' else False
    results_df = results_df.sort_values('Test_Stat', ascending=ascending).reset_index(drop=True)
    
    return results_df

def plot_distribution_comparison(df, columns, target='isFraud', 
                                 plot_type='cdf', cols_per_row=3, 
                                 figsize=(5, 4), show_stats=True):
    """
    Visualize distribution differences between fraud and normal transactions.
    
    Purpose: Visual validation of statistical tests
    WHY: CDF plots work better than density for imbalanced data
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): Features to plot
        target (str): Binary target
        plot_type (str): 'cdf', 'density', or 'hist'
        cols_per_row (int): Subplots per row
        figsize (tuple): Size per subplot
        show_stats (bool): Annotate with KS statistic
    """
    
    fraud = df[df[target] == 1]
    normal = df[df[target] == 0]
    
    n = len(columns)
    rows = int(np.ceil(n / cols_per_row))
    
    fig, axes = plt.subplots(rows, cols_per_row, 
                            figsize=(figsize[0] * cols_per_row, figsize[1] * rows))
    axes = np.array(axes).reshape(-1)
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        fvals = fraud[col].dropna()
        nvals = normal[col].dropna()
        
        if len(fvals) == 0 or len(nvals) == 0:
            ax.text(0.5, 0.5, f'No data\n{col}', ha='center', va='center')
            ax.axis('off')
            continue
        
        
        if plot_type == 'cdf':
            
            f_sorted = np.sort(fvals)
            n_sorted = np.sort(nvals)
            
            # Calculate empirical CDF
            f_cdf = np.arange(1, len(f_sorted) + 1) / len(f_sorted)
            n_cdf = np.arange(1, len(n_sorted) + 1) / len(n_sorted)
            
            
            ax.plot(f_sorted, f_cdf, label='Fraud', color='#ff6b6b', linewidth=2)
            ax.plot(n_sorted, n_cdf, label='Normal', color='#4ecdc4', linewidth=2)
            ax.set_ylabel('Cumulative Probability')
            
            # Add KS statistic annotation
            if show_stats:
                try:
                    ks_stat, p_val = ks_2samp(fvals, nvals)
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    ax.text(0.98, 0.02, f'{sig}\nKS={ks_stat:.3f}',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='bottom', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                except:
                    pass
        
        # kde :density
        elif plot_type == 'density':
            fvals.plot(kind='density', ax=ax, label='Fraud', 
                      color='#ff6b6b', linewidth=2)
            nvals.plot(kind='density', ax=ax, label='Normal', 
                      color='#4ecdc4', linewidth=2)
            ax.set_ylabel('Density')
        
        # histogram
        elif plot_type == 'hist':
            ax.hist(fvals, bins=40, density=True, alpha=0.5, 
                   label='Fraud', color='#ff6b6b')
            ax.hist(nvals, bins=40, density=True, alpha=0.5, 
                   label='Normal', color='#4ecdc4')
            ax.set_ylabel('Density')
        
        else:
            raise ValueError("plot_type must be 'cdf', 'density', or 'hist'")
        
        ax.set_title(col, fontweight='bold', fontsize=11)
        ax.set_xlabel(col, fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_boxplot_comparison(df, columns, target='isFraud', figsize=(18, 5)):
    """
    Side-by-side boxplots for fraud vs normal transactions.
    
    Purpose: Identify outliers and median differences
    WHY: Boxplots show quartiles, median, and outliers - good for understanding data spread
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): List of numerical columns to analyze
        target (str): Binary target variable
        figsize (tuple): Figure size
    
    Example:
        >>> plot_boxplot_comparison(train_df, ['TransactionAmt', 'D1', 'D2'])
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(figsize[0], figsize[1] * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Prepare data for boxplot
        plot_data = df[[col, target]].dropna()
        plot_data[target] = plot_data[target].map({0: 'Normal', 1: 'Fraud'})
        
        # Boxplot with custom colors
        sns.boxplot(data=plot_data, x=target, y=col, ax=ax, 
                   palette={'Normal': '#4ecdc4', 'Fraud': '#ff6b6b'},
                   showfliers=True)
        
        # Add median values as text
        medians = plot_data.groupby(target)[col].median()
        for xtick, median_val in zip(ax.get_xticks(), medians):
            ax.text(xtick, median_val, f'Median: {median_val:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(f'{col} - Fraud vs Normal', fontweight='bold', fontsize=11)
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()




# feature engineering functions

# v_cols
def group_by_missing_pattern(df, columns):
    """
    Groups columns that share the exact same missing-value pattern.

    Concept:
        Two columns belong to the same group if they have NaN in exactly 
        the same rows. High-dimensional datasets like IEEE-CIS V-features 
        often contain engineered feature blocks derived from the same source, 
        resulting in identical missing masks.

    Args:
        df (DataFrame): Input dataset
        columns (list): Columns to analyze (e.g., V1–V339)

    Returns:
        dict: pattern_id → { 'columns', 'size', 'missing_rate' }
    """

    missing_patterns = {}

    for col in columns:
        pattern = tuple(df[col].isnull().values)

        if pattern not in missing_patterns:
            missing_patterns[pattern] = []

        missing_patterns[pattern].append(col)

    pattern_groups = {}

    for pattern_id, (pattern, cols) in enumerate(missing_patterns.items(), start=1):
        missing_rate = sum(pattern) / len(pattern)

        pattern_groups[pattern_id] = {
            'columns': cols,
            'size': len(cols),
            'missing_rate': missing_rate
        }

    return pattern_groups


# 1. Time Engineering
def convert_dt_to_day(df):
    """Convert TransactionDT from seconds to days."""
    df['TransactionDay'] = df['TransactionDT'] / 86400
    return df

def create_d_null_features(df, d_cols=None):
    """
    Creates an indicator feature (is_null) for each D column and a total null count.
    Hypothesis: Missing values in D columns often signal first-time transactions/new users.
    """
    if d_cols is None:
        d_cols = [f'D{i}' for i in range(1, 16)]
    
    df = df.copy()
    existing_d = [c for c in d_cols if c in df.columns]
    
    # 1. Row-wise total null count
    df['D_null_count'] = df[existing_d].isnull().sum(axis=1).astype(np.int8)
    
    # 2. Individual is_null indicators
    for col in existing_d:
        df[f'{col}_is_null'] = df[col].isnull().astype(np.int8)
        
    return df

def create_d_ratio_and_diff(df, pairs=None):
    """
    Creates ratio and difference features between D columns to capture relative time gaps.
    Properly handles division by zero and replaces inf values with NaN.
    """
    if pairs is None:
        pairs = [('D1', 'D2'), ('D2', 'D3'), ('D1', 'D4'), ('D10', 'D15')]
    
    df = df.copy()
    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            # Ratio: Safe division - replace inf with NaN, then fill with median or 0
            ratio = df[col1] / df[col2].replace(0, np.nan)
            # Replace inf values with NaN
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            df[f'{col1}_{col2}_ratio'] = ratio
            
            # Diff: Absolute time gap
            df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
            
    return df

def normalize_d_columns(df, d_cols=None):
    """
    Normalize D columns by subtracting from TransactionDay.
    This anchors each timedelta to the transaction timestamp.
    """
    if d_cols is None:
        d_cols = [f'D{i}' for i in range(1, 16)]
    
    for col in d_cols:
        if col in df.columns:
            df[f'{col}_normalized'] = df['TransactionDay'] - df[col]
    return df

# 2. Transaction Amount Engineering
def extract_amt_decimal(df):
    """Extract decimal part (cents) from TransactionAmt."""
    df['TransactionAmt_decimal'] = df['TransactionAmt'] % 1
    df['TransactionAmt_cents'] = (df['TransactionAmt'] % 1 * 100).astype(int)
    return df

# 3. UID Creation --> direkt modele verme, overfit e sebep olabilir kullanıcı özelinde bilgiler bunlar.
def create_uid(df, uid_cols=['card1', 'addr1', 'D1']):
    """
    Create unique client identifier by combining card1, addr1, D1.
    """
    df['uid'] = df[uid_cols].astype(str).agg('_'.join, axis=1)
    return df


# Replace your function with this:
def create_uid_aggregations(df, uid_col='uid', agg_features=None, agg_maps=None):
    """
    UID tabanlı toplulaştırma özellikleri oluşturur.
    İşlem tutarının grup ortalamasına oranı (Magic Feature) eklenmiştir.
    """
    if agg_features is None:
        agg_features = ['TransactionAmt', 'D2', 'D15', 'C1', 'C9', 'C11', 'C13']
    
    is_train = agg_maps is None
    if is_train:
        agg_maps = {}
    
    for feat in agg_features:
        if feat not in df.columns:
            continue
        
        mean_col = f'{feat}_{uid_col}_mean'
        std_col = f'{feat}_{uid_col}_std'
        
        if is_train:
            # TRAIN: İstatistikleri hesapla ve sakla
            agg_stats = df.groupby(uid_col)[feat].agg(['mean', 'std'])
            agg_maps[feat] = {
                'mean_map': agg_stats['mean'].to_dict(),
                'std_map': agg_stats['std'].to_dict(),
                'global_mean': df[feat].mean(),
                'global_std': df[feat].std()
            }
            df[mean_col] = df.groupby(uid_col)[feat].transform('mean')
            df[std_col] = df.groupby(uid_col)[feat].transform('std')
        else:
            # TEST: Train'den gelen haritaları uygula
            df[mean_col] = df[uid_col].map(agg_maps[feat]['mean_map'])
            df[std_col] = df[uid_col].map(agg_maps[feat]['std_map'])
            
            # Yeni görülen UID'leri Train genel ortalaması ile doldur (Data Leakage Önlemi)
            df[mean_col] = df[mean_col].fillna(agg_maps[feat]['global_mean'])
            df[std_col] = df[std_col].fillna(agg_maps[feat]['global_std'])

    # MAGIC FEATURE: İşlem tutarının o kullanıcının ortalamasına oranı
    # Bu özellik döngünün dışında hesaplanmalı çünkü sadece 'TransactionAmt' için geçerli.
    mean_amt_col = f'TransactionAmt_{uid_col}_mean'
    if mean_amt_col in df.columns:
        df['Amt_to_mean_uid'] = df['TransactionAmt'] / (df[mean_amt_col] + 1e-5)
        
    return df, agg_maps


# # 4. UID-based Aggregations
# def create_uid_aggregations(df, uid_col='uid', agg_features=None, agg_maps=None):
#     """
#     Create aggregation features based on UID.
#     Each UID represents a unique client's behavior pattern.
    
#     Train/Test Safe Implementation:
#         - On train (agg_maps=None): Compute and store aggregation statistics
#         - On test (agg_maps provided): Apply pre-computed statistics via mapping
    
#     This prevents data leakage by ensuring test aggregations come from train statistics.
#     UIDs not seen in train will get NaN (can be filled with global mean later).
    
#     Reference:
#         This approach follows the standard practice of fitting transformations on 
#         training data only, as described in:
#         - Kuhn & Johnson (2013). Applied Predictive Modeling. Springer.
#         - Zheng & Casari (2018). Feature Engineering for Machine Learning. O'Reilly.
    
#     Args:
#         df: DataFrame
#         uid_col: Column containing unique identifiers
#         agg_features: List of features to aggregate (default: transaction-related features)
#         agg_maps: Dict of pre-computed aggregation mappings from train (for test data)
    
#     Returns:
#         df: DataFrame with aggregation features
#         agg_maps: Dictionary containing {feature: {uid: (mean, std)}} mappings
#     """
#     if agg_features is None:
#         agg_features = ['TransactionAmt', 'D2', 'D15', 'C1', 'C9', 'C11', 'C13']
    
#     is_train = agg_maps is None
#     if is_train:
#         agg_maps = {}
    
#     for feat in agg_features:
#         if feat not in df.columns:
#             continue
        
#         mean_col = f'{feat}_{uid_col}_mean'
#         std_col = f'{feat}_{uid_col}_std'
        
#         if is_train:
#             # Compute aggregations on train data
#             agg_stats = df.groupby(uid_col)[feat].agg(['mean', 'std'])
#             agg_maps[feat] = {
#                 'mean_map': agg_stats['mean'].to_dict(),
#                 'std_map': agg_stats['std'].to_dict(),
#                 'global_mean': df[feat].mean(),
#                 'global_std': df[feat].std()
#             }
#             df[mean_col] = df.groupby(uid_col)[feat].transform('mean')
#             df[std_col] = df.groupby(uid_col)[feat].transform('std')
#         else:
#             # Apply pre-computed mappings to test data
#             df[mean_col] = df[uid_col].map(agg_maps[feat]['mean_map'])
#             df[std_col] = df[uid_col].map(agg_maps[feat]['std_map'])
            
#             # Fill unseen UIDs with global statistics from train
#             df[mean_col] = df[mean_col].fillna(agg_maps[feat]['global_mean'])
#             df[std_col] = df[std_col].fillna(agg_maps[feat]['global_std'])
    
#     return df, agg_maps 


# 5. C-Column Velocity Features
def create_c_velocity_features(df):
    """
    C columns represent counts. Ratios capture transaction velocity.
    Example: C13/C1 = total operations / time since first operation
    """
    if 'C1' in df.columns and 'C13' in df.columns:
        df['C13_C1_ratio'] = df['C13'] / (df['C1'] + 1)
    
    if 'C1' in df.columns and 'C2' in df.columns:
        df['C2_C1_ratio'] = df['C2'] / (df['C1'] + 1)
    
    return df


# 6. Feature Evaluation
def test_single_feature(df, feature, target='isFraud'):
    """
    Quick AUC test for evaluating a single feature's predictive power.
    
    Uses a simple LightGBM model with train/validation split to estimate
    how well a single feature can predict the target variable.
    
    Note: This is a quick sanity check, not a rigorous evaluation.
    For proper feature importance, use cross-validation or permutation importance.
    
    Args:
        df: DataFrame containing the feature and target
        feature: Name of the feature to test
        target: Name of the target column (default 'isFraud')
    
    Returns:
        float: ROC-AUC score on validation set
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    
    valid_data = df[[feature, target]].dropna()
    X_train, X_val, y_train, y_val = train_test_split(
        valid_data[[feature]], valid_data[target], 
        test_size=0.3, random_state=42, stratify=valid_data[target]
    )
    
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    
    return auc


# 7. Correlation-based Feature Selection
def remove_high_correlation(df, features, ks_results_df, threshold=0.95):
    """
    Remove highly correlated features while preserving discriminative power.
    
    Method:
    -------
    When two features have correlation > threshold, keep the one with higher
    KS statistic (better fraud/normal separation). This is a target-aware
    feature selection method performed ONLY on training data.
    
    Academic Justification:
    ----------------------
    Using target information for feature selection on training data is standard
    practice in supervised learning, as long as:
    1. Selection is done ONLY on training fold (not test/validation)
    2. The same selected features are applied to test data
    
    References:
    - Guyon, I. & Elisseeff, A. (2003). "An Introduction to Variable and Feature 
      Selection". JMLR 3:1157-1182.
    - Chandrashekar, G. & Sahin, F. (2014). "A Survey on Feature Selection Methods". 
      Computers & Electrical Engineering.
    
    Args:
        df: DataFrame (training data)
        features: List of features to check for correlation
        ks_results_df: DataFrame with KS test results (Feature, Test_Stat columns)
        threshold: Correlation threshold (default 0.95)
    
    Returns:
        List of features to keep after removing highly correlated ones
    """
    # Build KS lookup dictionary for efficiency
    ks_lookup = dict(zip(ks_results_df['Feature'], ks_results_df['Test_Stat']))
    
    # Filter features that exist in both df and ks_results
    valid_features = [f for f in features if f in df.columns and f in ks_lookup]
    
    if len(valid_features) == 0:
        print("Warning: No valid features found for correlation analysis")
        return features
    
    corr_matrix = df[valid_features].corr().abs()
    
    # Upper triangle to avoid duplicates
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = set()
    
    for column in upper.columns:
        correlated = upper[column][upper[column] > threshold].index.tolist()
        if correlated:
            for corr_col in correlated:
                ks_col = ks_lookup.get(column, 0)
                ks_corr = ks_lookup.get(corr_col, 0)
                
                # Keep the feature with higher KS statistic
                if ks_col < ks_corr:
                    to_drop.add(column)
                else:
                    to_drop.add(corr_col)
    
    print(f"Removing {len(to_drop)} highly correlated features (r > {threshold})")
    print(f"Dropped features: {list(to_drop)[:10]}{'...' if len(to_drop) > 10 else ''}")
    
    return [f for f in valid_features if f not in to_drop]


# 8. Outlier Handling
def cap_outliers(df, columns, lower_percentile=1, upper_percentile=99):
    """
    Cap extreme outliers at specified percentiles (Winsorization).
    
    Method:
    -------
    Values below the lower percentile are set to the lower percentile value.
    Values above the upper percentile are set to the upper percentile value.
    
    This is a robust method that preserves the distribution shape while
    reducing the influence of extreme values on model training.
    
    Reference:
    - Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of 
      Statistical Learning". Springer. Section 9.6 on robust methods.
    
    Args:
        df: DataFrame
        columns: List of columns to cap
        lower_percentile: Lower bound percentile (default 1)
        upper_percentile: Upper bound percentile (default 99)
    
    Returns:
        DataFrame with capped values
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        lower = df[col].quantile(lower_percentile / 100)
        upper = df[col].quantile(upper_percentile / 100)
        df[col] = df[col].clip(lower, upper)
    return df


if __name__ == '__main__':
    print("Numerical analysis utility functions loaded successfully!")