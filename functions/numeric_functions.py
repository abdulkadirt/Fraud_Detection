
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
        
        # Decision logic  --> bu da gerekli olmayabilir yani okuyan kişiye bırakılması daha uygun olabilir.
        if test == 'ks':
            # KS: Higher D = better discrimination
            if p < alpha and stat > 0.1:
                decision = 'Keep'
            else:
                decision = 'Drop'
        else:
            # MW: Lower p = better
            decision = 'Keep' if p < alpha else 'Drop'
        
        results.append({
            'Feature': col,
            'Test_Stat': round(stat, 4),
            'P_Value': round(p, 6),
            'Significance': significance,
            'n_fraud': n_fraud,
            'n_normal': n_normal,
            'Unique_Ratio_Fraud': round(unique_ratio_fraud, 3),
            'Unique_Ratio_Normal': round(unique_ratio_normal, 3),
            'Decision': decision
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







if __name__ == '__main__':
    print("Numerical analysis utility functions loaded successfully!")