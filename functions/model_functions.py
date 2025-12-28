import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. ROC-AUC Curve and Score
# =============================================================================

def plot_roc_curve(y_true: np.ndarray, 
                   y_prob: np.ndarray, 
                   model_name: str = "Model",
                   plot: bool = True,
                   ax: Optional[plt.Axes] = None,
                   color: str = None) -> Dict:
    """
    Calculate ROC-AUC score and optionally plot the ROC curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        model_name: Name of the model (for plot legend)
        plot: Whether to display the plot
        ax: Matplotlib axes object (for multi-model comparison)
        color: Line color for the curve
    
    Returns:
        Dict containing:
            - 'auc': ROC-AUC score
            - 'fpr': False positive rates
            - 'tpr': True positive rates
            - 'thresholds': Threshold values
    
    Example:
        >>> results = plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        >>> print(f"AUC: {results['auc']:.4f}")
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    return {
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def compare_roc_curves(y_true: np.ndarray,
                       predictions: Dict[str, np.ndarray],
                       figsize: Tuple[int, int] = (10, 8)) -> pd.DataFrame:
    """
    Compare ROC curves for multiple models on the same plot.
    
    Args:
        y_true: True binary labels
        predictions: Dict of {model_name: predicted_probabilities}
        figsize: Figure size
    
    Returns:
        DataFrame with AUC scores for each model
    
    Example:
        >>> preds = {
        ...     'XGBoost': xgb_probs,
        ...     'LightGBM': lgb_probs,
        ...     'CatBoost': cat_probs
        ... }
        >>> auc_df = compare_roc_curves(y_test, preds)
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, len(predictions)))
    
    results = []
    for (name, y_prob), color in zip(predictions.items(), colors):
        result = plot_roc_curve(y_true, y_prob, model_name=name, 
                               plot=True, ax=ax, color=color)
        results.append({'Model': name, 'AUC': result['auc']})
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)


# =============================================================================
# 2. Precision-Recall Curve (Important for Imbalanced Data)
# =============================================================================

def plot_precision_recall_curve(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 model_name: str = "Model",
                                 plot: bool = True) -> Dict:
    """
    Calculate and optionally plot Precision-Recall curve.
    
    Note: For imbalanced datasets like fraud detection, PR curve is often
    more informative than ROC curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        model_name: Name of the model
        plot: Whether to display the plot
    
    Returns:
        Dict containing precision, recall, thresholds, and average precision
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, lw=2, 
                label=f'{model_name} (AP = {ap:.4f})')
        
        # Baseline: fraction of positives
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='r', linestyle='--', 
                   label=f'Baseline (Fraud Rate = {baseline:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'average_precision': ap,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }


# =============================================================================
# 3. Confusion Matrix
# =============================================================================

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str = "Model",
                          plot: bool = True,
                          normalize: bool = False,
                          figsize: Tuple[int, int] = (8, 6)) -> Dict:
    """
    Generate confusion matrix and optionally plot it.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (NOT probabilities)
        model_name: Name of the model
        plot: Whether to display the plot
        normalize: If True, show percentages instead of counts
        figsize: Figure size
    
    Returns:
        Dict containing confusion matrix and derived metrics
    
    Example:
        >>> y_pred = (y_prob >= 0.5).astype(int)
        >>> cm_results = plot_confusion_matrix(y_test, y_pred)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    metrics = {
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            cm_display = cm
            fmt = 'd'
        
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['Normal (0)', 'Fraud (1)'],
                    yticklabels=['Normal (0)', 'Fraud (1)'],
                    ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        # Add metrics annotation
        metrics_text = (
            f"Accuracy: {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall: {metrics['recall']:.4f}\n"
            f"F1-Score: {metrics['f1_score']:.4f}"
        )
        ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    return metrics


# =============================================================================
# 4. Optimal Threshold Finding
# =============================================================================

def find_optimal_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           method: str = 'youden',
                           plot: bool = True) -> Dict:
    """
    Find the optimal classification threshold using various methods.
    
    Methods:
        - 'youden': Maximizes Youden's J statistic (TPR - FPR)
          Best for balanced importance of sensitivity and specificity.
          Reference: Youden, W.J. (1950). Index for rating diagnostic tests. Cancer.
        
        - 'f1': Maximizes F1 score
          Best when precision and recall are equally important.
        
        - 'f2': Maximizes F2 score
          Best when recall is more important than precision (fraud detection).
          Weights recall 2x more than precision.
        
        - 'cost': Minimizes expected cost (requires cost_fp and cost_fn parameters)
          Best when you have specific business costs for each error type.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        method: Threshold optimization method ('youden', 'f1', 'f2')
        plot: Whether to display the threshold analysis plot
    
    Returns:
        Dict containing optimal threshold and metrics at that threshold
    
    Example:
        >>> result = find_optimal_threshold(y_test, y_prob, method='f1')
        >>> optimal_thresh = result['optimal_threshold']
        >>> y_pred_optimal = (y_prob >= optimal_thresh).astype(int)
    """
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    
    if method == 'youden':
        # Youden's J statistic: J = Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds_roc[optimal_idx]
        score_name = "Youden's J"
        optimal_score = j_scores[optimal_idx]
        
    elif method == 'f1':
        # F1 score across thresholds
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_pr[optimal_idx]
        score_name = "F1 Score"
        optimal_score = f1_scores[optimal_idx]
        
    elif method == 'f2':
        # F2 score: weights recall higher than precision
        beta = 2
        f2_scores = (1 + beta**2) * (precision[:-1] * recall[:-1]) / (beta**2 * precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds_pr[optimal_idx]
        score_name = "F2 Score"
        optimal_score = f2_scores[optimal_idx]
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'youden', 'f1', or 'f2'")
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    result = {
        'method': method,
        'optimal_threshold': optimal_threshold,
        'optimal_score': optimal_score,
        'score_name': score_name,
        'precision_at_threshold': precision_score(y_true, y_pred_optimal),
        'recall_at_threshold': recall_score(y_true, y_pred_optimal),
        'f1_at_threshold': f1_score(y_true, y_pred_optimal),
        'accuracy_at_threshold': accuracy_score(y_true, y_pred_optimal)
    }
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Score vs Threshold
        ax1 = axes[0]
        if method == 'youden':
            ax1.plot(thresholds_roc, j_scores[:-1] if len(j_scores) > len(thresholds_roc) else j_scores, 
                    'b-', lw=2, label=score_name)
        elif method == 'f1':
            ax1.plot(thresholds_pr, f1_scores, 'b-', lw=2, label=score_name)
        elif method == 'f2':
            ax1.plot(thresholds_pr, f2_scores, 'b-', lw=2, label=score_name)
        
        ax1.axvline(x=optimal_threshold, color='r', linestyle='--', 
                    label=f'Optimal = {optimal_threshold:.4f}')
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel(score_name, fontsize=12)
        ax1.set_title(f'{score_name} vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Precision-Recall Trade-off
        ax2 = axes[1]
        ax2.plot(thresholds_pr, precision[:-1], 'b-', lw=2, label='Precision')
        ax2.plot(thresholds_pr, recall[:-1], 'g-', lw=2, label='Recall')
        ax2.axvline(x=optimal_threshold, color='r', linestyle='--', 
                    label=f'Optimal = {optimal_threshold:.4f}')
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return result


# =============================================================================
# 5. Cross-Validation Evaluation
# =============================================================================

def evaluate_with_cv(model,
                     X: pd.DataFrame,
                     y: pd.Series,
                     cv: int = 5,
                     scoring_metrics: List[str] = None,
                     plot: bool = True,
                     model_name: str = "Model") -> Dict:
    """
    Evaluate model using Stratified K-Fold Cross-Validation.
    
    Args:
        model: Sklearn-compatible model (must have fit and predict_proba methods)
        X: Feature DataFrame
        y: Target Series
        cv: Number of folds (default 5)
        scoring_metrics: List of metrics to calculate (default: all)
        plot: Whether to display ROC curves for each fold
        model_name: Name of the model for display
    
    Returns:
        Dict containing:
            - Mean and std for each metric across folds
            - Fold-wise results
            - Aggregated predictions (out-of-fold)
    
    Example:
        >>> from lightgbm import LGBMClassifier
        >>> model = LGBMClassifier(n_estimators=100)
        >>> results = evaluate_with_cv(model, X_train, y_train, cv=5)
    """
    if scoring_metrics is None:
        scoring_metrics = ['auc', 'precision', 'recall', 'f1']
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    fold_results = []
    oof_predictions = np.zeros(len(y))  # Out-of-fold predictions
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, cv))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone and fit model
        model_clone = model.__class__(**model.get_params())
        
        # Suppress verbose output during CV (different models accept different values)
        model_name_lower = model_clone.__class__.__name__.lower()
        if hasattr(model_clone, 'verbose'):
            # LightGBM accepts -1, others need 0
            if 'lgbm' in model_name_lower or 'lightgbm' in model_name_lower:
                model_clone.set_params(verbose=-1)
            else:
                model_clone.set_params(verbose=0)
        
        model_clone.fit(X_train_fold, y_train_fold)
        
        # Predict probabilities
        y_prob = model_clone.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Store out-of-fold predictions
        oof_predictions[val_idx] = y_prob
        
        # Calculate metrics
        fold_metrics = {
            'fold': fold,
            'auc': roc_auc_score(y_val_fold, y_prob),
            'precision': precision_score(y_val_fold, y_pred),
            'recall': recall_score(y_val_fold, y_pred),
            'f1': f1_score(y_val_fold, y_pred),
            'accuracy': accuracy_score(y_val_fold, y_pred)
        }
        fold_results.append(fold_metrics)
        
        # Plot ROC curve for this fold
        if plot:
            fpr, tpr, _ = roc_curve(y_val_fold, y_prob)
            ax.plot(fpr, tpr, color=colors[fold-1], lw=1.5, alpha=0.7,
                    label=f'Fold {fold} (AUC = {fold_metrics["auc"]:.4f})')
    
    # Calculate mean and std for each metric
    results_df = pd.DataFrame(fold_results)
    
    summary = {}
    for metric in ['auc', 'precision', 'recall', 'f1', 'accuracy']:
        summary[f'{metric}_mean'] = results_df[metric].mean()
        summary[f'{metric}_std'] = results_df[metric].std()
    
    # Add overall OOF metrics
    oof_pred_binary = (oof_predictions >= 0.5).astype(int)
    summary['oof_auc'] = roc_auc_score(y, oof_predictions)
    summary['oof_precision'] = precision_score(y, oof_pred_binary)
    summary['oof_recall'] = recall_score(y, oof_pred_binary)
    summary['oof_f1'] = f1_score(y, oof_pred_binary)
    
    if plot:
        # Plot mean ROC curve
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        for fold_result in fold_results:
            fold_idx = fold_result['fold'] - 1
            # Interpolate each fold's TPR to common FPR points
            # (simplified - just show the overall)
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{model_name} - {cv}-Fold Cross-Validation ROC Curves\n'
                     f'Mean AUC = {summary["auc_mean"]:.4f} ± {summary["auc_std"]:.4f}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{model_name} - {cv}-Fold Cross-Validation Results")
    print(f"{'='*60}")
    print(f"AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"F1-Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"\nOut-of-Fold AUC: {summary['oof_auc']:.4f}")
    print(f"{'='*60}")
    
    return {
        'summary': summary,
        'fold_results': results_df,
        'oof_predictions': oof_predictions
    }


# =============================================================================
# 6. Comprehensive Model Evaluation Report
# =============================================================================

def full_evaluation_report(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           model_name: str = "Model",
                           threshold: float = None) -> Dict:
    """
    Generate a comprehensive evaluation report with all metrics and visualizations.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        model_name: Name of the model
        threshold: Classification threshold (if None, uses optimal F1 threshold)
    
    Returns:
        Dict containing all evaluation metrics and optimal threshold info
    
    Example:
        >>> report = full_evaluation_report(y_test, model.predict_proba(X_test)[:, 1], 
        ...                                  model_name="XGBoost")
    """
    print(f"\n{'='*70}")
    print(f"FULL EVALUATION REPORT: {model_name}")
    print(f"{'='*70}\n")
    
    # 1. Find optimal threshold if not provided
    if threshold is None:
        threshold_result = find_optimal_threshold(y_true, y_prob, method='f1', plot=True)
        threshold = threshold_result['optimal_threshold']
        print(f"\n>>> Using optimal threshold (F1-based): {threshold:.4f}\n")
    else:
        threshold_result = None
        print(f"\n>>> Using provided threshold: {threshold:.4f}\n")
    
    # 2. Generate predictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # 3. ROC-AUC
    print("\n--- ROC-AUC Analysis ---")
    roc_result = plot_roc_curve(y_true, y_prob, model_name=model_name, plot=True)
    
    # 4. Precision-Recall
    print("\n--- Precision-Recall Analysis ---")
    pr_result = plot_precision_recall_curve(y_true, y_prob, model_name=model_name, plot=True)
    
    # 5. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm_result = plot_confusion_matrix(y_true, y_pred, model_name=model_name, 
                                       plot=True, normalize=False)
    
    # 6. Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    # Compile results
    results = {
        'model_name': model_name,
        'threshold': threshold,
        'roc_auc': roc_result['auc'],
        'average_precision': pr_result['average_precision'],
        **cm_result,
        'threshold_analysis': threshold_result
    }
    
    return results


# =============================================================================
# 7. Model Comparison Summary
# =============================================================================

def compare_models(y_true: np.ndarray,
                   predictions: Dict[str, np.ndarray],
                   threshold: float = 0.5) -> pd.DataFrame:
    """
    Compare multiple models across various metrics.
    
    Args:
        y_true: True binary labels
        predictions: Dict of {model_name: predicted_probabilities}
        threshold: Classification threshold for binary metrics
    
    Returns:
        DataFrame with comparison of all models
    
    Example:
        >>> preds = {
        ...     'XGBoost': xgb_model.predict_proba(X_test)[:, 1],
        ...     'LightGBM': lgb_model.predict_proba(X_test)[:, 1],
        ...     'CatBoost': cat_model.predict_proba(X_test)[:, 1]
        ... }
        >>> comparison = compare_models(y_test, preds)
    """
    results = []
    
    for name, y_prob in predictions.items():
        y_pred = (y_prob >= threshold).astype(int)
        
        # Find optimal threshold for this model
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_thresh = thresholds[optimal_idx]
        
        y_pred_optimal = (y_prob >= optimal_thresh).astype(int)
        
        results.append({
            'Model': name,
            'ROC-AUC': roc_auc_score(y_true, y_prob),
            'Avg Precision': average_precision_score(y_true, y_prob),
            f'Precision (t={threshold})': precision_score(y_true, y_pred),
            f'Recall (t={threshold})': recall_score(y_true, y_pred),
            f'F1 (t={threshold})': f1_score(y_true, y_pred),
            'Optimal Threshold': optimal_thresh,
            'F1 (optimal)': f1_score(y_true, y_pred_optimal),
            'Recall (optimal)': recall_score(y_true, y_pred_optimal)
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        
    return comparison_df


# =============================================================================
# 8. Feature Importance Visualization
# =============================================================================

def plot_feature_importance(model,
                            feature_names: List[str],
                            top_n: int = 20,
                            importance_type: str = 'gain',
                            figsize: Tuple[int, int] = (10, 8)) -> pd.DataFrame:
    """
    Plot feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        importance_type: Type of importance ('gain', 'split', 'weight')
        figsize: Figure size
    
    Returns:
        DataFrame with feature importance values
    
    Example:
        >>> importance_df = plot_feature_importance(xgb_model, X_train.columns, top_n=20)
    """
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        # CatBoost
        importances = model.get_feature_importance()
    else:
        raise ValueError("Model doesn't have feature importance attribute")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return importance_df


# =============================================================================
# 9. Learning Curve (for detecting overfitting)
# =============================================================================

def plot_learning_curve(model,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv: int = 5,
                        train_sizes: np.ndarray = None,
                        scoring: str = 'roc_auc',
                        model_name: str = "Model") -> Dict:
    """
    Plot learning curve to diagnose overfitting/underfitting.
    
    Args:
        model: Sklearn-compatible model
        X: Feature DataFrame
        y: Target Series
        cv: Number of CV folds
        train_sizes: Array of training set sizes to evaluate
        scoring: Scoring metric
        model_name: Name of the model
    
    Returns:
        Dict containing train and validation scores
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, 
        train_sizes=train_sizes,
        cv=cv, 
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                    alpha=0.1, color='orange')
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    ax.plot(train_sizes_abs, val_mean, 'o-', color='orange', label='Validation Score')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel(f'{scoring.upper()} Score', fontsize=12)
    ax.set_title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'train_sizes': train_sizes_abs,
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std
    }


if __name__ == '__main__':
    print("Model evaluation functions loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_roc_curve(): ROC-AUC curve and score")
    print("  - compare_roc_curves(): Compare multiple models' ROC curves")
    print("  - plot_precision_recall_curve(): PR curve for imbalanced data")
    print("  - plot_confusion_matrix(): Confusion matrix with metrics")
    print("  - find_optimal_threshold(): Find best classification threshold")
    print("  - evaluate_with_cv(): Cross-validation evaluation")
    print("  - full_evaluation_report(): Comprehensive evaluation")
    print("  - compare_models(): Multi-model comparison table")
    print("  - plot_feature_importance(): Feature importance visualization")
    print("  - plot_learning_curve(): Learning curve for overfitting detection")
