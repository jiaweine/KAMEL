"""
Metrics calculation utilities for KAMEL framework.
Comprehensive evaluation metrics for imbalanced tabular classification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, log_loss
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Comprehensive metrics calculator for imbalanced classification.
    """
    
    def __init__(self, threshold: float = 0.5, optimize_threshold: bool = True):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Default classification threshold for binary classification
            optimize_threshold: Whether to optimize threshold for best F1 score
        """
        self.threshold = threshold
        self.optimize_threshold = optimize_threshold
        self.optimal_threshold = threshold
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, 
                              metric: str = 'f1', step: float = 0.01) -> float:
        """
        Find optimal threshold for binary classification based on specified metric.
        Adaptive search for extremely imbalanced datasets.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            metric: Metric to optimize ('f1', 'precision', 'recall')
            step: Step size for threshold search
            
        Returns:
            Optimal threshold value
        """
        best_score = 0.0
        best_threshold = 0.5
        
        # Calculate imbalance ratio
        positive_ratio = np.mean(y_true)
        
        # Adaptive threshold search based on imbalance level
        if positive_ratio < 0.01:  # Extremely imbalanced (< 1%)
            # Use fine-grained search in lower range + coarse in upper range
            thresholds = np.concatenate([
                np.arange(0.001, 0.1, 0.001),    # Fine search: 0.001-0.1 step 0.001
                np.arange(0.1, 0.5, 0.02),       # Medium search: 0.1-0.5 step 0.02
                np.arange(0.5, 1.0, 0.05)        # Coarse search: 0.5-1.0 step 0.05
            ])
        elif positive_ratio < 0.05:  # Highly imbalanced (1-5%)
            # Use medium-grained search in lower range
            thresholds = np.concatenate([
                np.arange(0.001, 0.2, 0.005),    # Fine search: 0.001-0.2 step 0.005
                np.arange(0.2, 1.0, 0.02)        # Medium search: 0.2-1.0 step 0.02
            ])
        else:  # Moderately imbalanced (>5%)
            # Standard search
            thresholds = np.arange(0.01, 1.0, 0.02)
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
        verbose: bool = False,
        is_validation: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (n_samples,)
            y_pred_proba: Predicted probabilities (n_samples,) for binary or (n_samples, n_classes) for multi-class
            num_classes: Number of classes
            class_names: Names of classes
            verbose: Whether to print detailed report
            is_validation: Whether this is validation data (used for threshold optimization)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Safety check for empty arrays
        if len(y_true) == 0 or len(y_pred_proba) == 0:
            print("Warning: Empty arrays passed to metrics calculation")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'auc': 0.0, 'auprc': 0.0, 'balanced_accuracy': 0.0
            }
        
        # Ensure inputs are numpy arrays and handle shape issues
        y_true = np.asarray(y_true).flatten()  # Ensure 1D
        y_pred_proba = np.asarray(y_pred_proba)
        
        # Handle different probability formats
        if num_classes == 2:
            if y_pred_proba.ndim == 1:
                # Binary classification with single probability
                y_proba_positive = y_pred_proba
            elif y_pred_proba.ndim == 2:
                if y_pred_proba.shape[1] == 2:
                    # Binary classification with two probabilities
                    y_proba_positive = y_pred_proba[:, 1]
                elif y_pred_proba.shape[1] == 1:
                    # Single column probability
                    y_proba_positive = y_pred_proba[:, 0]
                else:
                    print(f"Warning: Unexpected 2D probability shape: {y_pred_proba.shape}")
                    y_proba_positive = y_pred_proba[:, -1]  # Use last column
            else:
                print(f"Warning: Unexpected probability array shape: {y_pred_proba.shape}")
                # Fallback: flatten and use as-is
                y_proba_positive = y_pred_proba.flatten()
                
            # Ensure same length
            if len(y_true) != len(y_proba_positive):
                min_len = min(len(y_true), len(y_proba_positive))
                y_true = y_true[:min_len]
                y_proba_positive = y_proba_positive[:min_len]
                print(f"Warning: Length mismatch fixed, using {min_len} samples")
            
            # Optimize threshold on validation data, use optimal threshold for test
            if is_validation and self.optimize_threshold:
                self.optimal_threshold = self.find_optimal_threshold(y_true, y_proba_positive)
                if verbose:
                    print(f"Optimal threshold found: {self.optimal_threshold:.3f}")
            
            # Use optimal threshold for prediction
            threshold_to_use = self.optimal_threshold if self.optimize_threshold else self.threshold
            y_pred = (y_proba_positive >= threshold_to_use).astype(int)
            
        else:
            # Multi-class classification
            if y_pred_proba.size == 0:
                print("Warning: Empty probability array for multi-class classification")
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_proba_positive = y_pred_proba
            threshold_to_use = self.threshold
        
        # Standard binary classification metrics (no averaging)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if num_classes == 2:
            # Binary classification - use pos_label=1 (positive class)
            metrics['precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        else:
            # Multi-class - use macro averaging as fallback
            metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i in range(len(precision_per_class)):
            class_name = class_names[i] if class_names else f'class_{i}'
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # ROC AUC (Area Under ROC Curve)
        try:
            if num_classes == 2:
                metrics['auc'] = roc_auc_score(y_true, y_proba_positive)
            else:
                # Multi-class AUC (one-vs-rest)
                y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
                metrics['auc'] = roc_auc_score(y_true_binarized, y_proba_positive, average='macro', multi_class='ovr')
        except ValueError as e:
            metrics['auc'] = 0.0  # In case of single class
            if verbose:
                print(f"Warning: Could not compute AUC-ROC: {e}")
        
        # AU-PRC (Area Under Precision-Recall Curve)
        try:
            if num_classes == 2:
                metrics['au_prc'] = average_precision_score(y_true, y_proba_positive)
            else:
                y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
                ap_scores = []
                for i in range(num_classes):
                    ap_scores.append(average_precision_score(y_true_binarized[:, i], y_proba_positive[:, i]))
                metrics['au_prc'] = np.mean(ap_scores)
        except ValueError as e:
            metrics['au_prc'] = 0.0
            if verbose:
                print(f"Warning: Could not compute AU-PRC: {e}")
        
        # Other metrics
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)  # Matthews Correlation Coefficient
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)  # Cohen's Kappa
        
        # Log loss
        try:
            if num_classes == 2:
                # For binary classification, need probabilities for both classes
                if y_pred_proba.ndim == 1:
                    y_pred_proba_logloss = np.column_stack([1 - y_pred_proba, y_pred_proba])
                else:
                    y_pred_proba_logloss = y_pred_proba
            else:
                y_pred_proba_logloss = y_pred_proba

            metrics['log_loss'] = log_loss(
                y_true, y_pred_proba_logloss, labels=list(range(num_classes))
            )
        except ValueError as e:
            metrics['log_loss'] = float('inf')
            if verbose:
                print(f"Warning: Could not compute log loss: {e}")
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Class-specific metrics for imbalanced learning
        metrics.update(self._compute_imbalance_metrics(y_true, y_pred, y_proba_positive, num_classes))
        
        # Classification report
        if verbose:
            target_names = class_names if class_names else [f'Class {i}' for i in range(num_classes)]
            print("\nClassification Report:")
            print(classification_report(
                y_true, y_pred,
                labels=list(range(num_classes)),
                target_names=target_names,
                zero_division=0,
            ))
            print(f"\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
        
        return metrics
    
    def _compute_imbalance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba_positive: np.ndarray,
        num_classes: int
    ) -> Dict[str, float]:
        """
        Compute metrics specifically designed for imbalanced datasets.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba_positive: Predicted probabilities
            num_classes: Number of classes
            
        Returns:
            Dictionary of imbalance-specific metrics
        """
        imbalance_metrics = {}
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        total_samples = len(y_true)
        
        # Imbalance ratio
        if num_classes == 2:
            minority_count = min(counts)
            majority_count = max(counts)
            imbalance_metrics['imbalance_ratio'] = minority_count / majority_count
        else:
            # For multi-class, use minimum/maximum ratio
            imbalance_metrics['imbalance_ratio'] = min(counts) / max(counts)
        
        # Per-class sample ratios
        for class_idx, count in class_distribution.items():
            imbalance_metrics[f'class_{class_idx}_ratio'] = count / total_samples
        
        # Sensitivity and Specificity for binary classification
        if num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            imbalance_metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
            imbalance_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
            imbalance_metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            imbalance_metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Positive and Negative Predictive Values
            imbalance_metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
            imbalance_metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # G-mean (Geometric mean of sensitivity and specificity)
        if num_classes == 2:
            sensitivity = imbalance_metrics['sensitivity']
            specificity = imbalance_metrics['specificity']
            imbalance_metrics['g_mean'] = np.sqrt(sensitivity * specificity)
        else:
            # Multi-class G-mean: geometric mean of per-class recalls
            recalls = []
            for class_idx in range(num_classes):
                class_mask = (y_true == class_idx)
                if np.sum(class_mask) > 0:
                    class_recall = np.sum((y_pred == class_idx) & class_mask) / np.sum(class_mask)
                    recalls.append(class_recall)
            
            if recalls:
                imbalance_metrics['g_mean'] = np.power(np.prod(recalls), 1/len(recalls))
            else:
                imbalance_metrics['g_mean'] = 0.0
        
        return imbalance_metrics
    
    def compute_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, List[float]]:
        """
        Compute metrics at different classification thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        threshold_metrics = {
            'thresholds': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': [],
            'g_mean': []
        }
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Skip if all predictions are the same class
            if len(np.unique(y_pred)) < 2:
                continue
            
            threshold_metrics['thresholds'].append(threshold)
            threshold_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            threshold_metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            threshold_metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            threshold_metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            
            # Specificity and G-mean for binary classification
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            threshold_metrics['specificity'].append(specificity)
            threshold_metrics['g_mean'].append(np.sqrt(sensitivity * specificity))
        
        return threshold_metrics
    

    
    def compute_cost_sensitive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute cost-sensitive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: Cost matrix (n_classes x n_classes)
            
        Returns:
            Dictionary of cost-sensitive metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Total cost
        total_cost = np.sum(cm * cost_matrix)
        
        # Average cost per sample
        avg_cost = total_cost / len(y_true)
        
        # Cost reduction compared to always predicting majority class
        if len(y_true) == 0:
            majority_class = 0
        else:
            majority_class = np.argmax(np.bincount(y_true))
        majority_cost = len(y_true) * cost_matrix[y_true, majority_class].sum() / len(y_true)
        cost_reduction = (majority_cost - avg_cost) / majority_cost if majority_cost > 0 else 0.0
        
        return {
            'total_cost': total_cost,
            'average_cost': avg_cost,
            'majority_baseline_cost': majority_cost,
            'cost_reduction': cost_reduction
        }
