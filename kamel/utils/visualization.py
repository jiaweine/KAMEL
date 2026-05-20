"""
Visualization utilities for KAMEL framework.
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent image display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class VisualizationUtils:
    """Visualization utilities for KAMEL experiments."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualization utilities.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_training_curves(
        self,
        history: Dict[str, List],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot training curves (loss, accuracy, etc.).
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure
            show: Whether to show the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color=self.colors[0])
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color=self.colors[1])
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(epochs, history['val_accuracy'], label='Val Accuracy', color=self.colors[2])
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score curve
        axes[1, 0].plot(epochs, history['val_f1'], label='Val F1', color=self.colors[3])
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC curve
        axes[1, 1].plot(epochs, history['val_auc'], label='Val AUC', color=self.colors[4])
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Class names
            normalize: Whether to normalize
            save_path: Path to save figure
            show: Whether to show the plot
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=self.figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names or range(cm.shape[1]),
            yticklabels=class_names or range(cm.shape[0])
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure
            show: Whether to show the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = np.trapz(tpr, fpr)
        
        plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color=self.colors[0], linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure
            show: Whether to show the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = np.trapz(precision, recall)
        
        plt.figure(figsize=self.figsize)
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})', color=self.colors[1], linewidth=2)
        
        # Baseline (proportion of positive class)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        top_k: int = 20,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot feature importance.
        
        Args:
            importances: Feature importance values
            feature_names: Feature names
            top_k: Number of top features to show
            save_path: Path to save figure
            show: Whether to show the plot
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_k]
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]
        
        plt.figure(figsize=(12, max(6, top_k * 0.3)))
        bars = plt.barh(range(len(top_importances)), top_importances[::-1], color=self.colors[0])
        
        plt.yticks(range(len(top_importances)), top_names[::-1])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_importances[::-1])):
            plt.text(importance + 0.01 * max(top_importances), bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        title: str = 'Class Distribution',
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot class distribution.
        
        Args:
            class_counts: Dictionary of class counts
            title: Plot title
            save_path: Path to save figure
            show: Whether to show the plot
        """
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(classes, counts, color=self.colors[:len(classes)])
        
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + max(counts) * 0.01,
                    str(count), ha='center', va='bottom')
        
        # Add imbalance ratio annotation
        if len(counts) == 2:
            imbalance_ratio = min(counts) / max(counts)
            plt.text(0.02, 0.98, f'Imbalance Ratio: {imbalance_ratio:.3f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_embeddings_2d(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot 2D embeddings using dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Labels for coloring
            method: Dimensionality reduction method ('tsne', 'pca')
            save_path: Path to save figure
            show: Whether to show the plot
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=self.colors[i % len(self.colors)], label=f'Class {label}',
                       alpha=0.7, s=30)
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'2D Embeddings Visualization ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_moe_gate_weights(
        self,
        gate_weights: np.ndarray,
        expert_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot MoE gate weights distribution.
        
        Args:
            gate_weights: Gate weights (n_samples, n_experts)
            expert_names: Names of experts
            save_path: Path to save figure
            show: Whether to show the plot
        """
        n_experts = gate_weights.shape[1]
        expert_names = expert_names or [f'Expert {i+1}' for i in range(n_experts)]
        
        plt.figure(figsize=(12, 6))
        
        # Box plot of gate weights
        plt.subplot(1, 2, 1)
        plt.boxplot([gate_weights[:, i] for i in range(n_experts)], labels=expert_names)
        plt.title('Gate Weights Distribution')
        plt.ylabel('Gate Weight')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Average gate weights
        plt.subplot(1, 2, 2)
        avg_weights = np.mean(gate_weights, axis=0)
        bars = plt.bar(expert_names, avg_weights, color=self.colors[:n_experts])
        plt.title('Average Gate Weights')
        plt.ylabel('Average Weight')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, weight in zip(bars, avg_weights):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + max(avg_weights) * 0.01,
                    f'{weight:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def plot_learning_rate_schedule(
        self,
        learning_rates: List[float],
        epochs: List[int],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot learning rate schedule.
        
        Args:
            learning_rates: Learning rates over time
            epochs: Epoch numbers
            save_path: Path to save figure
            show: Whether to show the plot
        """
        plt.figure(figsize=self.figsize)
        plt.plot(epochs, learning_rates, color=self.colors[0], linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()
    
    def create_model_comparison_plot(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'f1_macro', 'auc'],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Create model comparison plot.
        
        Args:
            results_dict: Dictionary of model results
            metrics: List of metrics to compare
            save_path: Path to save figure
            show: Whether to show the plot
        """
        models = list(results_dict.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2, height + max(values) * 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.close()
        else:
            plt.close()

