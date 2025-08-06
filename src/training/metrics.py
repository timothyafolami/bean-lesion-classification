"""
Metrics calculation utilities for model evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_config import training_logger
from utils.helpers import save_json


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification tasks.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Weighted metrics (better for imbalanced datasets)
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # AUC metrics (if probabilities provided)
        if y_proba is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    # Multi-class classification
                    metrics['auc_roc_ovr'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
                    metrics['auc_roc_ovo'] = float(roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro'))
            except Exception as e:
                training_logger.warning(f"Could not calculate AUC metrics: {e}")
        
        return metrics
    
    def calculate_top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        k: int = 2
    ) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = 0
        
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            training_logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Convert to binary format for each class
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {auc:.2f})',
                linewidth=2
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            training_logger.info(f"ROC curves saved to: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot precision-recall curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Convert to binary format for each class
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Plot PR curve for each class
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            
            plt.plot(
                recall, precision,
                label=f'{class_name}',
                linewidth=2
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Multi-class Classification')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            training_logger.info(f"Precision-recall curves saved to: {save_path}")
        
        plt.show()


class ModelComparator:
    """
    Compare performance of multiple models.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize model comparator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.metrics_calculator = MetricsCalculator(class_names)
        self.model_results = {}
    
    def add_model_results(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        training_time: Optional[float] = None,
        inference_time: Optional[float] = None
    ) -> None:
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            training_time: Training time in seconds
            inference_time: Average inference time per sample
        """
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        self.model_results[model_name] = {
            'metrics': metrics,
            'training_time': training_time,
            'inference_time': inference_time,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist() if y_proba is not None else None
            }
        }
        
        training_logger.info(f"Added results for model: {model_name}")
        training_logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        training_logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    
    def get_comparison_table(self) -> Dict[str, Any]:
        """
        Get comparison table of all models.
        
        Returns:
            Dictionary containing comparison metrics
        """
        if not self.model_results:
            return {}
        
        comparison = {
            'models': list(self.model_results.keys()),
            'metrics': {}
        }
        
        # Key metrics to compare
        key_metrics = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_weighted', 'recall_weighted', 'f1_weighted'
        ]
        
        for metric in key_metrics:
            comparison['metrics'][metric] = {}
            values = []
            
            for model_name in comparison['models']:
                value = self.model_results[model_name]['metrics'][metric]
                comparison['metrics'][metric][model_name] = value
                values.append(value)
            
            # Add statistics
            comparison['metrics'][metric]['best_model'] = comparison['models'][np.argmax(values)]
            comparison['metrics'][metric]['best_value'] = max(values)
            comparison['metrics'][metric]['mean'] = np.mean(values)
            comparison['metrics'][metric]['std'] = np.std(values)
        
        # Add timing information if available
        comparison['timing'] = {}
        for model_name in comparison['models']:
            timing = {}
            if self.model_results[model_name]['training_time']:
                timing['training_time'] = self.model_results[model_name]['training_time']
            if self.model_results[model_name]['inference_time']:
                timing['inference_time'] = self.model_results[model_name]['inference_time']
            
            if timing:
                comparison['timing'][model_name] = timing
        
        return comparison
    
    def plot_model_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot comparison of model performance.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.model_results:
            training_logger.warning("No model results to plot")
            return
        
        # Prepare data for plotting
        models = list(self.model_results.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        data = []
        for metric in metrics:
            for model in models:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': self.model_results[model]['metrics'][metric]
                })
        
        # Create plot
        import pandas as pd
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Model', y='Value', hue='Metric')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            training_logger.info(f"Model comparison plot saved to: {save_path}")
        
        plt.show()
    
    def save_comparison_report(self, save_path: str) -> None:
        """
        Save comprehensive comparison report.
        
        Args:
            save_path: Path to save the report
        """
        report = {
            'timestamp': str(np.datetime64('now')),
            'class_names': self.class_names,
            'model_results': self.model_results,
            'comparison_table': self.get_comparison_table()
        }
        
        save_json(report, save_path)
        training_logger.info(f"Comparison report saved to: {save_path}")
    
    def get_best_model(self, metric: str = 'f1_macro') -> Tuple[str, float]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.model_results:
            raise ValueError("No model results available")
        
        best_model = None
        best_value = -1
        
        for model_name, results in self.model_results.items():
            value = results['metrics'][metric]
            if value > best_value:
                best_value = value
                best_model = model_name
        
        return best_model, best_value


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        class_names: List of class names
        save_dir: Directory to save plots and reports
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    calculator = MetricsCalculator(class_names)
    metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix plot
        calculator.plot_confusion_matrix(
            y_true, y_pred,
            save_path=str(save_dir / "confusion_matrix.png")
        )
        
        # Save ROC curves if probabilities available
        if y_proba is not None:
            calculator.plot_roc_curves(
                y_true, y_proba,
                save_path=str(save_dir / "roc_curves.png")
            )
            
            calculator.plot_precision_recall_curves(
                y_true, y_proba,
                save_path=str(save_dir / "precision_recall_curves.png")
            )
        
        # Save metrics report
        save_json(metrics, str(save_dir / "evaluation_metrics.json"))
    
    return metrics