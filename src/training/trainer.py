"""
Training infrastructure for bean lesion classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import time
from tqdm import tqdm
import copy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.models import ModelFactory, save_model, load_model
from training.metrics import MetricsCalculator, evaluate_model_predictions
from utils.logging_config import training_logger
from utils.helpers import Timer, get_device, set_seed, save_json


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                training_logger.info("Restored best weights from early stopping")
        
        return self.early_stop


class ModelTrainer:
    """
    Comprehensive model trainer with support for multiple architectures.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        device: Optional[torch.device] = None,
        save_dir: str = "experiments"
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            class_names: List of class names
            device: Device to train on
            save_dir: Directory to save checkpoints and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device or get_device()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(class_names)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        training_logger.info(f"Trainer initialized with device: {self.device}")
        training_logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def create_optimizer(
        self,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        momentum: float = 0.9
    ) -> optim.Optimizer:
        """
        Create optimizer.
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            learning_rate: Learning rate
            weight_decay: Weight decay
            momentum: Momentum (for SGD)
            
        Returns:
            PyTorch optimizer
        """
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        training_logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = "step",
        step_size: int = 20,
        gamma: float = 0.1,
        patience: int = 10
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('step', 'plateau', 'cosine')
            step_size: Step size for StepLR
            gamma: Multiplicative factor for learning rate decay
            patience: Patience for ReduceLROnPlateau
            
        Returns:
            PyTorch scheduler or None
        """
        if scheduler_type.lower() == 'step':
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=gamma, patience=patience, verbose=True
            )
        elif scheduler_type.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=50)  # Adjust T_max as needed
        elif scheduler_type.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        training_logger.info(f"Created {scheduler_type} scheduler")
        return scheduler
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            optimizer: PyTorch optimizer
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, criterion: nn.Module, epoch: int) -> Tuple[float, float, Dict[str, Any]]:
        """
        Validate for one epoch.
        
        Args:
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy, detailed_metrics)
        """
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                # Store results
                running_loss += loss.item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                current_loss = running_loss / (len(all_targets) // self.val_loader.batch_size + 1)
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        detailed_metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)
        epoch_acc = detailed_metrics['accuracy']
        
        return epoch_loss, epoch_acc, detailed_metrics
    
    def train(
        self,
        epochs: int,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        criterion: Optional[nn.Module] = None,
        early_stopping_patience: int = 10,
        save_best_only: bool = True,
        save_frequency: int = 10
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            optimizer_config: Optimizer configuration
            scheduler_config: Scheduler configuration
            criterion: Loss function (default: CrossEntropyLoss)
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            save_frequency: Frequency to save checkpoints
            
        Returns:
            Training history and final metrics
        """
        training_logger.info(f"Starting training for {epochs} epochs")
        
        # Set up loss function
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(**optimizer_config)
        scheduler = self.create_scheduler(optimizer, **scheduler_config)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Best model tracking
        best_val_loss = float('inf')
        best_metrics = None
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate_epoch(criterion, epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            training_logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                
                if save_best_only:
                    save_model(
                        self.model,
                        str(self.save_dir / "best_model.pth"),
                        epoch,
                        optimizer.state_dict(),
                        scheduler.state_dict() if scheduler else None,
                        val_metrics
                    )
            
            # Save checkpoint periodically
            if not save_best_only and (epoch + 1) % save_frequency == 0:
                save_model(
                    self.model,
                    str(self.save_dir / f"checkpoint_epoch_{epoch+1}.pth"),
                    epoch,
                    optimizer.state_dict(),
                    scheduler.state_dict() if scheduler else None,
                    val_metrics
                )
            
            # Early stopping check
            if early_stopping(val_loss, self.model):
                training_logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        training_logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model if not save_best_only
        if not save_best_only:
            save_model(
                self.model,
                str(self.save_dir / "final_model.pth"),
                epoch,
                optimizer.state_dict(),
                scheduler.state_dict() if scheduler else None,
                val_metrics
            )
        
        # Save training history
        history_path = self.save_dir / "training_history.json"
        save_json(self.history, str(history_path))
        
        return {
            'history': self.history,
            'best_metrics': best_metrics,
            'total_training_time': total_time,
            'final_epoch': epoch + 1
        }
    
    def evaluate(self, data_loader: DataLoader, save_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            save_plots: Whether to save evaluation plots
            
        Returns:
            Evaluation metrics
        """
        training_logger.info("Starting model evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        save_dir = self.save_dir / "evaluation" if save_plots else None
        metrics = evaluate_model_predictions(
            y_true, y_pred, y_proba,
            class_names=self.class_names,
            save_dir=str(save_dir) if save_dir else None
        )
        
        training_logger.info("Evaluation completed")
        training_logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
        training_logger.info(f"Final F1 (macro): {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.history['train_loss']:
            training_logger.warning("No training history to plot")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        axes[1, 1].plot(epochs, loss_diff, 'purple')
        axes[1, 1].set_title('Validation - Training Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            training_logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()


def train_multiple_models(
    architectures: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    training_config: Dict[str, Any],
    save_dir: str = "experiments"
) -> Dict[str, Any]:
    """
    Train multiple model architectures and compare their performance.
    
    Args:
        architectures: List of architecture names to train
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
        training_config: Training configuration
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing results for all models
    """
    from training.metrics import ModelComparator
    
    training_logger.info(f"Training {len(architectures)} different architectures")
    
    results = {}
    comparator = ModelComparator(class_names)
    
    for arch in architectures:
        training_logger.info(f"\n{'='*60}")
        training_logger.info(f"Training {arch}")
        training_logger.info(f"{'='*60}")
        
        try:
            # Create model
            model = ModelFactory.create_model(
                architecture=arch,
                num_classes=len(class_names),
                **training_config.get('model', {})
            )
            
            # Create trainer
            arch_save_dir = Path(save_dir) / arch
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                save_dir=str(arch_save_dir)
            )
            
            # Train model
            training_start = time.time()
            training_results = trainer.train(
                epochs=training_config.get('epochs', 50),
                optimizer_config=training_config.get('optimizer', {}),
                scheduler_config=training_config.get('scheduler', {}),
                early_stopping_patience=training_config.get('early_stopping_patience', 10)
            )
            training_time = time.time() - training_start
            
            # Evaluate model
            eval_metrics = trainer.evaluate(val_loader, save_plots=True)
            
            # Store results
            results[arch] = {
                'training_results': training_results,
                'evaluation_metrics': eval_metrics,
                'training_time': training_time,
                'model_path': str(arch_save_dir / "best_model.pth")
            }
            
            # Add to comparator
            # We need to get predictions for comparison
            model.eval()
            all_preds, all_targets, all_probs = [], [], []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(trainer.device)
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    _, preds = torch.max(output, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            comparator.add_model_results(
                model_name=arch,
                y_true=np.array(all_targets),
                y_pred=np.array(all_preds),
                y_proba=np.array(all_probs),
                training_time=training_time
            )
            
            training_logger.info(f"✅ {arch} training completed successfully")
            
        except Exception as e:
            training_logger.error(f"❌ {arch} training failed: {e}")
            results[arch] = {'error': str(e)}
    
    # Save comparison results
    comparison_dir = Path(save_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    comparator.save_comparison_report(str(comparison_dir / "model_comparison.json"))
    comparator.plot_model_comparison(str(comparison_dir / "model_comparison.png"))
    
    # Get best model
    try:
        best_model, best_score = comparator.get_best_model('f1_macro')
        training_logger.info(f"\n🏆 Best model: {best_model} (F1: {best_score:.4f})")
        results['best_model'] = {'name': best_model, 'f1_score': best_score}
    except Exception as e:
        training_logger.warning(f"Could not determine best model: {e}")
    
    return results