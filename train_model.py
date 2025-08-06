"""
Main training script for bean lesion classification.
Trains a model for 10 epochs and saves the best result for production use.
"""

import torch
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.append('src')

from src.training.models import ModelFactory
from src.training.trainer import ModelTrainer
from src.training.dataset import create_data_loaders
from src.utils.logging_config import setup_logging, training_logger
from src.utils.config import load_training_config
from src.utils.helpers import set_seed, get_device, create_experiment_dir, save_json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train bean lesion classification model')
    
    parser.add_argument(
        '--architecture', 
        type=str, 
        default='efficientnet_b0',
        choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1', 'vgg16', 'densenet121'],
        help='Model architecture to train'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Number of epochs to train'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--pretrained', 
        action='store_true', 
        default=True,
        help='Use pretrained weights'
    )
    
    parser.add_argument(
        '--save-dir', 
        type=str, 
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.architecture}_{timestamp}"
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    training_logger.info("="*60)
    training_logger.info("🚀 BEAN LESION CLASSIFICATION - MODEL TRAINING")
    training_logger.info("="*60)
    training_logger.info(f"Experiment: {experiment_name}")
    training_logger.info(f"Architecture: {args.architecture}")
    training_logger.info(f"Epochs: {args.epochs}")
    training_logger.info(f"Batch size: {args.batch_size}")
    training_logger.info(f"Learning rate: {args.learning_rate}")
    training_logger.info(f"Pretrained: {args.pretrained}")
    training_logger.info(f"Save directory: {save_dir}")
    training_logger.info(f"Device: {get_device()}")
    training_logger.info("="*60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    training_logger.info(f"Random seed set to: {args.seed}")
    
    try:
        # Load configuration
        config = load_training_config()
        training_logger.info("✅ Configuration loaded successfully")
        
        # Create data loaders
        training_logger.info("📊 Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            train_csv="train.csv",
            val_csv="val.csv",
            batch_size=args.batch_size,
            image_size=config.data['image_size'],
            augmentation_config=config.augmentation,
            num_workers=4,  # Use multiple workers for faster data loading
            root_dir="."
        )
        training_logger.info(f"✅ Data loaders created - Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
        
        # Create model
        training_logger.info(f"🏗️ Creating {args.architecture} model...")
        model = ModelFactory.create_model(
            architecture=args.architecture,
            num_classes=3,
            pretrained=args.pretrained,
            dropout_rate=0.5
        )
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        training_logger.info(f"✅ Model created with {param_count:,} trainable parameters")
        
        # Create trainer
        training_logger.info("🎯 Initializing trainer...")
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=['healthy', 'angular_leaf_spot', 'bean_rust'],
            save_dir=str(save_dir)
        )
        training_logger.info("✅ Trainer initialized")
        
        # Save training configuration
        training_config = {
            'architecture': args.architecture,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'pretrained': args.pretrained,
            'seed': args.seed,
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'dataset_info': {
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset),
                'num_classes': 3,
                'class_names': ['healthy', 'angular_leaf_spot', 'bean_rust']
            }
        }
        
        config_path = save_dir / "training_config.json"
        save_json(training_config, str(config_path))
        training_logger.info(f"✅ Training configuration saved to: {config_path}")
        
        # Start training
        training_logger.info("🔥 Starting training...")
        training_logger.info("-" * 60)
        
        results = trainer.train(
            epochs=args.epochs,
            optimizer_config={
                'optimizer_type': 'adam',
                'learning_rate': args.learning_rate,
                'weight_decay': 0.0001
            },
            scheduler_config={
                'scheduler_type': 'step',
                'step_size': max(args.epochs // 3, 5),  # Reduce LR every 1/3 of epochs
                'gamma': 0.1
            },
            early_stopping_patience=max(args.epochs // 2, 5),  # Early stopping patience
            save_best_only=True,
            save_frequency=5
        )
        
        training_logger.info("-" * 60)
        training_logger.info("🎉 Training completed successfully!")
        
        # Print final results
        final_train_loss = results['history']['train_loss'][-1]
        final_val_loss = results['history']['val_loss'][-1]
        final_train_acc = results['history']['train_acc'][-1]
        final_val_acc = results['history']['val_acc'][-1]
        
        training_logger.info("📈 FINAL RESULTS:")
        training_logger.info(f"  Train Loss: {final_train_loss:.4f}")
        training_logger.info(f"  Train Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        training_logger.info(f"  Val Loss: {final_val_loss:.4f}")
        training_logger.info(f"  Val Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        training_logger.info(f"  Training Time: {results['total_training_time']:.2f} seconds")
        training_logger.info(f"  Final Epoch: {results['final_epoch']}")
        
        # Run comprehensive evaluation
        training_logger.info("🔍 Running final evaluation...")
        eval_metrics = trainer.evaluate(val_loader, save_plots=True)
        
        training_logger.info("📊 EVALUATION METRICS:")
        training_logger.info(f"  Accuracy: {eval_metrics['accuracy']:.4f} ({eval_metrics['accuracy']*100:.2f}%)")
        training_logger.info(f"  Precision (macro): {eval_metrics['precision_macro']:.4f}")
        training_logger.info(f"  Recall (macro): {eval_metrics['recall_macro']:.4f}")
        training_logger.info(f"  F1 Score (macro): {eval_metrics['f1_macro']:.4f}")
        
        # Per-class metrics
        training_logger.info("📋 PER-CLASS METRICS:")
        for class_name, metrics in eval_metrics['per_class'].items():
            training_logger.info(f"  {class_name.replace('_', ' ').title()}:")
            training_logger.info(f"    Precision: {metrics['precision']:.4f}")
            training_logger.info(f"    Recall: {metrics['recall']:.4f}")
            training_logger.info(f"    F1 Score: {metrics['f1']:.4f}")
        
        # Plot training history
        training_logger.info("📊 Generating training plots...")
        trainer.plot_training_history(save_path=str(save_dir / "training_history.png"))
        
        # Save comprehensive results
        final_results = {
            'training_config': training_config,
            'training_results': results,
            'evaluation_metrics': eval_metrics,
            'model_path': str(save_dir / "best_model.pth"),
            'experiment_name': experiment_name
        }
        
        results_path = save_dir / "final_results.json"
        save_json(final_results, str(results_path))
        training_logger.info(f"✅ Final results saved to: {results_path}")
        
        # Summary
        training_logger.info("="*60)
        training_logger.info("🎯 TRAINING SUMMARY")
        training_logger.info("="*60)
        training_logger.info(f"✅ Model: {args.architecture}")
        training_logger.info(f"✅ Final Validation Accuracy: {eval_metrics['accuracy']*100:.2f}%")
        training_logger.info(f"✅ Final F1 Score: {eval_metrics['f1_macro']:.4f}")
        training_logger.info(f"✅ Best Model Saved: {save_dir / 'best_model.pth'}")
        training_logger.info(f"✅ Experiment Directory: {save_dir}")
        training_logger.info("="*60)
        
        # Check if model is good enough for production
        if eval_metrics['accuracy'] > 0.8:
            training_logger.info("🌟 Model performance is excellent (>80% accuracy)!")
        elif eval_metrics['accuracy'] > 0.7:
            training_logger.info("👍 Model performance is good (>70% accuracy)")
        else:
            training_logger.info("⚠️ Model performance could be improved (<70% accuracy)")
            training_logger.info("💡 Consider training for more epochs or trying a different architecture")
        
        training_logger.info("🚀 Ready for ONNX conversion and deployment!")
        
        return True
        
    except Exception as e:
        training_logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Model training completed successfully!")
        print("The trained model is ready for ONNX conversion and deployment.")
    else:
        print("\n❌ Model training failed!")
    
    sys.exit(0 if success else 1)