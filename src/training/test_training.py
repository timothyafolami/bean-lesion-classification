"""
Test script for the training infrastructure.
"""

import sys
from pathlib import Path
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.training.models import ModelFactory, create_model_from_config
from src.training.trainer import ModelTrainer, train_multiple_models
from src.training.metrics import MetricsCalculator, ModelComparator
from src.training.dataset import create_data_loaders
from src.utils.logging_config import setup_logging, training_logger
from src.utils.config import load_training_config
from src.utils.helpers import set_seed, get_device


def test_model_factory():
    """Test model factory functionality."""
    training_logger.info("Testing ModelFactory...")
    
    try:
        # Test supported architectures
        # architectures = ['resnet50', 'efficientnet_b0', 'vgg16', 'densenet121']
        architectures = ['resnet50']
        
        for arch in architectures:
            training_logger.info(f"Testing {arch}...")
            
            # Create model
            model = ModelFactory.create_model(
                architecture=arch,
                num_classes=3,
                pretrained=False,  # Use False for faster testing
                dropout_rate=0.5
            )
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            
            assert output.shape == (2, 3), f"Expected output shape (2, 3), got {output.shape}"
            training_logger.info(f"✅ {arch} test passed - output shape: {output.shape}")
        
        # Test model info
        info = ModelFactory.get_model_info('resnet50')
        training_logger.info(f"ResNet50 info: {info}")
        
        return True
        
    except Exception as e:
        training_logger.error(f"ModelFactory test failed: {e}")
        return False


def test_metrics_calculator():
    """Test metrics calculation."""
    training_logger.info("Testing MetricsCalculator...")
    
    try:
        import numpy as np
        
        # Create dummy predictions
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        y_proba = np.random.rand(100, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize to probabilities
        
        class_names = ['healthy', 'angular_leaf_spot', 'bean_rust']
        calculator = MetricsCalculator(class_names)
        
        # Calculate metrics
        metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        # Check required metrics exist
        required_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            training_logger.info(f"{metric}: {metrics[metric]:.4f}")
        
        training_logger.info("✅ MetricsCalculator test passed")
        return True
        
    except Exception as e:
        training_logger.error(f"MetricsCalculator test failed: {e}")
        return False


def test_model_comparator():
    """Test model comparison functionality."""
    training_logger.info("Testing ModelComparator...")
    
    try:
        import numpy as np
        
        class_names = ['healthy', 'angular_leaf_spot', 'bean_rust']
        comparator = ModelComparator(class_names)
        
        # Add dummy results for two models
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 50)
        
        for model_name in ['resnet50', 'efficientnet_b0']:
            y_pred = np.random.randint(0, 3, 50)
            y_proba = np.random.rand(50, 3)
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
            
            comparator.add_model_results(
                model_name=model_name,
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                training_time=100.0,
                inference_time=0.05
            )
        
        # Get comparison table
        comparison = comparator.get_comparison_table()
        assert 'models' in comparison, "Missing models in comparison"
        assert len(comparison['models']) == 2, "Expected 2 models in comparison"
        
        # Get best model
        best_model, best_score = comparator.get_best_model('accuracy')
        training_logger.info(f"Best model: {best_model} (accuracy: {best_score:.4f})")
        
        training_logger.info("✅ ModelComparator test passed")
        return True
        
    except Exception as e:
        training_logger.error(f"ModelComparator test failed: {e}")
        return False


def test_trainer_setup():
    """Test trainer setup without full training."""
    training_logger.info("Testing ModelTrainer setup...")
    
    try:
        # Load config
        config = load_training_config()
        
        # Create small data loaders for testing
        train_loader, val_loader = create_data_loaders(
            train_csv="train.csv",
            val_csv="val.csv",
            batch_size=4,
            image_size=config.data['image_size'],
            augmentation_config=config.augmentation,
            num_workers=0,
            root_dir="."
        )
        
        # Create model
        model = ModelFactory.create_model(
            architecture='resnet50',
            num_classes=3,
            pretrained=False
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=['healthy', 'angular_leaf_spot', 'bean_rust'],
            save_dir="test_experiments"
        )
        
        # Test optimizer creation
        optimizer = trainer.create_optimizer(
            optimizer_type="adam",
            learning_rate=0.001
        )
        assert optimizer is not None, "Failed to create optimizer"
        
        # Test scheduler creation
        scheduler = trainer.create_scheduler(
            optimizer=optimizer,
            scheduler_type="step"
        )
        assert scheduler is not None, "Failed to create scheduler"
        
        training_logger.info("✅ ModelTrainer setup test passed")
        return True
        
    except Exception as e:
        training_logger.error(f"ModelTrainer setup test failed: {e}")
        return False


def test_mini_training():
    """Test a mini training loop with 1 epoch."""
    training_logger.info("Testing mini training loop...")
    
    try:
        # Set seed for reproducibility
        set_seed(42)
        
        # Load config
        config = load_training_config()
        
        # Create small data loaders
        train_loader, val_loader = create_data_loaders(
            train_csv="train.csv",
            val_csv="val.csv",
            batch_size=8,
            image_size=(64, 64),  # Smaller images for faster testing
            augmentation_config=None,  # No augmentation for testing
            num_workers=0,
            root_dir="."
        )
        
        # Create small model
        model = ModelFactory.create_model(
            architecture='resnet18',  # Smaller model
            num_classes=3,
            pretrained=False
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=['healthy', 'angular_leaf_spot', 'bean_rust'],
            save_dir="test_mini_training"
        )
        
        # Train for 1 epoch
        results = trainer.train(
            epochs=1,
            optimizer_config={
                'optimizer_type': 'adam',
                'learning_rate': 0.001
            },
            scheduler_config={
                'scheduler_type': 'none'
            },
            early_stopping_patience=10,
            save_best_only=True
        )
        
        # Check results
        assert 'history' in results, "Missing training history"
        assert len(results['history']['train_loss']) == 1, "Expected 1 epoch in history"
        
        training_logger.info(f"Mini training completed:")
        training_logger.info(f"  Train loss: {results['history']['train_loss'][0]:.4f}")
        training_logger.info(f"  Val loss: {results['history']['val_loss'][0]:.4f}")
        training_logger.info(f"  Train acc: {results['history']['train_acc'][0]:.4f}")
        training_logger.info(f"  Val acc: {results['history']['val_acc'][0]:.4f}")
        
        training_logger.info("✅ Mini training test passed")
        return True
        
    except Exception as e:
        training_logger.error(f"Mini training test failed: {e}")
        return False


def main():
    """Run all training infrastructure tests."""
    # Setup logging
    setup_logging(log_level="INFO")
    
    training_logger.info("Starting training infrastructure tests...")
    
    # Run tests
    tests = [
        ("Model Factory", test_model_factory),
        ("Metrics Calculator", test_metrics_calculator),
        ("Model Comparator", test_model_comparator),
        ("Trainer Setup", test_trainer_setup),
        ("Mini Training", test_mini_training)
    ]
    
    results = {}
    for test_name, test_func in tests:
        training_logger.info(f"\n{'='*50}")
        training_logger.info(f"Running test: {test_name}")
        training_logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            training_logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    training_logger.info(f"\n{'='*50}")
    training_logger.info("TEST SUMMARY")
    training_logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        training_logger.info(f"{test_name}: {status}")
    
    training_logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        training_logger.info("🎉 All training infrastructure tests passed!")
        return True
    else:
        training_logger.error(f"❌ {total - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)