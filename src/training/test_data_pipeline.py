"""
Test script for the data pipeline components.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.dataset import create_data_loaders, BeanLesionDataset
from src.training.data_validation import validate_dataset
from src.training.data_statistics import generate_dataset_statistics
from src.utils.logging_config import setup_logging, training_logger
from src.utils.config import load_training_config


def test_dataset_creation():
    """Test basic dataset creation."""
    training_logger.info("Testing dataset creation...")
    
    try:
        # Create a simple dataset
        dataset = BeanLesionDataset(
            csv_file="train.csv",
            root_dir=".",
            validate_images=False  # Skip validation for quick test
        )
        
        training_logger.info(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test getting a sample
        if len(dataset) > 0:
            image, label = dataset[0]
            training_logger.info(f"Sample 0: image shape={image.shape if hasattr(image, 'shape') else 'PIL Image'}, label={label}")
        
        # Test class distribution
        class_dist = dataset.get_class_distribution()
        training_logger.info(f"Class distribution: {class_dist}")
        
        return True
        
    except Exception as e:
        training_logger.error(f"Dataset creation failed: {e}")
        return False


def test_data_loaders():
    """Test data loader creation."""
    training_logger.info("Testing data loader creation...")
    
    try:
        # Load configuration
        config = load_training_config()
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_csv="train.csv",
            val_csv="val.csv",
            batch_size=4,  # Small batch for testing
            image_size=config.data['image_size'],
            augmentation_config=config.augmentation,
            num_workers=0,  # No multiprocessing for testing
            root_dir="."
        )
        
        training_logger.info(f"Data loaders created successfully")
        training_logger.info(f"Train loader: {len(train_loader)} batches")
        training_logger.info(f"Val loader: {len(val_loader)} batches")
        
        # Test getting a batch
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        training_logger.info(f"Train batch: images shape={images.shape}, labels shape={labels.shape}")
        
        val_batch = next(iter(val_loader))
        images, labels = val_batch
        training_logger.info(f"Val batch: images shape={images.shape}, labels shape={labels.shape}")
        
        return True
        
    except Exception as e:
        training_logger.error(f"Data loader creation failed: {e}")
        return False


def test_data_validation():
    """Test data validation functionality."""
    training_logger.info("Testing data validation...")
    
    try:
        # Run validation
        results = validate_dataset(root_dir=".", save_report=False, create_plots=False)
        
        training_logger.info(f"Validation completed with status: {results['summary']['overall_status']}")
        training_logger.info(f"Total errors: {results['summary']['total_errors']}")
        training_logger.info(f"Total warnings: {results['summary']['total_warnings']}")
        
        return results['summary']['overall_status'] != 'FAIL'
        
    except Exception as e:
        training_logger.error(f"Data validation failed: {e}")
        return False


def test_data_statistics():
    """Test data statistics generation."""
    training_logger.info("Testing data statistics generation...")
    
    try:
        # Generate statistics
        stats = generate_dataset_statistics(root_dir=".", save_report=False, create_plots=False)
        
        training_logger.info("Statistics generated successfully")
        
        # Log some key statistics
        if 'summary' in stats:
            summary = stats['summary']
            training_logger.info(f"Total samples: {summary['dataset_size']['total_samples']}")
            training_logger.info(f"Train/Val ratio: {summary['dataset_size']['train_val_ratio']}")
            training_logger.info(f"Balance status: {summary['class_balance']['balance_status']}")
        
        return True
        
    except Exception as e:
        training_logger.error(f"Statistics generation failed: {e}")
        return False


def main():
    """Run all data pipeline tests."""
    # Setup logging
    setup_logging(log_level="INFO")
    
    training_logger.info("Starting data pipeline tests...")
    
    # Run tests
    tests = [
        ("Dataset Creation", test_dataset_creation),
        ("Data Loaders", test_data_loaders),
        ("Data Validation", test_data_validation),
        ("Data Statistics", test_data_statistics)
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
        training_logger.info("All tests passed! Data pipeline is working correctly.")
        return True
    else:
        training_logger.error(f"{total - passed} tests failed. Please check the logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)