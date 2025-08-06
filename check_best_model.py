"""
Script to check and compare trained models to find the best one.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to Python path
sys.path.append('src')

from src.utils.logging_config import setup_logging, training_logger


def load_model_results(model_dir: Path) -> Dict[str, Any]:
    """Load results from a model directory."""
    results_file = model_dir / "final_results.json"
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        training_logger.warning(f"Could not load results from {model_dir}: {e}")
        return None


def find_all_trained_models(models_dir: str = "models") -> List[Tuple[str, Dict[str, Any]]]:
    """Find all trained models and their results."""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        training_logger.error(f"Models directory not found: {models_path}")
        return []
    
    trained_models = []
    
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            results = load_model_results(model_dir)
            if results:
                trained_models.append((model_dir.name, results))
    
    return trained_models


def compare_models(trained_models: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Compare trained models and find the best one."""
    if not trained_models:
        training_logger.error("No trained models found!")
        return
    
    training_logger.info("="*80)
    training_logger.info("🏆 MODEL COMPARISON")
    training_logger.info("="*80)
    
    # Sort by validation accuracy
    models_by_accuracy = sorted(
        trained_models,
        key=lambda x: x[1]['evaluation_metrics']['accuracy'],
        reverse=True
    )
    
    training_logger.info(f"Found {len(trained_models)} trained models:\n")
    
    # Display comparison table
    print(f"{'Rank':<4} {'Model':<25} {'Architecture':<15} {'Val Acc':<8} {'F1 Score':<8} {'Train Time':<12}")
    print("-" * 80)
    
    for rank, (model_name, results) in enumerate(models_by_accuracy, 1):
        architecture = results['training_config']['architecture']
        val_acc = results['evaluation_metrics']['accuracy']
        f1_score = results['evaluation_metrics']['f1_macro']
        train_time = results['training_results']['total_training_time']
        
        print(f"{rank:<4} {model_name:<25} {architecture:<15} {val_acc:.4f}   {f1_score:.4f}   {train_time:.1f}s")
    
    # Best model details
    best_model_name, best_results = models_by_accuracy[0]
    
    training_logger.info(f"\n🥇 BEST MODEL: {best_model_name}")
    training_logger.info("="*50)
    
    config = best_results['training_config']
    metrics = best_results['evaluation_metrics']
    
    training_logger.info(f"Architecture: {config['architecture']}")
    training_logger.info(f"Training Time: {best_results['training_results']['total_training_time']:.1f} seconds")
    training_logger.info(f"Final Epoch: {best_results['training_results']['final_epoch']}")
    
    training_logger.info(f"\n📊 Performance Metrics:")
    training_logger.info(f"  Validation Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    training_logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    training_logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    training_logger.info(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
    
    training_logger.info(f"\n📋 Per-Class Performance:")
    for class_name, class_metrics in metrics['per_class'].items():
        training_logger.info(f"  {class_name.replace('_', ' ').title()}:")
        training_logger.info(f"    Precision: {class_metrics['precision']:.4f}")
        training_logger.info(f"    Recall: {class_metrics['recall']:.4f}")
        training_logger.info(f"    F1 Score: {class_metrics['f1']:.4f}")
    
    # Model file path
    model_path = Path("models") / best_model_name / "best_model.pth"
    training_logger.info(f"\n📁 Model File: {model_path}")
    
    if model_path.exists():
        training_logger.info("✅ Model file exists and ready for ONNX conversion")
    else:
        training_logger.error("❌ Model file not found!")
    
    # Recommendations
    training_logger.info(f"\n💡 RECOMMENDATIONS:")
    
    if metrics['accuracy'] > 0.9:
        training_logger.info("🌟 Excellent performance! Ready for production deployment.")
    elif metrics['accuracy'] > 0.8:
        training_logger.info("👍 Good performance! Suitable for most applications.")
    elif metrics['accuracy'] > 0.7:
        training_logger.info("⚠️ Moderate performance. Consider more training or data augmentation.")
    else:
        training_logger.info("🔄 Low performance. Consider different architecture or more training data.")
    
    # Next steps
    training_logger.info(f"\n🚀 NEXT STEPS:")
    training_logger.info("1. Convert the best model to ONNX format for faster inference")
    training_logger.info("2. Set up the FastAPI backend with the ONNX model")
    training_logger.info("3. Create the React frontend for image upload and classification")
    
    training_logger.info(f"\n📝 To convert to ONNX, use:")
    training_logger.info(f"   python convert_to_onnx.py --model-path {model_path} --architecture {config['architecture']}")


def main():
    """Main function to check and compare models."""
    setup_logging(log_level="INFO")
    
    # Find all trained models
    trained_models = find_all_trained_models()
    
    if not trained_models:
        training_logger.error("No trained models found in the models/ directory.")
        training_logger.info("Run 'python train_model.py' to train a model first.")
        return False
    
    # Compare models
    compare_models(trained_models)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)