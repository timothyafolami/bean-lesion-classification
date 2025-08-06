"""
Script to train multiple architectures and compare their performance.
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to Python path
sys.path.append('src')

from src.utils.logging_config import setup_logging, training_logger


def run_training(architecture, epochs=10, batch_size=32):
    """Run training for a specific architecture."""
    cmd = [
        sys.executable, 'train_model.py',
        '--architecture', architecture,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--pretrained'
    ]
    
    training_logger.info(f"🚀 Starting training for {architecture}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        training_logger.info(f"✅ {architecture} training completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        training_logger.error(f"❌ {architecture} training failed")
        training_logger.error(f"Error: {e.stderr}")
        return False, e.stderr


def main():
    """Train multiple architectures and compare results."""
    setup_logging(log_level="INFO")
    
    # Architectures to train (ordered by expected performance/speed)
    architectures = [
        'efficientnet_b0',  # Good balance of accuracy and speed
        'resnet50',         # Classic, reliable architecture
        'densenet121',      # Good accuracy, parameter efficient
        'resnet18'          # Fastest, good baseline
    ]
    
    training_logger.info("="*60)
    training_logger.info("🏭 MULTI-ARCHITECTURE TRAINING")
    training_logger.info("="*60)
    training_logger.info(f"Training {len(architectures)} architectures:")
    for arch in architectures:
        training_logger.info(f"  - {arch}")
    training_logger.info("="*60)
    
    results = {}
    successful_trainings = 0
    
    for i, architecture in enumerate(architectures, 1):
        training_logger.info(f"\n📊 Training {i}/{len(architectures)}: {architecture}")
        training_logger.info("-" * 40)
        
        success, output = run_training(architecture, epochs=10, batch_size=32)
        
        results[architecture] = {
            'success': success,
            'output': output,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            successful_trainings += 1
        
        training_logger.info(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Summary
    training_logger.info("\n" + "="*60)
    training_logger.info("📈 TRAINING SUMMARY")
    training_logger.info("="*60)
    training_logger.info(f"Successful trainings: {successful_trainings}/{len(architectures)}")
    
    for arch, result in results.items():
        status = "✅" if result['success'] else "❌"
        training_logger.info(f"{status} {arch}")
    
    # Save results
    results_file = Path("models") / "multi_architecture_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    training_logger.info(f"\n📁 Results saved to: {results_file}")
    
    if successful_trainings > 0:
        training_logger.info("\n🎉 Multi-architecture training completed!")
        training_logger.info("Check the models/ directory for trained models.")
        training_logger.info("You can now proceed with ONNX conversion for the best performing model.")
    else:
        training_logger.error("\n❌ All trainings failed!")
    
    return successful_trainings > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)