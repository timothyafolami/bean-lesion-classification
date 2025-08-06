"""
Model factory and architecture definitions for bean lesion classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_config import training_logger
from utils.helpers import count_parameters


class ModelFactory:
    """
    Factory class for creating different CNN architectures.
    """
    
    SUPPORTED_ARCHITECTURES = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
    }
    
    @classmethod
    def create_model(
        self,
        architecture: str,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ) -> nn.Module:
        """
        Create a model with the specified architecture.
        
        Args:
            architecture: Name of the architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for the classifier
            freeze_backbone: Whether to freeze backbone parameters
            
        Returns:
            PyTorch model
        """
        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                           f"Supported: {list(self.SUPPORTED_ARCHITECTURES.keys())}")
        
        training_logger.info(f"Creating {architecture} model with {num_classes} classes")
        
        # Create base model
        model_fn = self.SUPPORTED_ARCHITECTURES[architecture]
        
        # Handle different torchvision versions
        try:
            if pretrained:
                if hasattr(models, f"{architecture.upper()}_Weights"):
                    # New torchvision API (>= 0.13)
                    weights = getattr(models, f"{architecture.upper()}_Weights").DEFAULT
                    model = model_fn(weights=weights)
                else:
                    # Old torchvision API
                    model = model_fn(pretrained=True)
            else:
                model = model_fn(pretrained=False)
        except Exception:
            # Fallback to old API
            model = model_fn(pretrained=pretrained)
        
        # Modify the classifier based on architecture family
        if architecture.startswith('resnet'):
            model = self._modify_resnet_classifier(model, num_classes, dropout_rate)
        elif architecture.startswith('efficientnet'):
            model = self._modify_efficientnet_classifier(model, num_classes, dropout_rate)
        elif architecture.startswith('vgg'):
            model = self._modify_vgg_classifier(model, num_classes, dropout_rate)
        elif architecture.startswith('densenet'):
            model = self._modify_densenet_classifier(model, num_classes, dropout_rate)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone(model, architecture)
        
        # Log model info
        param_count = count_parameters(model)
        training_logger.info(f"Model created with {param_count:,} trainable parameters")
        
        return model
    
    @staticmethod
    def _modify_resnet_classifier(model: nn.Module, num_classes: int, dropout_rate: float) -> nn.Module:
        """Modify ResNet classifier."""
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    
    @staticmethod
    def _modify_efficientnet_classifier(model: nn.Module, num_classes: int, dropout_rate: float) -> nn.Module:
        """Modify EfficientNet classifier."""
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    
    @staticmethod
    def _modify_vgg_classifier(model: nn.Module, num_classes: int, dropout_rate: float) -> nn.Module:
        """Modify VGG classifier."""
        # VGG has a more complex classifier, replace the last layer
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        # Add dropout to the classifier if not present
        if not any(isinstance(layer, nn.Dropout) for layer in model.classifier):
            # Insert dropout before the last linear layer
            classifier_layers = list(model.classifier.children())
            classifier_layers.insert(-1, nn.Dropout(dropout_rate))
            model.classifier = nn.Sequential(*classifier_layers)
        
        return model
    
    @staticmethod
    def _modify_densenet_classifier(model: nn.Module, num_classes: int, dropout_rate: float) -> nn.Module:
        """Modify DenseNet classifier."""
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    
    @staticmethod
    def _freeze_backbone(model: nn.Module, architecture: str) -> None:
        """Freeze backbone parameters for transfer learning."""
        training_logger.info(f"Freezing backbone parameters for {architecture}")
        
        if architecture.startswith('resnet'):
            # Freeze all layers except fc
            for name, param in model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False
        elif architecture.startswith('efficientnet'):
            # Freeze all layers except classifier
            for name, param in model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
        elif architecture.startswith('vgg'):
            # Freeze features, keep classifier trainable
            for param in model.features.parameters():
                param.requires_grad = False
        elif architecture.startswith('densenet'):
            # Freeze all layers except classifier
            for name, param in model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
    
    @classmethod
    def get_model_info(cls, architecture: str) -> Dict[str, Any]:
        """Get information about a model architecture."""
        if architecture not in cls.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Create a dummy model to get info
        model = cls.create_model(architecture, num_classes=3, pretrained=False)
        
        return {
            'architecture': architecture,
            'total_parameters': count_parameters(model),
            'input_size': (3, 224, 224),  # Standard ImageNet input size
            'supported': True
        }
    
    @classmethod
    def list_supported_architectures(cls) -> Dict[str, Dict[str, Any]]:
        """List all supported architectures with their info."""
        architectures_info = {}
        
        for arch in cls.SUPPORTED_ARCHITECTURES.keys():
            try:
                architectures_info[arch] = cls.get_model_info(arch)
            except Exception as e:
                training_logger.warning(f"Could not get info for {arch}: {e}")
                architectures_info[arch] = {
                    'architecture': arch,
                    'supported': False,
                    'error': str(e)
                }
        
        return architectures_info


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple models for improved performance.
    """
    
    def __init__(self, models: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None):
        """
        Initialize model ensemble.
        
        Args:
            models: Dictionary of model_name -> model
            weights: Optional weights for each model in ensemble
        """
        super().__init__()
        self.models = nn.ModuleDict(models)
        
        # Set equal weights if not provided
        if weights is None:
            weights = {name: 1.0 / len(models) for name in models.keys()}
        
        self.weights = weights
        training_logger.info(f"Created ensemble with {len(models)} models: {list(models.keys())}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []
        
        for name, model in self.models.items():
            output = model(x)
            # Apply softmax to get probabilities
            output = torch.softmax(output, dim=1)
            # Weight the output
            output = output * self.weights[name]
            outputs.append(output)
        
        # Average the weighted outputs
        ensemble_output = torch.stack(outputs).sum(dim=0)
        
        # Convert back to logits for loss calculation
        ensemble_output = torch.log(ensemble_output + 1e-8)
        
        return ensemble_output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions from ensemble."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


def save_model(
    model: nn.Module,
    filepath: str,
    epoch: int,
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    metrics: Optional[Dict[str, float]] = None,
    model_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint with additional information.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the checkpoint
        epoch: Current epoch number
        optimizer_state: Optimizer state dict
        scheduler_state: Scheduler state dict
        metrics: Training metrics
        model_config: Model configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_config': model_config or {},
        'metrics': metrics or {}
    }
    
    if optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if scheduler_state:
        checkpoint['scheduler_state_dict'] = scheduler_state
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    training_logger.info(f"Model checkpoint saved to: {filepath}")


def load_model(
    filepath: str,
    architecture: str,
    num_classes: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model from checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        architecture: Model architecture name
        num_classes: Number of classes
        device: Device to load the model on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model
    model_config = checkpoint.get('model_config', {})
    model = ModelFactory.create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,  # Don't use pretrained when loading from checkpoint
        **model_config
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    training_logger.info(f"Model loaded from: {filepath}")
    training_logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, checkpoint


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        PyTorch model
    """
    model_config = config.get('model', {})
    
    architecture = model_config.get('architecture', 'resnet50')
    if isinstance(architecture, list):
        architecture = architecture[0]  # Take first architecture if list provided
    
    return ModelFactory.create_model(
        architecture=architecture,
        num_classes=model_config.get('num_classes', 3),
        pretrained=model_config.get('pretrained', True),
        dropout_rate=model_config.get('dropout_rate', 0.5),
        freeze_backbone=model_config.get('freeze_backbone', False)
    )