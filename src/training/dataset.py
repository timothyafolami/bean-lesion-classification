"""
Dataset classes for bean lesion classification.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

from ..utils.logging_config import training_logger
from ..utils.helpers import validate_file_extension


class BeanLesionDataset(Dataset):
    """
    Dataset class for bean lesion classification.
    
    Args:
        csv_file: Path to CSV file containing image paths and labels
        root_dir: Root directory containing images
        class_names: List of class names
        transform: Optional transform to be applied on images
        validate_images: Whether to validate image integrity during initialization
    """
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str = ".",
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        validate_images: bool = True
    ):
        self.csv_file = csv_file
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names or ["healthy", "angular_leaf_spot", "bean_rust"]
        
        # Load data from CSV
        self.data_frame = self._load_csv()
        
        # Validate images if requested
        if validate_images:
            self.data_frame = self._validate_images()
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        training_logger.info(f"Loaded dataset with {len(self.data_frame)} samples")
        training_logger.info(f"Classes: {self.class_names}")
        
    def _load_csv(self) -> pd.DataFrame:
        """Load and validate CSV file."""
        try:
            df = pd.read_csv(self.csv_file)
            
            # Check required columns
            required_columns = ['image:FILE', 'category']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Rename columns for easier access
            df = df.rename(columns={'image:FILE': 'image_path', 'category': 'label'})
            
            training_logger.info(f"Loaded CSV with {len(df)} entries from {self.csv_file}")
            return df
            
        except Exception as e:
            training_logger.error(f"Error loading CSV file {self.csv_file}: {e}")
            raise
    
    def _validate_images(self) -> pd.DataFrame:
        """Validate that all images exist and are readable."""
        valid_indices = []
        invalid_count = 0
        
        for idx, row in self.data_frame.iterrows():
            image_path = self.root_dir / row['image_path']
            
            # Check if file exists
            if not image_path.exists():
                training_logger.warning(f"Image not found: {image_path}")
                invalid_count += 1
                continue
            
            # Check file extension
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            if not validate_file_extension(str(image_path), allowed_extensions):
                training_logger.warning(f"Invalid image extension: {image_path}")
                invalid_count += 1
                continue
            
            # Try to open image
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify image integrity
                valid_indices.append(idx)
            except Exception as e:
                training_logger.warning(f"Corrupted image {image_path}: {e}")
                invalid_count += 1
        
        if invalid_count > 0:
            training_logger.warning(f"Found {invalid_count} invalid images, keeping {len(valid_indices)} valid images")
        
        return self.data_frame.iloc[valid_indices].reset_index(drop=True)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        row = self.data_frame.iloc[idx]
        image_path = self.root_dir / row['image_path']
        label = int(row['label'])
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            training_logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        label_counts = Counter(self.data_frame['label'])
        class_distribution = {}
        
        for label, count in label_counts.items():
            class_name = self.class_names[label] if label < len(self.class_names) else f"unknown_{label}"
            class_distribution[class_name] = count
        
        return class_distribution
    
    def get_image_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics about images in the dataset."""
        widths, heights, file_sizes = [], [], []
        
        for idx in range(min(100, len(self.data_frame))):  # Sample first 100 images
            row = self.data_frame.iloc[idx]
            image_path = self.root_dir / row['image_path']
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
                    file_sizes.append(image_path.stat().st_size)
            except Exception:
                continue
        
        if not widths:
            return {"error": "No valid images found for statistics"}
        
        return {
            "num_samples": len(self.data_frame),
            "image_dimensions": {
                "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths)},
                "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights)}
            },
            "file_sizes": {
                "min_bytes": min(file_sizes),
                "max_bytes": max(file_sizes),
                "mean_bytes": np.mean(file_sizes)
            }
        }


def create_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """
    Create image transforms for training or validation.
    
    Args:
        image_size: Target image size (height, width)
        is_training: Whether to apply training augmentations
        augmentation_config: Configuration for data augmentation
        
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if is_training and augmentation_config:
        # Training augmentations
        if augmentation_config.get('horizontal_flip', 0) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=augmentation_config['horizontal_flip'])
            )
        
        if augmentation_config.get('vertical_flip', 0) > 0:
            transform_list.append(
                transforms.RandomVerticalFlip(p=augmentation_config['vertical_flip'])
            )
        
        if augmentation_config.get('rotation', 0) > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=augmentation_config['rotation'])
            )
        
        # Color jitter
        color_jitter_params = {}
        for param in ['brightness', 'contrast', 'saturation', 'hue']:
            if augmentation_config.get(param, 0) > 0:
                color_jitter_params[param] = augmentation_config[param]
        
        if color_jitter_params:
            transform_list.append(transforms.ColorJitter(**color_jitter_params))
        
        # Random resized crop for training
        transform_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
    else:
        # Validation transforms - just resize and center crop
        transform_list.extend([
            transforms.Resize(int(image_size[0] * 1.14)),  # Resize to slightly larger
            transforms.CenterCrop(image_size)
        ])
    
    # Common transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])
    
    return transforms.Compose(transform_list)


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 4,
    root_dir: str = "."
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        batch_size: Batch size for data loaders
        image_size: Target image size
        augmentation_config: Data augmentation configuration
        num_workers: Number of worker processes for data loading
        root_dir: Root directory containing images
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    train_transform = create_transforms(image_size, is_training=True, augmentation_config=augmentation_config)
    val_transform = create_transforms(image_size, is_training=False)
    
    # Create datasets
    train_dataset = BeanLesionDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        transform=train_transform,
        validate_images=True
    )
    
    val_dataset = BeanLesionDataset(
        csv_file=val_csv,
        root_dir=root_dir,
        transform=val_transform,
        validate_images=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # Log dataset information
    training_logger.info(f"Training dataset: {len(train_dataset)} samples")
    training_logger.info(f"Validation dataset: {len(val_dataset)} samples")
    training_logger.info(f"Training class distribution: {train_dataset.get_class_distribution()}")
    training_logger.info(f"Validation class distribution: {val_dataset.get_class_distribution()}")
    
    return train_loader, val_loader