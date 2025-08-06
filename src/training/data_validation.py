"""
Data validation utilities for bean lesion classification dataset.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging_config import training_logger
from ..utils.helpers import format_bytes, save_json


class DataValidator:
    """
    Comprehensive data validation for the bean lesion dataset.
    """
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.validation_results = {}
        
    def validate_csv_files(self, csv_files: List[str]) -> Dict[str, any]:
        """
        Validate CSV files structure and content.
        
        Args:
            csv_files: List of CSV file paths to validate
            
        Returns:
            Dictionary containing validation results
        """
        results = {}
        
        for csv_file in csv_files:
            csv_path = self.root_dir / csv_file
            file_results = {
                "exists": csv_path.exists(),
                "errors": [],
                "warnings": [],
                "stats": {}
            }
            
            if not file_results["exists"]:
                file_results["errors"].append(f"CSV file not found: {csv_path}")
                results[csv_file] = file_results
                continue
            
            try:
                # Load CSV
                df = pd.read_csv(csv_path)
                
                # Check required columns
                required_columns = ['image:FILE', 'category']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    file_results["errors"].append(f"Missing columns: {missing_columns}")
                
                # Check for empty rows
                empty_rows = df.isnull().all(axis=1).sum()
                if empty_rows > 0:
                    file_results["warnings"].append(f"Found {empty_rows} empty rows")
                
                # Check for duplicate entries
                if 'image:FILE' in df.columns:
                    duplicates = df['image:FILE'].duplicated().sum()
                    if duplicates > 0:
                        file_results["warnings"].append(f"Found {duplicates} duplicate image paths")
                
                # Basic statistics
                file_results["stats"] = {
                    "total_rows": len(df),
                    "unique_images": df['image:FILE'].nunique() if 'image:FILE' in df.columns else 0,
                    "class_distribution": df['category'].value_counts().to_dict() if 'category' in df.columns else {}
                }
                
            except Exception as e:
                file_results["errors"].append(f"Error reading CSV: {str(e)}")
            
            results[csv_file] = file_results
        
        return results
    
    def validate_image_files(self, csv_file: str) -> Dict[str, any]:
        """
        Validate that all images referenced in CSV exist and are readable.
        
        Args:
            csv_file: Path to CSV file containing image references
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "total_images": 0,
            "valid_images": 0,
            "missing_images": [],
            "corrupted_images": [],
            "invalid_extensions": [],
            "image_stats": {}
        }
        
        try:
            df = pd.read_csv(self.root_dir / csv_file)
            if 'image:FILE' not in df.columns:
                results["error"] = "CSV missing 'image:FILE' column"
                return results
            
            results["total_images"] = len(df)
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            widths, heights, file_sizes = [], [], []
            
            for idx, image_path in enumerate(df['image:FILE']):
                full_path = self.root_dir / image_path
                
                # Check if file exists
                if not full_path.exists():
                    results["missing_images"].append(str(image_path))
                    continue
                
                # Check file extension
                if full_path.suffix.lower() not in allowed_extensions:
                    results["invalid_extensions"].append(str(image_path))
                    continue
                
                # Try to open and validate image
                try:
                    with Image.open(full_path) as img:
                        img.verify()  # Verify image integrity
                        
                    # Get image properties (reopen after verify)
                    with Image.open(full_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                        file_sizes.append(full_path.stat().st_size)
                    
                    results["valid_images"] += 1
                    
                except Exception as e:
                    results["corrupted_images"].append({
                        "path": str(image_path),
                        "error": str(e)
                    })
            
            # Calculate image statistics
            if widths:
                results["image_stats"] = {
                    "dimensions": {
                        "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths), "std": np.std(widths)},
                        "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights), "std": np.std(heights)}
                    },
                    "file_sizes": {
                        "min": format_bytes(min(file_sizes)),
                        "max": format_bytes(max(file_sizes)),
                        "mean": format_bytes(int(np.mean(file_sizes))),
                        "total": format_bytes(sum(file_sizes))
                    },
                    "aspect_ratios": {
                        "min": min(w/h for w, h in zip(widths, heights)),
                        "max": max(w/h for w, h in zip(widths, heights)),
                        "mean": np.mean([w/h for w, h in zip(widths, heights)])
                    }
                }
        
        except Exception as e:
            results["error"] = f"Error validating images: {str(e)}"
        
        return results
    
    def check_class_balance(self, csv_files: List[str]) -> Dict[str, any]:
        """
        Check class balance across datasets.
        
        Args:
            csv_files: List of CSV files to analyze
            
        Returns:
            Dictionary containing class balance analysis
        """
        results = {}
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(self.root_dir / csv_file)
                if 'category' not in df.columns:
                    results[csv_file] = {"error": "Missing 'category' column"}
                    continue
                
                class_counts = df['category'].value_counts().sort_index()
                total_samples = len(df)
                
                # Calculate class percentages
                class_percentages = (class_counts / total_samples * 100).round(2)
                
                # Check for severe imbalance (any class < 10% or > 60%)
                imbalance_warnings = []
                for class_id, percentage in class_percentages.items():
                    if percentage < 10:
                        imbalance_warnings.append(f"Class {class_id} is underrepresented ({percentage}%)")
                    elif percentage > 60:
                        imbalance_warnings.append(f"Class {class_id} is overrepresented ({percentage}%)")
                
                results[csv_file] = {
                    "class_counts": class_counts.to_dict(),
                    "class_percentages": class_percentages.to_dict(),
                    "total_samples": total_samples,
                    "num_classes": len(class_counts),
                    "imbalance_warnings": imbalance_warnings,
                    "balance_ratio": class_counts.min() / class_counts.max()  # Closer to 1 is better
                }
                
            except Exception as e:
                results[csv_file] = {"error": f"Error analyzing class balance: {str(e)}"}
        
        return results
    
    def validate_directory_structure(self) -> Dict[str, any]:
        """
        Validate the expected directory structure.
        
        Returns:
            Dictionary containing directory structure validation results
        """
        expected_dirs = ['train', 'val']
        expected_subdirs = ['healthy', 'angular_leaf_spot', 'bean_rust']
        expected_files = ['train.csv', 'val.csv', 'classname.txt']
        
        results = {
            "directories": {},
            "files": {},
            "structure_valid": True
        }
        
        # Check main directories
        for dir_name in expected_dirs:
            dir_path = self.root_dir / dir_name
            results["directories"][dir_name] = {
                "exists": dir_path.exists(),
                "is_directory": dir_path.is_dir() if dir_path.exists() else False,
                "subdirectories": {}
            }
            
            if dir_path.exists() and dir_path.is_dir():
                # Check subdirectories
                for subdir_name in expected_subdirs:
                    subdir_path = dir_path / subdir_name
                    results["directories"][dir_name]["subdirectories"][subdir_name] = {
                        "exists": subdir_path.exists(),
                        "is_directory": subdir_path.is_dir() if subdir_path.exists() else False,
                        "file_count": len(list(subdir_path.glob("*"))) if subdir_path.exists() and subdir_path.is_dir() else 0
                    }
                    
                    if not (subdir_path.exists() and subdir_path.is_dir()):
                        results["structure_valid"] = False
            else:
                results["structure_valid"] = False
        
        # Check expected files
        for file_name in expected_files:
            file_path = self.root_dir / file_name
            results["files"][file_name] = {
                "exists": file_path.exists(),
                "is_file": file_path.is_file() if file_path.exists() else False,
                "size": file_path.stat().st_size if file_path.exists() else 0
            }
            
            if not (file_path.exists() and file_path.is_file()):
                results["structure_valid"] = False
        
        return results
    
    def run_full_validation(self, save_report: bool = True) -> Dict[str, any]:
        """
        Run comprehensive validation of the entire dataset.
        
        Args:
            save_report: Whether to save validation report to file
            
        Returns:
            Complete validation results
        """
        training_logger.info("Starting comprehensive data validation...")
        
        validation_results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "directory_structure": self.validate_directory_structure(),
            "csv_validation": self.validate_csv_files(['train.csv', 'val.csv']),
            "image_validation": {
                "train": self.validate_image_files('train.csv'),
                "val": self.validate_image_files('val.csv')
            },
            "class_balance": self.check_class_balance(['train.csv', 'val.csv'])
        }
        
        # Generate summary
        summary = self._generate_validation_summary(validation_results)
        validation_results["summary"] = summary
        
        if save_report:
            report_path = self.root_dir / "data_validation_report.json"
            save_json(validation_results, str(report_path))
            training_logger.info(f"Validation report saved to: {report_path}")
        
        # Log summary
        training_logger.info("Data validation completed:")
        for key, value in summary.items():
            training_logger.info(f"  {key}: {value}")
        
        return validation_results
    
    def _generate_validation_summary(self, results: Dict[str, any]) -> Dict[str, any]:
        """Generate a summary of validation results."""
        summary = {
            "overall_status": "PASS",
            "total_errors": 0,
            "total_warnings": 0,
            "structure_valid": results["directory_structure"]["structure_valid"]
        }
        
        # Count errors and warnings from CSV validation
        for csv_file, csv_results in results["csv_validation"].items():
            summary["total_errors"] += len(csv_results.get("errors", []))
            summary["total_warnings"] += len(csv_results.get("warnings", []))
        
        # Count image validation issues
        for split, img_results in results["image_validation"].items():
            summary["total_errors"] += len(img_results.get("missing_images", []))
            summary["total_errors"] += len(img_results.get("corrupted_images", []))
            summary["total_warnings"] += len(img_results.get("invalid_extensions", []))
        
        # Check class balance warnings
        for csv_file, balance_results in results["class_balance"].items():
            if "imbalance_warnings" in balance_results:
                summary["total_warnings"] += len(balance_results["imbalance_warnings"])
        
        # Set overall status
        if summary["total_errors"] > 0 or not summary["structure_valid"]:
            summary["overall_status"] = "FAIL"
        elif summary["total_warnings"] > 0:
            summary["overall_status"] = "PASS_WITH_WARNINGS"
        
        return summary
    
    def create_data_visualization(self, save_plots: bool = True) -> None:
        """
        Create visualizations of the dataset.
        
        Args:
            save_plots: Whether to save plots to files
        """
        try:
            # Load data
            train_df = pd.read_csv(self.root_dir / 'train.csv')
            val_df = pd.read_csv(self.root_dir / 'val.csv')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Bean Lesion Dataset Analysis', fontsize=16)
            
            # Class distribution in training set
            train_counts = train_df['category'].value_counts().sort_index()
            class_names = ['Healthy', 'Angular Leaf Spot', 'Bean Rust']
            
            axes[0, 0].bar(range(len(train_counts)), train_counts.values, color=['green', 'orange', 'red'])
            axes[0, 0].set_title('Training Set Class Distribution')
            axes[0, 0].set_xlabel('Class')
            axes[0, 0].set_ylabel('Number of Samples')
            axes[0, 0].set_xticks(range(len(class_names)))
            axes[0, 0].set_xticklabels(class_names, rotation=45)
            
            # Class distribution in validation set
            val_counts = val_df['category'].value_counts().sort_index()
            
            axes[0, 1].bar(range(len(val_counts)), val_counts.values, color=['green', 'orange', 'red'])
            axes[0, 1].set_title('Validation Set Class Distribution')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Number of Samples')
            axes[0, 1].set_xticks(range(len(class_names)))
            axes[0, 1].set_xticklabels(class_names, rotation=45)
            
            # Combined class distribution
            combined_train = pd.Series(train_counts.values, index=class_names, name='Train')
            combined_val = pd.Series(val_counts.values, index=class_names, name='Validation')
            combined_df = pd.DataFrame([combined_train, combined_val]).T
            
            combined_df.plot(kind='bar', ax=axes[1, 0], color=['skyblue', 'lightcoral'])
            axes[1, 0].set_title('Train vs Validation Class Distribution')
            axes[1, 0].set_xlabel('Class')
            axes[1, 0].set_ylabel('Number of Samples')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Class balance ratios
            train_percentages = (train_counts / train_counts.sum() * 100).values
            val_percentages = (val_counts / val_counts.sum() * 100).values
            
            x = np.arange(len(class_names))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, train_percentages, width, label='Train', color='skyblue')
            axes[1, 1].bar(x + width/2, val_percentages, width, label='Validation', color='lightcoral')
            axes[1, 1].set_title('Class Distribution Percentages')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Percentage (%)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(class_names, rotation=45)
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.root_dir / "dataset_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                training_logger.info(f"Dataset visualization saved to: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            training_logger.error(f"Error creating data visualization: {e}")


def validate_dataset(root_dir: str = ".", save_report: bool = True, create_plots: bool = True) -> Dict[str, any]:
    """
    Convenience function to run complete dataset validation.
    
    Args:
        root_dir: Root directory containing the dataset
        save_report: Whether to save validation report
        create_plots: Whether to create visualization plots
        
    Returns:
        Validation results dictionary
    """
    validator = DataValidator(root_dir)
    results = validator.run_full_validation(save_report=save_report)
    
    if create_plots:
        validator.create_data_visualization(save_plots=True)
    
    return results