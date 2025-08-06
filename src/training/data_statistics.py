"""
Data statistics and analysis utilities for bean lesion classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json

from ..utils.logging_config import training_logger
from ..utils.helpers import format_bytes, save_json, Timer


class DatasetStatistics:
    """
    Comprehensive statistics calculator for the bean lesion dataset.
    """
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.class_names = ["healthy", "angular_leaf_spot", "bean_rust"]
        
    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic dataset statistics."""
        stats = {}
        
        for split in ['train', 'val']:
            csv_path = self.root_dir / f"{split}.csv"
            if not csv_path.exists():
                training_logger.warning(f"CSV file not found: {csv_path}")
                continue
                
            df = pd.read_csv(csv_path)
            
            # Basic counts
            total_samples = len(df)
            class_distribution = df['category'].value_counts().sort_index()
            
            # Class percentages
            class_percentages = (class_distribution / total_samples * 100).round(2)
            
            # Class balance metrics
            balance_ratio = class_distribution.min() / class_distribution.max()
            entropy = self._calculate_entropy(class_distribution.values)
            
            stats[split] = {
                "total_samples": total_samples,
                "num_classes": len(class_distribution),
                "class_distribution": {
                    self.class_names[i]: int(count) 
                    for i, count in class_distribution.items()
                },
                "class_percentages": {
                    self.class_names[i]: float(pct) 
                    for i, pct in class_percentages.items()
                },
                "balance_metrics": {
                    "balance_ratio": float(balance_ratio),
                    "entropy": float(entropy),
                    "gini_impurity": float(self._calculate_gini_impurity(class_distribution.values))
                }
            }
        
        return stats
    
    def calculate_image_statistics(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate detailed image statistics.
        
        Args:
            sample_size: Number of images to sample for analysis (None for all)
        """
        stats = {}
        
        for split in ['train', 'val']:
            csv_path = self.root_dir / f"{split}.csv"
            if not csv_path.exists():
                continue
                
            df = pd.read_csv(csv_path)
            
            # Sample images if requested
            if sample_size and len(df) > sample_size:
                df_sample = df.sample(n=sample_size, random_state=42)
                training_logger.info(f"Sampling {sample_size} images from {split} set for analysis")
            else:
                df_sample = df
            
            # Initialize collections
            dimensions = []
            file_sizes = []
            aspect_ratios = []
            color_stats = defaultdict(list)
            
            training_logger.info(f"Analyzing {len(df_sample)} images from {split} set...")
            
            with Timer(f"Image analysis for {split} set"):
                for idx, row in df_sample.iterrows():
                    image_path = self.root_dir / row['image:FILE']
                    
                    try:
                        # Get file size
                        file_size = image_path.stat().st_size
                        file_sizes.append(file_size)
                        
                        # Open image and get properties
                        with Image.open(image_path) as img:
                            width, height = img.size
                            dimensions.append((width, height))
                            aspect_ratios.append(width / height)
                            
                            # Convert to RGB if needed and get color statistics
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Sample pixels for color analysis (to avoid memory issues)
                            img_array = np.array(img)
                            if img_array.size > 224 * 224 * 3:  # Resize large images for analysis
                                img_resized = img.resize((224, 224))
                                img_array = np.array(img_resized)
                            
                            # Calculate color statistics
                            for channel, name in enumerate(['red', 'green', 'blue']):
                                channel_data = img_array[:, :, channel].flatten()
                                color_stats[name].extend([
                                    float(np.mean(channel_data)),
                                    float(np.std(channel_data))
                                ])
                    
                    except Exception as e:
                        training_logger.warning(f"Error analyzing image {image_path}: {e}")
                        continue
            
            # Calculate statistics
            if dimensions:
                widths, heights = zip(*dimensions)
                
                stats[split] = {
                    "sample_size": len(df_sample),
                    "total_size": len(df),
                    "dimensions": {
                        "width": {
                            "min": int(min(widths)),
                            "max": int(max(widths)),
                            "mean": float(np.mean(widths)),
                            "std": float(np.std(widths)),
                            "median": float(np.median(widths))
                        },
                        "height": {
                            "min": int(min(heights)),
                            "max": int(max(heights)),
                            "mean": float(np.mean(heights)),
                            "std": float(np.std(heights)),
                            "median": float(np.median(heights))
                        }
                    },
                    "aspect_ratios": {
                        "min": float(min(aspect_ratios)),
                        "max": float(max(aspect_ratios)),
                        "mean": float(np.mean(aspect_ratios)),
                        "std": float(np.std(aspect_ratios)),
                        "median": float(np.median(aspect_ratios))
                    },
                    "file_sizes": {
                        "min_bytes": int(min(file_sizes)),
                        "max_bytes": int(max(file_sizes)),
                        "mean_bytes": int(np.mean(file_sizes)),
                        "median_bytes": int(np.median(file_sizes)),
                        "total_bytes": int(sum(file_sizes)),
                        "min_formatted": format_bytes(min(file_sizes)),
                        "max_formatted": format_bytes(max(file_sizes)),
                        "mean_formatted": format_bytes(int(np.mean(file_sizes))),
                        "total_formatted": format_bytes(sum(file_sizes))
                    }
                }
                
                # Add color statistics if available
                if color_stats:
                    stats[split]["color_statistics"] = {}
                    for color in ['red', 'green', 'blue']:
                        if color in color_stats:
                            means = color_stats[color][::2]  # Every other value is mean
                            stds = color_stats[color][1::2]   # Every other value is std
                            stats[split]["color_statistics"][color] = {
                                "mean_across_images": float(np.mean(means)),
                                "std_across_images": float(np.mean(stds)),
                                "mean_variation": float(np.std(means))
                            }
        
        return stats
    
    def analyze_class_specific_statistics(self) -> Dict[str, Any]:
        """Analyze statistics for each class separately."""
        class_stats = {}
        
        for split in ['train', 'val']:
            csv_path = self.root_dir / f"{split}.csv"
            if not csv_path.exists():
                continue
                
            df = pd.read_csv(csv_path)
            class_stats[split] = {}
            
            for class_id in df['category'].unique():
                class_df = df[df['category'] == class_id]
                class_name = self.class_names[class_id]
                
                # Sample images for analysis
                sample_size = min(50, len(class_df))  # Analyze up to 50 images per class
                class_sample = class_df.sample(n=sample_size, random_state=42)
                
                dimensions = []
                file_sizes = []
                
                for _, row in class_sample.iterrows():
                    image_path = self.root_dir / row['image:FILE']
                    
                    try:
                        file_size = image_path.stat().st_size
                        file_sizes.append(file_size)
                        
                        with Image.open(image_path) as img:
                            width, height = img.size
                            dimensions.append((width, height))
                    
                    except Exception as e:
                        training_logger.warning(f"Error analyzing {class_name} image {image_path}: {e}")
                        continue
                
                if dimensions:
                    widths, heights = zip(*dimensions)
                    
                    class_stats[split][class_name] = {
                        "sample_count": len(class_df),
                        "analyzed_count": len(dimensions),
                        "avg_width": float(np.mean(widths)),
                        "avg_height": float(np.mean(heights)),
                        "avg_file_size": format_bytes(int(np.mean(file_sizes))),
                        "total_size": format_bytes(sum(file_sizes) * len(class_df) // len(dimensions))
                    }
        
        return class_stats
    
    def generate_comprehensive_report(self, save_report: bool = True) -> Dict[str, Any]:
        """Generate a comprehensive statistics report."""
        training_logger.info("Generating comprehensive dataset statistics report...")
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "root_directory": str(self.root_dir),
                "class_names": self.class_names,
                "splits": ["train", "val"]
            },
            "basic_statistics": self.calculate_basic_statistics(),
            "image_statistics": self.calculate_image_statistics(sample_size=200),
            "class_specific_statistics": self.analyze_class_specific_statistics()
        }
        
        # Add summary
        report["summary"] = self._generate_summary(report)
        
        if save_report:
            report_path = self.root_dir / "dataset_statistics_report.json"
            save_json(report, str(report_path))
            training_logger.info(f"Statistics report saved to: {report_path}")
        
        return report
    
    def create_statistical_visualizations(self, save_plots: bool = True) -> None:
        """Create comprehensive statistical visualizations."""
        try:
            # Load basic statistics
            basic_stats = self.calculate_basic_statistics()
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Class distribution comparison
            ax1 = plt.subplot(3, 3, 1)
            train_counts = [basic_stats['train']['class_distribution'][cls] for cls in self.class_names]
            val_counts = [basic_stats['val']['class_distribution'][cls] for cls in self.class_names]
            
            x = np.arange(len(self.class_names))
            width = 0.35
            
            ax1.bar(x - width/2, train_counts, width, label='Train', alpha=0.8)
            ax1.bar(x + width/2, val_counts, width, label='Validation', alpha=0.8)
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Class Distribution Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels([cls.replace('_', ' ').title() for cls in self.class_names], rotation=45)
            ax1.legend()
            
            # 2. Class percentages
            ax2 = plt.subplot(3, 3, 2)
            train_pcts = [basic_stats['train']['class_percentages'][cls] for cls in self.class_names]
            val_pcts = [basic_stats['val']['class_percentages'][cls] for cls in self.class_names]
            
            ax2.bar(x - width/2, train_pcts, width, label='Train', alpha=0.8)
            ax2.bar(x + width/2, val_pcts, width, label='Validation', alpha=0.8)
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Class Distribution Percentages')
            ax2.set_xticks(x)
            ax2.set_xticklabels([cls.replace('_', ' ').title() for cls in self.class_names], rotation=45)
            ax2.legend()
            
            # 3. Balance metrics
            ax3 = plt.subplot(3, 3, 3)
            metrics = ['balance_ratio', 'entropy', 'gini_impurity']
            train_metrics = [basic_stats['train']['balance_metrics'][m] for m in metrics]
            val_metrics = [basic_stats['val']['balance_metrics'][m] for m in metrics]
            
            x_metrics = np.arange(len(metrics))
            ax3.bar(x_metrics - width/2, train_metrics, width, label='Train', alpha=0.8)
            ax3.bar(x_metrics + width/2, val_metrics, width, label='Validation', alpha=0.8)
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Value')
            ax3.set_title('Class Balance Metrics')
            ax3.set_xticks(x_metrics)
            ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
            ax3.legend()
            
            # Load image statistics for remaining plots
            img_stats = self.calculate_image_statistics(sample_size=100)
            
            # 4. Image dimensions distribution
            ax4 = plt.subplot(3, 3, 4)
            if 'train' in img_stats:
                train_widths = [img_stats['train']['dimensions']['width']['mean']]
                train_heights = [img_stats['train']['dimensions']['height']['mean']]
                val_widths = [img_stats['val']['dimensions']['width']['mean']]
                val_heights = [img_stats['val']['dimensions']['height']['mean']]
                
                ax4.scatter(train_widths, train_heights, label='Train Avg', s=100, alpha=0.7)
                ax4.scatter(val_widths, val_heights, label='Val Avg', s=100, alpha=0.7)
                ax4.set_xlabel('Width (pixels)')
                ax4.set_ylabel('Height (pixels)')
                ax4.set_title('Average Image Dimensions')
                ax4.legend()
            
            # 5. File size distribution
            ax5 = plt.subplot(3, 3, 5)
            if 'train' in img_stats:
                splits = ['train', 'val']
                avg_sizes = [img_stats[split]['file_sizes']['mean_bytes'] / 1024 for split in splits]  # Convert to KB
                
                ax5.bar(splits, avg_sizes, alpha=0.8, color=['skyblue', 'lightcoral'])
                ax5.set_ylabel('Average File Size (KB)')
                ax5.set_title('Average File Sizes by Split')
            
            # 6. Aspect ratio distribution
            ax6 = plt.subplot(3, 3, 6)
            if 'train' in img_stats:
                splits = ['train', 'val']
                aspect_ratios = [img_stats[split]['aspect_ratios']['mean'] for split in splits]
                
                ax6.bar(splits, aspect_ratios, alpha=0.8, color=['lightgreen', 'orange'])
                ax6.set_ylabel('Average Aspect Ratio')
                ax6.set_title('Average Aspect Ratios by Split')
                ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Square (1:1)')
                ax6.legend()
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.root_dir / "dataset_statistics_visualization.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                training_logger.info(f"Statistical visualization saved to: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            training_logger.error(f"Error creating statistical visualizations: {e}")
    
    def _calculate_entropy(self, class_counts: np.ndarray) -> float:
        """Calculate entropy of class distribution."""
        probabilities = class_counts / class_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_gini_impurity(self, class_counts: np.ndarray) -> float:
        """Calculate Gini impurity of class distribution."""
        probabilities = class_counts / class_counts.sum()
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the statistics report."""
        summary = {}
        
        # Dataset size summary
        train_total = report['basic_statistics']['train']['total_samples']
        val_total = report['basic_statistics']['val']['total_samples']
        
        summary['dataset_size'] = {
            'total_samples': train_total + val_total,
            'train_samples': train_total,
            'val_samples': val_total,
            'train_val_ratio': round(train_total / val_total, 2)
        }
        
        # Class balance summary
        train_balance = report['basic_statistics']['train']['balance_metrics']['balance_ratio']
        val_balance = report['basic_statistics']['val']['balance_metrics']['balance_ratio']
        
        summary['class_balance'] = {
            'train_balance_ratio': round(train_balance, 3),
            'val_balance_ratio': round(val_balance, 3),
            'balance_status': 'Good' if min(train_balance, val_balance) > 0.5 else 'Imbalanced'
        }
        
        # Image characteristics summary
        if 'train' in report['image_statistics']:
            train_img = report['image_statistics']['train']
            summary['image_characteristics'] = {
                'avg_width': int(train_img['dimensions']['width']['mean']),
                'avg_height': int(train_img['dimensions']['height']['mean']),
                'avg_file_size': train_img['file_sizes']['mean_formatted'],
                'total_dataset_size': train_img['file_sizes']['total_formatted']
            }
        
        return summary


def generate_dataset_statistics(root_dir: str = ".", save_report: bool = True, create_plots: bool = True) -> Dict[str, Any]:
    """
    Convenience function to generate comprehensive dataset statistics.
    
    Args:
        root_dir: Root directory containing the dataset
        save_report: Whether to save the statistics report
        create_plots: Whether to create visualization plots
        
    Returns:
        Complete statistics report
    """
    stats_calculator = DatasetStatistics(root_dir)
    report = stats_calculator.generate_comprehensive_report(save_report=save_report)
    
    if create_plots:
        stats_calculator.create_statistical_visualizations(save_plots=True)
    
    return report