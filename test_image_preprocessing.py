"""
Test script for the enhanced image preprocessing functionality.
"""

import sys
import os
from pathlib import Path
import asyncio
import time

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PIL import Image
import numpy as np
import io

from src.api.image_processor import ImagePreprocessor, ImageValidationError


async def create_test_images():
    """Create various test images for validation."""
    test_images = {}
    
    # 1. Valid RGB image
    rgb_image = Image.new('RGB', (300, 300), color='green')
    rgb_bytes = io.BytesIO()
    rgb_image.save(rgb_bytes, format='JPEG')
    test_images['valid_rgb'] = rgb_bytes.getvalue()
    
    # 2. Small image (below minimum)
    small_image = Image.new('RGB', (20, 20), color='red')
    small_bytes = io.BytesIO()
    small_image.save(small_bytes, format='PNG')
    test_images['too_small'] = small_bytes.getvalue()
    
    # 3. Large image
    large_image = Image.new('RGB', (2000, 2000), color='blue')
    large_bytes = io.BytesIO()
    large_image.save(large_bytes, format='PNG')
    test_images['large_image'] = large_bytes.getvalue()
    
    # 4. RGBA image (needs conversion)
    rgba_image = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
    rgba_bytes = io.BytesIO()
    rgba_image.save(rgba_bytes, format='PNG')
    test_images['rgba_image'] = rgba_bytes.getvalue()
    
    # 5. WebP image
    webp_image = Image.new('RGB', (400, 300), color='purple')
    webp_bytes = io.BytesIO()
    webp_image.save(webp_bytes, format='WEBP')
    test_images['webp_image'] = webp_bytes.getvalue()
    
    return test_images


async def test_single_image_preprocessing():
    """Test single image preprocessing."""
    print("🧪 Testing Single Image Preprocessing")
    print("=" * 50)
    
    preprocessor = ImagePreprocessor()
    test_images = await create_test_images()
    
    for name, image_data in test_images.items():
        print(f"\nTesting: {name}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            # Test with validation info
            result, validation_info = await preprocessor.preprocess_single_image(
                image_data,
                filename=f"{name}.jpg",
                content_type="image/jpeg",
                return_validation_info=True
            )
            
            processing_time = time.time() - start_time
            
            print(f"✅ Success!")
            print(f"   Output shape: {result.shape}")
            print(f"   File size: {validation_info['file_size']} bytes")
            print(f"   Detected format: {validation_info['detected_format']}")
            print(f"   Dimensions: {validation_info['width']}x{validation_info['height']}")
            print(f"   Quality score: {validation_info.get('quality_score', 'N/A')}")
            print(f"   Processing time: {processing_time:.3f}s")
            
            if 'quality_issues' in validation_info and validation_info['quality_issues']:
                print(f"   Quality issues: {validation_info['quality_issues']}")
            
        except ImageValidationError as e:
            print(f"❌ Validation Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")


async def test_batch_preprocessing():
    """Test batch image preprocessing."""
    print("\n\n🧪 Testing Batch Image Preprocessing")
    print("=" * 50)
    
    preprocessor = ImagePreprocessor()
    test_images = await create_test_images()
    
    # Prepare batch data
    batch_data = [
        (image_data, f"{name}.jpg", "image/jpeg")
        for name, image_data in test_images.items()
    ]
    
    try:
        start_time = time.time()
        
        # Test batch processing with validation info
        results, validation_infos = await preprocessor.preprocess_batch_images(
            batch_data,
            return_validation_info=True,
            fail_on_error=False  # Skip invalid images
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Batch processing completed!")
        print(f"   Total images: {len(batch_data)}")
        print(f"   Successful: {len(results)}")
        print(f"   Failed: {len(batch_data) - len(results)}")
        print(f"   Total processing time: {processing_time:.3f}s")
        print(f"   Average per image: {processing_time/len(batch_data):.3f}s")
        
        # Show details for successful images
        for i, (result, validation_info) in enumerate(zip(results, validation_infos)):
            print(f"\n   Image {i+1}:")
            print(f"     Shape: {result.shape}")
            print(f"     Quality score: {validation_info.get('quality_score', 'N/A')}")
            
            if 'batch_summary' in validation_info:
                batch_summary = validation_info['batch_summary']
                if batch_summary['failed_images'] > 0:
                    print(f"     Failed images in batch: {batch_summary['failed_images']}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")


async def test_validation_edge_cases():
    """Test validation with edge cases."""
    print("\n\n🧪 Testing Validation Edge Cases")
    print("=" * 50)
    
    preprocessor = ImagePreprocessor()
    
    # Test cases
    test_cases = [
        ("Empty data", b""),
        ("Invalid JPEG header", b"\xff\xd8\xff\xe0invalid"),
        ("Text file", b"This is not an image file"),
        ("Partial PNG", b"\x89PNG\r\n\x1a\n" + b"incomplete"),
    ]
    
    for name, data in test_cases:
        print(f"\nTesting: {name}")
        print("-" * 30)
        
        try:
            await preprocessor.validate_file_data(data, f"{name}.jpg", "image/jpeg")
            print("❌ Should have failed but didn't!")
        except ImageValidationError as e:
            print(f"✅ Correctly caught error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


async def test_preprocessing_stats():
    """Test preprocessing statistics and configuration."""
    print("\n\n🧪 Testing Preprocessing Statistics")
    print("=" * 50)
    
    preprocessor = ImagePreprocessor()
    stats = preprocessor.get_preprocessing_stats()
    
    print("Configuration:")
    for key, value in stats['configuration'].items():
        print(f"  {key}: {value}")
    
    print("\nNormalization:")
    for key, value in stats['normalization'].items():
        print(f"  {key}: {value}")


async def main():
    """Run all preprocessing tests."""
    print("🚀 Bean Lesion Classification - Image Preprocessing Tests")
    print("=" * 60)
    
    try:
        await test_single_image_preprocessing()
        await test_batch_preprocessing()
        await test_validation_edge_cases()
        await test_preprocessing_stats()
        
        print("\n\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)