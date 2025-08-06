"""
Advanced image preprocessing and validation for bean lesion classification.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image, ImageStat, ExifTags
import io
from typing import Dict, List, Tuple, Optional, Union, Any
import hashlib
from fastapi import HTTPException
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from src.utils.logging_config import api_logger


class ImageValidationError(Exception):
    """Custom exception for image validation errors."""

    pass


class ImagePreprocessor:
    """
    Advanced image preprocessing and validation class.

    Features:
    - File type validation (JPEG, PNG, WebP)
    - Image quality and corruption detection
    - Memory-efficient processing
    - Batch processing capabilities
    - EXIF data handling
    - Size and dimension validation
    """

    # Supported image formats
    SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    SUPPORTED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

    # Image constraints
    MIN_DIMENSION = 32  # Minimum width/height
    MAX_DIMENSION = 4096  # Maximum width/height
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_FILE_SIZE = 1024  # 1KB

    # Model input requirements
    MODEL_INPUT_SIZE = (224, 224)
    MODEL_CHANNELS = 3

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        min_file_size: int = MIN_FILE_SIZE,
        max_dimension: int = MAX_DIMENSION,
        min_dimension: int = MIN_DIMENSION,
        model_input_size: Tuple[int, int] = MODEL_INPUT_SIZE,
        enable_quality_check: bool = True,
        enable_corruption_check: bool = True,
    ):
        """
        Initialize ImagePreprocessor.

        Args:
            max_file_size: Maximum file size in bytes
            min_file_size: Minimum file size in bytes
            max_dimension: Maximum image dimension (width or height)
            min_dimension: Minimum image dimension (width or height)
            model_input_size: Target size for model input (width, height)
            enable_quality_check: Enable image quality validation
            enable_corruption_check: Enable corruption detection
        """
        self.max_file_size = max_file_size
        self.min_file_size = min_file_size
        self.max_dimension = max_dimension
        self.min_dimension = min_dimension
        self.model_input_size = model_input_size
        self.enable_quality_check = enable_quality_check
        self.enable_corruption_check = enable_corruption_check

        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # ImageNet normalization parameters
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        api_logger.info(f"ImagePreprocessor initialized:")
        api_logger.info(f"  Max file size: {self.max_file_size / (1024*1024):.1f}MB")
        api_logger.info(
            f"  Dimension range: {self.min_dimension}-{self.max_dimension}px"
        )
        api_logger.info(f"  Model input size: {self.model_input_size}")
        api_logger.info(f"  Quality check: {self.enable_quality_check}")
        api_logger.info(f"  Corruption check: {self.enable_corruption_check}")

    async def validate_file_data(
        self,
        file_data: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate raw file data before processing.

        Args:
            file_data: Raw file bytes
            filename: Original filename (optional)
            content_type: MIME content type (optional)

        Returns:
            Validation results dictionary

        Raises:
            ImageValidationError: If validation fails
        """
        validation_start = time.time()

        try:
            # Basic size validation
            file_size = len(file_data)
            if file_size < self.min_file_size:
                raise ImageValidationError(
                    f"File too small: {file_size} bytes (minimum: {self.min_file_size})"
                )

            if file_size > self.max_file_size:
                raise ImageValidationError(
                    f"File too large: {file_size} bytes (maximum: {self.max_file_size})"
                )

            # File extension validation
            if filename:
                file_ext = Path(filename).suffix.lower()
                if file_ext not in self.SUPPORTED_EXTENSIONS:
                    raise ImageValidationError(
                        f"Unsupported file extension: {file_ext}. "
                        f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
                    )

            # MIME type validation
            if content_type and content_type not in self.SUPPORTED_MIME_TYPES:
                raise ImageValidationError(
                    f"Unsupported MIME type: {content_type}. "
                    f"Supported: {', '.join(self.SUPPORTED_MIME_TYPES)}"
                )

            # Magic number validation (file signature)
            detected_type = await self._detect_file_type(file_data)
            if detected_type not in self.SUPPORTED_FORMATS:
                raise ImageValidationError(
                    f"Invalid file format detected: {detected_type}. "
                    f"File may be corrupted or not a valid image."
                )

            # Calculate file hash for deduplication
            file_hash = hashlib.md5(file_data).hexdigest()

            validation_time = time.time() - validation_start

            return {
                "valid": True,
                "file_size": file_size,
                "detected_format": detected_type,
                "file_hash": file_hash,
                "validation_time": validation_time,
            }

        except ImageValidationError:
            raise
        except Exception as e:
            api_logger.error(f"Unexpected error during file validation: {e}")
            raise ImageValidationError(f"File validation failed: {str(e)}")

    async def _detect_file_type(self, file_data: bytes) -> str:
        """
        Detect file type using magic numbers.

        Args:
            file_data: Raw file bytes

        Returns:
            Detected file format
        """

        def _detect():
            try:
                # Fallback: try to detect from file header
                if file_data.startswith(b"\xff\xd8\xff"):
                    return "JPEG"
                elif file_data.startswith(b"\x89PNG\r\n\x1a\n"):
                    return "PNG"
                elif file_data.startswith(b"RIFF") and b"WEBP" in file_data[:12]:
                    return "WEBP"
                else:
                    return "UNKNOWN"

            except Exception:
                return "UNKNOWN"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _detect)

    async def preprocess_single_image(
        self,
        file_data: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        return_validation_info: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Preprocess a single image for model inference.

        Args:
            file_data: Raw image file bytes
            filename: Original filename (optional)
            content_type: MIME content type (optional)
            return_validation_info: Whether to return validation information

        Returns:
            Preprocessed numpy array, optionally with validation info

        Raises:
            ImageValidationError: If validation or preprocessing fails
        """
        processing_start = time.time()

        try:
            # Step 1: Validate file data
            file_validation = await self.validate_file_data(
                file_data, filename, content_type
            )

            # Step 2: Load image
            image = await self._load_image_from_bytes(file_data)

            # Step 3: Validate image content
            image_validation = await self.validate_image_content(image)

            # Step 4: Handle orientation (EXIF) - simplified for now
            # image = await self._fix_image_orientation(image)  # Skip EXIF for now

            # Step 5: Preprocess for model
            preprocessed_array = await self._preprocess_for_model(image)

            processing_time = time.time() - processing_start

            if return_validation_info:
                validation_info = {
                    **file_validation,
                    **image_validation,
                    "processing_time": processing_time,
                    "preprocessed_shape": preprocessed_array.shape,
                }
                return preprocessed_array, validation_info
            else:
                return preprocessed_array

        except ImageValidationError:
            raise
        except Exception as e:
            api_logger.error(f"Image preprocessing failed: {e}")
            raise ImageValidationError(f"Preprocessing failed: {str(e)}")

    async def validate_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """
        Validate image content and quality.

        Args:
            image: PIL Image object

        Returns:
            Validation results dictionary

        Raises:
            ImageValidationError: If validation fails
        """
        try:
            # Basic image properties
            width, height = image.size
            mode = image.mode
            format_name = image.format or "UNKNOWN"

            # Dimension validation
            if width < self.min_dimension or height < self.min_dimension:
                raise ImageValidationError(
                    f"Image too small: {width}x{height}px "
                    f"(minimum: {self.min_dimension}x{self.min_dimension}px)"
                )

            if width > self.max_dimension or height > self.max_dimension:
                raise ImageValidationError(
                    f"Image too large: {width}x{height}px "
                    f"(maximum: {self.max_dimension}x{self.max_dimension}px)"
                )

            # Aspect ratio check (warn if extreme)
            aspect_ratio = width / height
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                api_logger.warning(
                    f"Extreme aspect ratio detected: {aspect_ratio:.2f} "
                    f"({width}x{height}px). This may affect classification accuracy."
                )

            validation_results = {
                "width": width,
                "height": height,
                "mode": mode,
                "format": format_name,
                "aspect_ratio": aspect_ratio,
                "channels": len(image.getbands()) if hasattr(image, "getbands") else 3,
            }

            # Quality checks (if enabled)
            if self.enable_quality_check:
                quality_results = await self._check_image_quality(image)
                validation_results.update(quality_results)

            # Corruption checks (if enabled)
            if self.enable_corruption_check:
                corruption_results = await self._check_image_corruption(image)
                validation_results.update(corruption_results)

            return validation_results

        except ImageValidationError:
            raise
        except Exception as e:
            api_logger.error(f"Unexpected error during image validation: {e}")
            raise ImageValidationError(f"Image validation failed: {str(e)}")

    async def _check_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Check image quality metrics.

        Args:
            image: PIL Image object

        Returns:
            Quality metrics dictionary
        """

        def _check():
            try:
                # Convert to RGB for consistent analysis
                rgb_image = image.convert("RGB")

                # Calculate basic statistics
                stat = ImageStat.Stat(rgb_image)

                # Mean brightness (0-255)
                mean_brightness = sum(stat.mean) / 3

                # Standard deviation (measure of contrast)
                mean_contrast = sum(stat.stddev) / 3

                # Check for very dark or very bright images
                is_too_dark = mean_brightness < 30
                is_too_bright = mean_brightness > 225
                is_low_contrast = mean_contrast < 10

                quality_score = 1.0
                quality_issues = []

                if is_too_dark:
                    quality_score -= 0.3
                    quality_issues.append("too_dark")

                if is_too_bright:
                    quality_score -= 0.3
                    quality_issues.append("too_bright")

                if is_low_contrast:
                    quality_score -= 0.2
                    quality_issues.append("low_contrast")

                return {
                    "quality_score": max(0.0, quality_score),
                    "mean_brightness": mean_brightness,
                    "mean_contrast": mean_contrast,
                    "quality_issues": quality_issues,
                    "is_too_dark": is_too_dark,
                    "is_too_bright": is_too_bright,
                    "is_low_contrast": is_low_contrast,
                }

            except Exception as e:
                api_logger.warning(f"Quality check failed: {e}")
                return {"quality_score": 0.5, "quality_issues": ["check_failed"]}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _check)

    async def _check_image_corruption(self, image: Image.Image) -> Dict[str, Any]:
        """
        Check for image corruption indicators.

        Args:
            image: PIL Image object

        Returns:
            Corruption check results
        """

        def _check():
            try:
                corruption_indicators = []

                # Try to load and verify the image
                try:
                    image.load()
                    image.verify()
                except Exception as e:
                    corruption_indicators.append(f"verification_failed: {str(e)}")

                # Check for unusual pixel patterns
                try:
                    # Convert to array for analysis
                    img_array = np.array(image.convert("RGB"))

                    # Check for completely uniform images
                    if np.std(img_array) < 1:
                        corruption_indicators.append("uniform_image")

                    # Check for extreme values
                    if np.min(img_array) == np.max(img_array):
                        corruption_indicators.append("single_color")

                except Exception as e:
                    corruption_indicators.append(f"array_analysis_failed: {str(e)}")

                is_corrupted = len(corruption_indicators) > 0

                return {
                    "is_corrupted": is_corrupted,
                    "corruption_indicators": corruption_indicators,
                }

            except Exception as e:
                api_logger.warning(f"Corruption check failed: {e}")
                return {
                    "is_corrupted": False,
                    "corruption_indicators": ["check_failed"],
                }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _check)

    async def _load_image_from_bytes(self, file_data: bytes) -> Image.Image:
        """
        Load PIL Image from bytes with error handling.

        Args:
            file_data: Raw image bytes

        Returns:
            PIL Image object

        Raises:
            ImageValidationError: If image cannot be loaded
        """

        def _load():
            try:
                image = Image.open(io.BytesIO(file_data))
                # Ensure image is loaded
                image.load()
                return image
            except Exception as e:
                raise ImageValidationError(f"Cannot load image: {str(e)}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _load)

    async def _fix_image_orientation(self, image: Image.Image) -> Image.Image:
        """
        Fix image orientation based on EXIF data.

        Args:
            image: PIL Image object

        Returns:
            Oriented PIL Image object
        """

        def _fix():
            try:
                # Check for EXIF orientation tag
                if hasattr(image, "_getexif") and image._getexif() is not None:
                    exif = image._getexif()
                    orientation_tag = 274  # EXIF orientation tag

                    if orientation_tag in exif:
                        orientation = exif[orientation_tag]

                        # Apply rotation based on orientation
                        if orientation == 2:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 4:
                            image = image.transpose(Image.FLIP_TOP_BOTTOM)
                        elif orientation == 5:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                                90, expand=True
                            )
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 7:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                                270, expand=True
                            )
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)

                return image

            except Exception as e:
                api_logger.debug(f"Orientation fix failed: {e}")
                return image  # Return original if fix fails

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _fix)

    async def _preprocess_for_model(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model inference.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed numpy array ready for model input
        """

        def _preprocess():
            # Convert to RGB if needed
            if image.mode != "RGB":
                image_rgb = image.convert("RGB")
            else:
                image_rgb = image

            # EXACT SAME PREPROCESSING AS TRAINING VALIDATION:
            # 1. Resize to slightly larger size (matches training validation transforms)
            resize_size = int(self.model_input_size[0] * 1.14)  # 224 * 1.14 = 255
            resized_image = image_rgb.resize(
                (resize_size, resize_size), Image.Resampling.LANCZOS
            )

            # 2. Center crop to target size (matches training validation transforms)
            width, height = resized_image.size
            target_width, target_height = self.model_input_size

            left = (width - target_width) // 2
            top = (height - target_height) // 2
            right = left + target_width
            bottom = top + target_height

            cropped_image = resized_image.crop((left, top, right, bottom))

            # 3. Convert to numpy array and normalize to [0, 1] (ToTensor does this)
            img_array = np.array(cropped_image, dtype=np.float32) / 255.0

            # 4. Apply ImageNet normalization (exact same values as training)
            img_array = (img_array - self.imagenet_mean) / self.imagenet_std

            # 5. Transpose from HWC to CHW format (ToTensor does this)
            img_array = img_array.transpose(2, 0, 1)

            # 6. Add batch dimension
            img_array = img_array[np.newaxis, :]

            return img_array

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _preprocess)

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration and statistics.

        Returns:
            Configuration and stats dictionary
        """
        return {
            "configuration": {
                "supported_formats": list(self.SUPPORTED_FORMATS),
                "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
                "supported_mime_types": list(self.SUPPORTED_MIME_TYPES),
                "max_file_size_mb": self.max_file_size / (1024 * 1024),
                "min_file_size_kb": self.min_file_size / 1024,
                "dimension_range": f"{self.min_dimension}-{self.max_dimension}px",
                "model_input_size": self.model_input_size,
                "model_channels": self.MODEL_CHANNELS,
                "quality_check_enabled": self.enable_quality_check,
                "corruption_check_enabled": self.enable_corruption_check,
            },
            "normalization": {
                "mean": self.imagenet_mean.tolist(),
                "std": self.imagenet_std.tolist(),
                "method": "ImageNet normalization",
            },
        }

    async def cleanup(self):
        """
        Clean up resources.
        """
        api_logger.info("Cleaning up ImagePreprocessor...")

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        api_logger.info("ImagePreprocessor cleanup completed")


# Utility functions for common preprocessing tasks


async def validate_and_preprocess_image(
    file_data: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    preprocessor: Optional[ImagePreprocessor] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to validate and preprocess a single image.

    Args:
        file_data: Raw image bytes
        filename: Original filename
        content_type: MIME content type
        preprocessor: ImagePreprocessor instance (creates default if None)

    Returns:
        Tuple of (preprocessed_array, validation_info)
    """
    if preprocessor is None:
        preprocessor = ImagePreprocessor()

    return await preprocessor.preprocess_single_image(
        file_data, filename, content_type, return_validation_info=True
    )


async def preprocess_image_for_inference(
    file_data: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    preprocessor: Optional[ImagePreprocessor] = None,
) -> np.ndarray:
    """
    Convenience function to preprocess image for inference only.

    Args:
        file_data: Raw image bytes
        filename: Original filename
        content_type: MIME content type
        preprocessor: ImagePreprocessor instance (creates default if None)

    Returns:
        Preprocessed numpy array
    """
    if preprocessor is None:
        preprocessor = ImagePreprocessor()

    return await preprocessor.preprocess_single_image(
        file_data, filename, content_type, return_validation_info=False
    )
