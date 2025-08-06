"""
High-performance ONNX inference engine for bean lesion classification.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from collections import OrderedDict

from src.utils.logging_config import api_logger
from src.utils.helpers import get_device


class InferenceProvider(str, Enum):
    """Available ONNX Runtime providers."""

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorRTExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    COREML = "CoreMLExecutionProvider"


@dataclass
class InferenceResult:
    """Single inference result with metadata."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    inference_time: float = 0.0
    session_id: Optional[str] = None
    provider: Optional[str] = None
    batch_index: Optional[int] = None


@dataclass
class BatchInferenceResult:
    """Batch inference result with metadata."""

    results: List[InferenceResult]
    batch_size: int
    total_inference_time: float
    average_inference_time: float
    session_id: str
    provider: str


class SessionPool:
    """Thread-safe ONNX Runtime session pool for efficient resource management."""

    def __init__(
        self,
        model_path: str,
        providers: List[str],
        max_sessions: int = 4,
        session_options: Optional[ort.SessionOptions] = None,
    ):
        """
        Initialize session pool.

        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers
            max_sessions: Maximum number of sessions in pool
            session_options: ONNX Runtime session options
        """
        self.model_path = model_path
        self.providers = providers
        self.max_sessions = max_sessions
        self.session_options = session_options or ort.SessionOptions()

        # Thread-safe session management
        self._sessions = []
        self._available_sessions = []
        self._lock = threading.Lock()
        self._session_counter = 0

        # Initialize sessions
        self._initialize_sessions()

        api_logger.info(f"SessionPool initialized with {len(self._sessions)} sessions")
        api_logger.info(f"Providers: {self.providers}")

    def _initialize_sessions(self):
        """Initialize ONNX Runtime sessions."""
        for i in range(self.max_sessions):
            try:
                session = ort.InferenceSession(
                    self.model_path,
                    providers=self.providers,
                    sess_options=self.session_options,
                )

                session_id = f"session_{i}"
                session._session_id = session_id  # Add custom ID

                self._sessions.append(session)
                self._available_sessions.append(session)

                api_logger.debug(f"Created session {session_id}")

            except Exception as e:
                api_logger.error(f"Failed to create session {i}: {e}")
                raise

    def acquire_session(self, timeout: float = 5.0) -> ort.InferenceSession:
        """
        Acquire a session from the pool.

        Args:
            timeout: Maximum time to wait for available session

        Returns:
            ONNX Runtime session

        Raises:
            RuntimeError: If no session available within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._lock:
                if self._available_sessions:
                    session = self._available_sessions.pop()
                    api_logger.debug(f"Acquired session {session._session_id}")
                    return session

            # Wait a bit before retrying
            time.sleep(0.01)

        raise RuntimeError(f"No session available within {timeout}s timeout")

    def release_session(self, session: ort.InferenceSession):
        """
        Release a session back to the pool.

        Args:
            session: ONNX Runtime session to release
        """
        with self._lock:
            if session not in self._available_sessions:
                self._available_sessions.append(session)
                api_logger.debug(f"Released session {session._session_id}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about session pool."""
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "available_sessions": len(self._available_sessions),
                "busy_sessions": len(self._sessions) - len(self._available_sessions),
                "providers": self.providers,
                "model_path": self.model_path,
            }

    def cleanup(self):
        """Clean up all sessions."""
        with self._lock:
            api_logger.info("Cleaning up session pool...")
            self._sessions.clear()
            self._available_sessions.clear()


class ResultCache:
    """LRU cache for inference results to improve performance."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize result cache.

        Args:
            max_size: Maximum number of cached results
        """
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()

        api_logger.info(f"ResultCache initialized with max_size={max_size}")

    def _generate_key(self, input_data: np.ndarray) -> str:
        """Generate cache key from input data."""
        # Use hash of input data as key
        data_bytes = input_data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def get(self, input_data: np.ndarray) -> Optional[InferenceResult]:
        """
        Get cached result for input data.

        Args:
            input_data: Input numpy array

        Returns:
            Cached inference result or None
        """
        key = self._generate_key(input_data)

        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                result = self._cache.pop(key)
                self._cache[key] = result
                api_logger.debug(f"Cache hit for key {key[:8]}...")
                return result

        return None

    def put(self, input_data: np.ndarray, result: InferenceResult):
        """
        Cache inference result.

        Args:
            input_data: Input numpy array
            result: Inference result to cache
        """
        key = self._generate_key(input_data)

        with self._lock:
            # Remove oldest items if cache is full
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = result
            api_logger.debug(f"Cached result for key {key[:8]}...")

    def clear(self):
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            api_logger.info("Result cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size,
            }


class ONNXInferenceEngine:
    """
    High-performance ONNX inference engine with advanced features.

    Features:
    - Session pooling for concurrent inference
    - Result caching for improved performance
    - Batch processing optimization
    - Provider selection and optimization
    - Performance monitoring and metrics
    - Memory-efficient processing
    """

    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        providers: Optional[List[str]] = None,
        max_sessions: int = 4,
        enable_caching: bool = True,
        cache_size: int = 1000,
        optimization_level: str = "basic",  # basic, extended, all
    ):
        """
        Initialize ONNX inference engine.

        Args:
            model_path: Path to ONNX model file
            class_names: List of class names for predictions
            providers: List of execution providers (auto-detected if None)
            max_sessions: Maximum number of concurrent sessions
            enable_caching: Whether to enable result caching
            cache_size: Maximum number of cached results
            optimization_level: ONNX optimization level
        """
        self.model_path = Path(model_path)
        self.class_names = class_names or ["healthy", "angular_leaf_spot", "bean_rust"]
        self.num_classes = len(self.class_names)
        self.enable_caching = enable_caching

        # Validate model file
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Setup providers
        self.providers = self._setup_providers(providers)

        # Setup session options
        session_options = self._create_session_options(optimization_level)

        # Initialize session pool
        self.session_pool = SessionPool(
            str(self.model_path), self.providers, max_sessions, session_options
        )

        # Initialize result cache
        self.result_cache = ResultCache(cache_size) if enable_caching else None

        # Get model metadata
        self.model_metadata = self._get_model_metadata()

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self._stats_lock = threading.Lock()

        api_logger.info(f"ONNXInferenceEngine initialized:")
        api_logger.info(f"  Model: {self.model_path}")
        api_logger.info(f"  Classes: {self.num_classes}")
        api_logger.info(f"  Providers: {self.providers}")
        api_logger.info(f"  Sessions: {max_sessions}")
        api_logger.info(f"  Caching: {enable_caching}")
        api_logger.info(f"  Input shape: {self.model_metadata['input_shape']}")
        api_logger.info(f"  Output shape: {self.model_metadata['output_shape']}")

    def _setup_providers(self, providers: Optional[List[str]]) -> List[str]:
        """Setup and validate execution providers."""
        if providers:
            return providers

        # Auto-detect best available providers
        available_providers = ort.get_available_providers()
        preferred_providers = []

        # Add providers in order of preference
        if InferenceProvider.CUDA.value in available_providers:
            preferred_providers.append(InferenceProvider.CUDA.value)
        if InferenceProvider.COREML.value in available_providers:
            preferred_providers.append(InferenceProvider.COREML.value)
        if InferenceProvider.OPENVINO.value in available_providers:
            preferred_providers.append(InferenceProvider.OPENVINO.value)

        # Always include CPU as fallback
        preferred_providers.append(InferenceProvider.CPU.value)

        api_logger.info(f"Available providers: {available_providers}")
        api_logger.info(f"Selected providers: {preferred_providers}")

        return preferred_providers

    def _create_session_options(self, optimization_level: str) -> ort.SessionOptions:
        """Create optimized session options."""
        session_options = ort.SessionOptions()

        # Set optimization level
        if optimization_level == "basic":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        elif optimization_level == "extended":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        elif optimization_level == "all":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        # Enable memory pattern optimization
        session_options.enable_mem_pattern = True

        # Enable CPU memory arena
        session_options.enable_cpu_mem_arena = True

        # Set execution mode
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Set thread count (use half of available cores)
        import multiprocessing

        session_options.intra_op_num_threads = max(1, multiprocessing.cpu_count() // 2)

        api_logger.info(f"Session options configured:")
        api_logger.info(f"  Optimization level: {optimization_level}")
        api_logger.info(f"  Threads: {session_options.intra_op_num_threads}")

        return session_options

    def _get_model_metadata(self) -> Dict[str, Any]:
        """Get model input/output metadata."""
        # Use a temporary session to get metadata
        temp_session = ort.InferenceSession(
            str(self.model_path), providers=self.providers
        )

        input_info = temp_session.get_inputs()[0]
        output_info = temp_session.get_outputs()[0]

        metadata = {
            "input_name": input_info.name,
            "input_shape": input_info.shape,
            "input_type": input_info.type,
            "output_name": output_info.name,
            "output_shape": output_info.shape,
            "output_type": output_info.type,
        }

        return metadata

    async def predict_single(
        self,
        input_data: np.ndarray,
        return_probabilities: bool = True,
        use_cache: bool = True,
    ) -> InferenceResult:
        """
        Perform single image inference.

        Args:
            input_data: Preprocessed input array (1, C, H, W)
            return_probabilities: Whether to return class probabilities
            use_cache: Whether to use result caching

        Returns:
            Inference result with metadata
        """
        # Check cache first
        if use_cache and self.result_cache:
            cached_result = self.result_cache.get(input_data)
            if cached_result:
                return cached_result

        # Perform inference
        start_time = time.time()

        # Acquire session from pool
        session = self.session_pool.acquire_session()

        try:
            # Run inference
            input_name = self.model_metadata["input_name"]
            outputs = session.run(None, {input_name: input_data})
            predictions = outputs[0]

            inference_time = time.time() - start_time

            # Process results
            result = self._process_single_prediction(
                predictions[0],  # Remove batch dimension
                return_probabilities,
                inference_time,
                session._session_id,
                session.get_providers()[0],
            )

            # Cache result
            if use_cache and self.result_cache:
                self.result_cache.put(input_data, result)

            # Update statistics
            self._update_stats(inference_time)

            return result

        finally:
            # Always release session back to pool
            self.session_pool.release_session(session)

    async def predict_batch(
        self,
        input_batch: np.ndarray,
        return_probabilities: bool = True,
        batch_size: Optional[int] = None,
    ) -> BatchInferenceResult:
        """
        Perform batch inference with optimal batching.

        Args:
            input_batch: Batch of preprocessed input arrays (N, C, H, W)
            return_probabilities: Whether to return class probabilities
            batch_size: Optional batch size for chunking large batches

        Returns:
            Batch inference result with metadata
        """
        total_start_time = time.time()

        # Determine optimal batch size
        if batch_size is None:
            batch_size = min(input_batch.shape[0], 32)  # Max 32 per batch

        # Split into chunks if necessary
        if input_batch.shape[0] <= batch_size:
            # Single batch processing
            return await self._process_single_batch(
                input_batch, return_probabilities, total_start_time
            )
        else:
            # Multi-batch processing
            return await self._process_multi_batch(
                input_batch, return_probabilities, batch_size, total_start_time
            )

    async def _process_single_batch(
        self, input_batch: np.ndarray, return_probabilities: bool, start_time: float
    ) -> BatchInferenceResult:
        """Process a single batch."""
        # Acquire session from pool
        session = self.session_pool.acquire_session()

        try:
            # Run batch inference
            input_name = self.model_metadata["input_name"]
            outputs = session.run(None, {input_name: input_batch})
            predictions = outputs[0]

            inference_time = time.time() - start_time

            # Process all predictions
            results = []
            for i, pred in enumerate(predictions):
                result = self._process_single_prediction(
                    pred,
                    return_probabilities,
                    inference_time / len(predictions),
                    session._session_id,
                    session.get_providers()[0],
                    batch_index=i,
                )
                results.append(result)

            # Update statistics
            self._update_stats(inference_time)

            return BatchInferenceResult(
                results=results,
                batch_size=len(results),
                total_inference_time=inference_time,
                average_inference_time=inference_time / len(results),
                session_id=session._session_id,
                provider=session.get_providers()[0],
            )

        finally:
            self.session_pool.release_session(session)

    async def _process_multi_batch(
        self,
        input_batch: np.ndarray,
        return_probabilities: bool,
        batch_size: int,
        start_time: float,
    ) -> BatchInferenceResult:
        """Process multiple batches concurrently."""
        # Split input into chunks
        chunks = [
            input_batch[i : i + batch_size]
            for i in range(0, input_batch.shape[0], batch_size)
        ]

        # Process chunks concurrently
        tasks = [
            self._process_single_batch(chunk, return_probabilities, time.time())
            for chunk in chunks
        ]

        chunk_results = await asyncio.gather(*tasks)

        # Combine results
        all_results = []
        total_inference_time = 0.0

        for chunk_result in chunk_results:
            all_results.extend(chunk_result.results)
            total_inference_time += chunk_result.total_inference_time

        total_time = time.time() - start_time

        return BatchInferenceResult(
            results=all_results,
            batch_size=len(all_results),
            total_inference_time=total_time,
            average_inference_time=total_time / len(all_results),
            session_id="multi_batch",
            provider=chunk_results[0].provider if chunk_results else "unknown",
        )

    def _process_single_prediction(
        self,
        prediction: np.ndarray,
        return_probabilities: bool,
        inference_time: float,
        session_id: str,
        provider: str,
        batch_index: Optional[int] = None,
    ) -> InferenceResult:
        """
        Process a single prediction into structured result.

        Args:
            prediction: Raw model output
            return_probabilities: Whether to include probabilities
            inference_time: Time taken for inference
            session_id: ID of session used
            provider: Execution provider used
            batch_index: Index in batch (if applicable)

        Returns:
            Structured inference result
        """
        # Apply softmax to get probabilities
        probabilities = self._softmax(prediction)

        # Get predicted class
        predicted_class_id = int(np.argmax(probabilities))
        predicted_class_name = self.class_names[predicted_class_id]
        confidence = float(probabilities[predicted_class_id])

        # Create result
        result = InferenceResult(
            class_id=predicted_class_id,
            class_name=predicted_class_name,
            confidence=confidence,
            inference_time=inference_time,
            session_id=session_id,
            provider=provider,
            batch_index=batch_index,
        )

        # Add probabilities if requested
        if return_probabilities:
            result.probabilities = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            }

        return result

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax function to get probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _update_stats(self, inference_time: float):
        """Update performance statistics."""
        with self._stats_lock:
            self.inference_count += 1
            self.total_inference_time += inference_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Performance metrics dictionary
        """
        with self._stats_lock:
            avg_inference_time = (
                self.total_inference_time / self.inference_count
                if self.inference_count > 0
                else 0.0
            )

            stats = {
                "inference_count": self.inference_count,
                "total_inference_time": self.total_inference_time,
                "average_inference_time": avg_inference_time,
                "throughput_per_second": (
                    1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
                ),
            }

        # Add session pool stats
        stats["session_pool"] = self.session_pool.get_session_info()

        # Add cache stats
        if self.result_cache:
            stats["cache"] = self.result_cache.get_stats()

        return stats

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.

        Returns:
            Model information dictionary
        """
        return {
            "model_path": str(self.model_path),
            "model_size_mb": self.model_path.stat().st_size / (1024 * 1024),
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "providers": self.providers,
            "metadata": self.model_metadata,
            "caching_enabled": self.enable_caching,
            "session_pool_size": self.session_pool.max_sessions,
        }

    async def warmup(self, num_warmup_runs: int = 5) -> Dict[str, Any]:
        """
        Warm up the inference engine with dummy data.

        Args:
            num_warmup_runs: Number of warmup inference runs

        Returns:
            Warmup statistics
        """
        api_logger.info(f"Warming up inference engine with {num_warmup_runs} runs...")

        # Create dummy input data
        input_shape = self.model_metadata["input_shape"]
        # Handle dynamic batch size and string dimensions
        processed_shape = []
        for i, dim in enumerate(input_shape):
            if dim == -1 or dim is None or isinstance(dim, str):
                if i == 0:  # Batch dimension
                    processed_shape.append(1)
                else:
                    # For other dimensions, use common values
                    if i == 1:  # Channels
                        processed_shape.append(3)
                    else:  # Height/Width
                        processed_shape.append(224)
            else:
                processed_shape.append(int(dim))

        input_shape = processed_shape

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        warmup_times = []

        for i in range(num_warmup_runs):
            start_time = time.time()

            try:
                await self.predict_single(
                    dummy_input, return_probabilities=False, use_cache=False
                )
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)

                api_logger.debug(f"Warmup run {i+1}: {warmup_time:.3f}s")

            except Exception as e:
                api_logger.error(f"Warmup run {i+1} failed: {e}")

        if warmup_times:
            avg_warmup_time = sum(warmup_times) / len(warmup_times)
            min_warmup_time = min(warmup_times)
            max_warmup_time = max(warmup_times)

            api_logger.info(f"Warmup completed:")
            api_logger.info(f"  Average time: {avg_warmup_time:.3f}s")
            api_logger.info(f"  Min time: {min_warmup_time:.3f}s")
            api_logger.info(f"  Max time: {max_warmup_time:.3f}s")

            return {
                "warmup_runs": len(warmup_times),
                "average_time": avg_warmup_time,
                "min_time": min_warmup_time,
                "max_time": max_warmup_time,
                "total_time": sum(warmup_times),
            }
        else:
            api_logger.warning("No successful warmup runs completed")
            return {"warmup_runs": 0, "error": "All warmup runs failed"}

    async def benchmark(
        self,
        num_runs: int = 100,
        batch_sizes: List[int] = None,
        return_detailed_stats: bool = False,
    ) -> Dict[str, Any]:
        """
        Benchmark the inference engine performance.

        Args:
            num_runs: Number of benchmark runs per batch size
            batch_sizes: List of batch sizes to test
            return_detailed_stats: Whether to return detailed timing data

        Returns:
            Benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        api_logger.info(f"Starting benchmark with {num_runs} runs per batch size...")

        benchmark_results = {}

        for batch_size in batch_sizes:
            api_logger.info(f"Benchmarking batch size {batch_size}...")

            # Create dummy batch data
            input_shape = self.model_metadata["input_shape"]
            # Handle dynamic batch size and string dimensions
            processed_shape = []
            for i, dim in enumerate(input_shape):
                if i == 0:  # Batch dimension
                    processed_shape.append(batch_size)
                elif dim == -1 or dim is None or isinstance(dim, str):
                    # For other dimensions, use common values
                    if i == 1:  # Channels
                        processed_shape.append(3)
                    else:  # Height/Width
                        processed_shape.append(224)
                else:
                    processed_shape.append(int(dim))

            input_shape = processed_shape

            dummy_batch = np.random.randn(*input_shape).astype(np.float32)

            batch_times = []

            for run in range(num_runs):
                start_time = time.time()

                try:
                    if batch_size == 1:
                        await self.predict_single(
                            dummy_batch, return_probabilities=False, use_cache=False
                        )
                    else:
                        await self.predict_batch(
                            dummy_batch, return_probabilities=False
                        )

                    run_time = time.time() - start_time
                    batch_times.append(run_time)

                except Exception as e:
                    api_logger.error(
                        f"Benchmark run {run+1} failed for batch size {batch_size}: {e}"
                    )

            if batch_times:
                avg_time = sum(batch_times) / len(batch_times)
                min_time = min(batch_times)
                max_time = max(batch_times)
                throughput = batch_size / avg_time

                batch_result = {
                    "batch_size": batch_size,
                    "runs": len(batch_times),
                    "average_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "throughput_images_per_second": throughput,
                    "average_time_per_image": avg_time / batch_size,
                }

                if return_detailed_stats:
                    batch_result["all_times"] = batch_times

                benchmark_results[f"batch_{batch_size}"] = batch_result

                api_logger.info(
                    f"Batch {batch_size}: {avg_time:.3f}s avg, "
                    f"{throughput:.1f} images/sec"
                )

        return {
            "benchmark_results": benchmark_results,
            "model_info": self.get_model_info(),
            "performance_stats": self.get_performance_stats(),
        }

    def clear_cache(self):
        """Clear the result cache."""
        if self.result_cache:
            self.result_cache.clear()
            api_logger.info("Result cache cleared")

    def reset_stats(self):
        """Reset performance statistics."""
        with self._stats_lock:
            self.inference_count = 0
            self.total_inference_time = 0.0
        api_logger.info("Performance statistics reset")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the inference engine.

        Returns:
            Health check results
        """
        try:
            # Test with dummy data
            input_shape = self.model_metadata["input_shape"]
            # Handle dynamic batch size and string dimensions
            processed_shape = []
            for i, dim in enumerate(input_shape):
                if dim == -1 or dim is None or isinstance(dim, str):
                    if i == 0:  # Batch dimension
                        processed_shape.append(1)
                    else:
                        # For other dimensions, use common values
                        if i == 1:  # Channels
                            processed_shape.append(3)
                        else:  # Height/Width
                            processed_shape.append(224)
                else:
                    processed_shape.append(int(dim))

            input_shape = processed_shape

            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            start_time = time.time()
            result = await self.predict_single(
                dummy_input, return_probabilities=False, use_cache=False
            )
            health_check_time = time.time() - start_time

            return {
                "status": "healthy",
                "health_check_time": health_check_time,
                "test_prediction": {
                    "class_name": result.class_name,
                    "confidence": result.confidence,
                    "inference_time": result.inference_time,
                },
                "session_pool": self.session_pool.get_session_info(),
                "cache_stats": (
                    self.result_cache.get_stats() if self.result_cache else None
                ),
            }

        except Exception as e:
            api_logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "session_pool": self.session_pool.get_session_info(),
            }

    async def cleanup(self):
        """Clean up resources."""
        api_logger.info("Cleaning up ONNX inference engine...")

        # Clear cache
        if self.result_cache:
            self.result_cache.clear()

        # Cleanup session pool
        self.session_pool.cleanup()

        api_logger.info("ONNX inference engine cleanup completed")


# Utility functions for easy engine creation


def create_optimized_engine(
    model_path: str, class_names: List[str] = None, device: str = "auto"
) -> ONNXInferenceEngine:
    """
    Create an optimized ONNX inference engine with best practices.

    Args:
        model_path: Path to ONNX model file
        class_names: List of class names
        device: Target device (auto, cpu, cuda)

    Returns:
        Configured ONNX inference engine
    """
    # Determine optimal providers based on device
    if device == "auto":
        providers = None  # Auto-detect
    elif device == "cpu":
        providers = [InferenceProvider.CPU]
    elif device == "cuda":
        providers = [InferenceProvider.CUDA, InferenceProvider.CPU]
    else:
        providers = None

    # Create engine with optimized settings
    engine = ONNXInferenceEngine(
        model_path=model_path,
        class_names=class_names,
        providers=providers,
        max_sessions=4,  # Good balance for most systems
        enable_caching=True,
        cache_size=1000,
        optimization_level="extended",  # Good balance of speed vs compatibility
    )

    return engine


async def benchmark_model(
    model_path: str, class_names: List[str] = None, num_runs: int = 50
) -> Dict[str, Any]:
    """
    Quick benchmark of an ONNX model.

    Args:
        model_path: Path to ONNX model file
        class_names: List of class names
        num_runs: Number of benchmark runs

    Returns:
        Benchmark results
    """
    engine = create_optimized_engine(model_path, class_names)

    try:
        # Warmup
        await engine.warmup(num_warmup_runs=5)

        # Benchmark
        results = await engine.benchmark(num_runs=num_runs)

        return results

    finally:
        await engine.cleanup()
