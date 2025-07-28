#!/usr/bin/env python3
# this_file: src/uubed/gpu.py
"""GPU acceleration for uubed encoding using CuPy.

This module provides GPU-accelerated implementations of uubed encoders
using CuPy for CUDA operations. It includes functions for checking GPU
availability, retrieving GPU information, and performing batch and streaming
encoding on the GPU. A fallback to CPU implementations is provided when
GPU is not available or for methods that do not significantly benefit from GPU.

Requires: `pip install cupy-cuda11x` (or appropriate CUDA version for your system).
"""

import struct
from collections.abc import Iterable, Iterator
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import cupy as cp
    from cupy import ndarray as CupyArray
    GPU_AVAILABLE: bool = True
except ImportError:
    # If CuPy is not installed, set GPU_AVAILABLE to False and define dummy types.
    cp = None
    CupyArray = Any  # type: ignore # Dummy type for type hints when CuPy is absent.
    GPU_AVAILABLE: bool = False

from .api import encode as cpu_encode
from .encoders.q64 import q64_encode as cpu_q64_encode


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available on the system.

    Returns:
        bool: True if CuPy is installed and a CUDA-enabled GPU is detected, False otherwise.
    """
    return GPU_AVAILABLE


def get_gpu_info() -> dict[str, Any]:
    """Retrieve detailed information about the detected GPU(s).

    Returns:
        Dict[str, Any]: A dictionary containing GPU availability status and, if available,
                        details such as device count, current device ID, device name,
                        compute capability, and memory statistics (total, free, used).
                        If GPU is not available, it provides a reason.
    """
    if not GPU_AVAILABLE:
        return {"available": False, "reason": "CuPy not installed or no CUDA-enabled GPU detected."}

    try:
        # Get information about the current CUDA device.
        device = cp.cuda.Device()
        # Get memory usage information.
        memory_info = cp.cuda.MemoryInfo()

        return {
            "available": True,
            "device_count": cp.cuda.runtime.getDeviceCount(),
            "current_device": device.id,
            "device_name": device.name,
            "compute_capability": device.compute_capability,
            "total_memory": memory_info.total,
            "free_memory": memory_info.free,
            "used_memory": memory_info.used,
        }
    except Exception as e:
        # Catch any exceptions during GPU info retrieval (e.g., no CUDA context).
        return {"available": False, "reason": f"Error retrieving GPU info: {e!s}"}


class GPUEncoder:
    """Provides GPU-accelerated encoding methods for batch processing of embeddings.

    This class encapsulates the logic for performing encoding operations directly
    on the GPU using CuPy, leveraging CUDA capabilities for performance.
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize the GPUEncoder.

        Args:
            device_id (int): The ID of the CUDA device to use for encoding operations.
                             Defaults to 0 (the first GPU).

        Raises:
            RuntimeError: If GPU acceleration is not available (CuPy not installed or no CUDA device).
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration not available. Please install CuPy and ensure a CUDA-enabled GPU is present.")

        self.device_id: int = device_id
        # Set the current CUDA device for this encoder instance.
        cp.cuda.Device(device_id).use()

    def _to_gpu(self, data: np.ndarray | list[int]) -> CupyArray:
        """Transfers data from CPU (NumPy array or list) to GPU (CuPy array).

        Args:
            data (Union[np.ndarray, List[int]]): The input data to transfer. If a list,
                                                  it will be converted to a NumPy array first.

        Returns:
            CupyArray: The data as a CuPy ndarray on the GPU.
        """
        if isinstance(data, list):
            # Convert list to NumPy array, ensuring uint8 dtype.
            data = np.array(data, dtype=np.uint8)
        elif data.dtype != np.uint8:
            # Ensure NumPy array is of uint8 dtype.
            data = data.astype(np.uint8)
        # Transfer the NumPy array to the GPU.
        return cp.asarray(data)

    def _to_cpu(self, data: CupyArray) -> np.ndarray:
        """Transfers data from GPU (CuPy array) to CPU (NumPy array).

        Args:
            data (CupyArray): The CuPy ndarray on the GPU.

        Returns:
            np.ndarray: The data as a NumPy ndarray on the CPU.
        """
        # Transfer the CuPy array back to the CPU as a NumPy array.
        return cp.asnumpy(data)

    def batch_simhash_q64_gpu(
        self,
        embeddings: list[np.ndarray] | np.ndarray,
        planes: int = 64,
        seed: int = 42
    ) -> list[str]:
        """
        Performs GPU-accelerated batch SimHash (shq64) encoding.

        This method generates random projection planes on the GPU and performs
        matrix multiplication to compute SimHash bits. The final bit packing
        and q64 encoding are done on the CPU.

        Args:
            embeddings (Union[List[np.ndarray], np.ndarray]): A list of NumPy arrays
                                                               or a 2D NumPy array representing
                                                               a batch of embeddings.
            planes (int): The number of random projection planes to use. Defaults to 64.
            seed (int): The random seed for reproducibility of random projections. Defaults to 42.
            
        Returns:
            List[str]: A list of encoded strings, one for each embedding in the batch.
        """
        # Determine batch size and embedding dimension.
        if isinstance(embeddings, list):
            batch_size: int = len(embeddings)
            # Assuming all embeddings in the list have the same dimension.
            embedding_dim: int = embeddings[0].shape[0]
            # Stack embeddings into a single CuPy array on GPU.
            gpu_embeddings: CupyArray = cp.stack([self._to_gpu(emb) for emb in embeddings])
        else:
            batch_size, embedding_dim = embeddings.shape
            gpu_embeddings = self._to_gpu(embeddings)

        # Generate random projection matrix on the GPU for SimHash.
        cp.random.seed(seed)
        rand_vectors: CupyArray = cp.random.normal(size=(planes, embedding_dim), dtype=cp.float32)

        # Convert embeddings to float32 and center them for projection.
        gpu_embeddings = gpu_embeddings.astype(cp.float32)
        gpu_embeddings = (gpu_embeddings - 128) / 128

        # Perform batch matrix multiplication: (batch_size, embedding_dim) @ (embedding_dim, planes) -> (batch_size, planes).
        projections: CupyArray = gpu_embeddings @ rand_vectors.T

        # Get sign bits (1 if positive, 0 if non-positive).
        bits: CupyArray = (projections > 0).astype(cp.uint8)

        # Transfer the bits back to CPU for packing into bytes and q64 encoding.
        cpu_bits: np.ndarray = self._to_cpu(bits)

        results: list[str] = []
        for i in range(batch_size):
            # Pack 8 bits into a single byte.
            byte_data: list[int] = []
            for j in range(0, planes, 8):
                byte_val: int = 0
                for k in range(8):
                    if j + k < planes:
                        # Shift bits into position (MSB first).
                        byte_val |= int(cpu_bits[i, j + k]) << (7 - k)
                byte_data.append(byte_val)

            # Encode the packed bytes using the CPU-based q64 encoder.
            encoded: str = cpu_q64_encode(bytes(byte_data))
            results.append(encoded)

        return results

    def batch_top_k_q64_gpu(
        self,
        embeddings: list[np.ndarray] | np.ndarray,
        k: int = 8
    ) -> list[str]:
        """
        Performs GPU-accelerated batch Top-K (t8q64) encoding.

        This method identifies the top-k indices for each embedding on the GPU,
        sorts them, clamps them to a byte range, and then transfers them to CPU
        for q64 encoding.

        Args:
            embeddings (Union[List[np.ndarray], np.ndarray]): A list of NumPy arrays
                                                               or a 2D NumPy array representing
                                                               a batch of embeddings.
            k (int): The number of top indices to keep for each embedding. Defaults to 8.
            
        Returns:
            List[str]: A list of encoded strings, one for each embedding in the batch.
        """
        # Convert embeddings to CuPy array on GPU.
        if isinstance(embeddings, list):
            batch_size: int = len(embeddings)
            gpu_embeddings: CupyArray = cp.stack([self._to_gpu(emb) for emb in embeddings])
        else:
            batch_size: int = embeddings.shape[0]
            gpu_embeddings = self._to_gpu(embeddings)

        # Get the indices of the top-k largest elements along the last axis (embedding dimension).
        # `argpartition` is used for efficiency as it only guarantees the k-th element is in place.
        top_k_indices: CupyArray = cp.argpartition(gpu_embeddings, -k, axis=1)[:, -k:]

        # Sort the top-k indices to ensure a consistent order for encoding.
        sorted_indices: CupyArray = cp.sort(top_k_indices, axis=1)

        # Clamp the indices to 255 (max value for uint8) to prevent overflow during byte conversion.
        clamped_indices: CupyArray = cp.minimum(sorted_indices, 255)

        # Transfer the processed indices back to CPU for q64 encoding.
        cpu_indices: np.ndarray = self._to_cpu(clamped_indices)

        results: list[str] = []
        for i in range(batch_size):
            indices: list[int] = cpu_indices[i].tolist()

            # Pad the list of indices with 255 if its length is less than k.
            # This ensures a consistent output length for encoding.
            while len(indices) < k:
                indices.append(255)

            # Encode the indices as bytes using the CPU-based q64 encoder.
            encoded: str = cpu_q64_encode(bytes(indices))
            results.append(encoded)

        return results

    def batch_z_order_q64_gpu(
        self,
        embeddings: list[np.ndarray] | np.ndarray
    ) -> list[str]:
        """
        Performs GPU-accelerated batch Z-order (zoq64) encoding.

        This method quantizes embeddings and then interleaves bits to create
        Z-order curves. The bit interleaving and final q64 encoding are done on the CPU.

        Args:
            embeddings (Union[List[np.ndarray], np.ndarray]): A list of NumPy arrays
                                                               or a 2D NumPy array representing
                                                               a batch of embeddings.
            
        Returns:
            List[str]: A list of encoded strings, one for each embedding in the batch.
        """
        # Convert embeddings to CuPy array on GPU.
        if isinstance(embeddings, list):
            batch_size: int = len(embeddings)
            gpu_embeddings: CupyArray = cp.stack([self._to_gpu(emb) for emb in embeddings])
        else:
            batch_size: int = embeddings.shape[0]
            gpu_embeddings = self._to_gpu(embeddings)

        # Quantize each dimension to 2 bits (take the top 2 most significant bits).
        quantized: CupyArray = (gpu_embeddings >> 6) & 0b11

        # Take the first 16 dimensions for Z-order encoding (common practice for 32-bit Z-order).
        quantized_16: CupyArray = quantized[:, :16]

        # Transfer the quantized data to CPU for bit interleaving, as it's complex on GPU.
        cpu_quantized: np.ndarray = self._to_cpu(quantized_16)

        results: list[str] = []
        for i in range(batch_size):
            result: int = 0
            # Iterate through each quantized value (dimension).
            for j, val in enumerate(cpu_quantized[i]):
                # Interleave the 2 bits of the current value into the result.
                for bit_pos in range(2):
                    bit: int = (val >> bit_pos) & 1
                    # Place the bit at the correct interleaved position.
                    result |= bit << (j * 2 + bit_pos)

            # Pack the 32-bit integer result into 4 bytes.
            packed: bytes = struct.pack(">I", result) # '>I' for big-endian unsigned int.
            # Encode the packed bytes using the CPU-based q64 encoder.
            encoded: str = cpu_q64_encode(packed)
            results.append(encoded)

        return results

def gpu_encode_batch(
    embeddings: list[np.ndarray] | np.ndarray,
    method: str = "shq64",
    device_id: int = 0,
    **kwargs: Any
) -> list[str]:
    """
    Performs GPU-accelerated batch encoding for a list or array of embeddings.

    This function acts as a dispatcher, routing the encoding task to the appropriate
    GPU-accelerated method within `GPUEncoder`. It also provides a fallback to
    CPU encoding if GPU is not available or if the method does not have a GPU implementation.

    Args:
        embeddings (Union[List[np.ndarray], np.ndarray]): A list of NumPy arrays
                                                           or a 2D NumPy array representing
                                                           a batch of embeddings.
        method (str): The encoding method to use. Supported GPU methods are "shq64", "t8q64", "zoq64".
                      "eq64" will always fall back to CPU as it typically doesn't benefit from GPU.
        device_id (int): The ID of the CUDA device to use. Defaults to 0.
        **kwargs: Method-specific parameters (e.g., `planes` for shq64, `k` for t8q64).
        
    Returns:
        List[str]: A list of encoded strings, one for each embedding in the batch.
        
    Example:
        >>> import numpy as np
        >>> embeddings = [np.random.randint(0, 256, 768, dtype=np.uint8) for _ in range(1000)]
        >>> encoded = gpu_encode_batch(embeddings, method="shq64", planes=64)
    """
    if not GPU_AVAILABLE:
        # Fallback to CPU batch processing if GPU is not available.
        return [cpu_encode(emb, method=method, **kwargs) for emb in embeddings]

    # Initialize the GPU encoder for the specified device.
    encoder: GPUEncoder = GPUEncoder(device_id=device_id)

    # Dispatch to the appropriate GPU encoding method based on the 'method' parameter.
    if method == "shq64":
        planes: int = kwargs.get("planes", 64)
        return encoder.batch_simhash_q64_gpu(embeddings, planes=planes)
    if method == "t8q64":
        k: int = kwargs.get("k", 8)
        return encoder.batch_top_k_q64_gpu(embeddings, k=k)
    if method == "zoq64":
        return encoder.batch_z_order_q64_gpu(embeddings)
    if method == "eq64":
        # eq64 (lossless encoding) typically does not benefit significantly from GPU acceleration
        # due to its bit-level operations and lack of large matrix multiplications.
        # Therefore, it's more efficient to process it on the CPU.
        return [cpu_encode(emb, method=method, **kwargs) for emb in embeddings]
    # Raise an error if an unsupported method is requested for GPU acceleration.
    raise ValueError(f"GPU acceleration not available or supported for method: {method}")


class GPUStreamingEncoder:
    """
    Provides GPU-accelerated streaming encoding for large datasets.

    This class processes embeddings in batches on the GPU to maximize throughput,
    while maintaining a streaming interface for memory efficiency. It handles
    batching internally and dispatches to the appropriate GPU encoding methods.
    """

    def __init__(
        self,
        batch_size: int = 1000,
        method: str = "shq64",
        device_id: int = 0,
        **kwargs: Any
    ):
        """
        Initialize the GPU streaming encoder.

        Args:
            batch_size (int): The number of embeddings to process in each GPU batch.
                              Defaults to 1000.
            method (str): The encoding method to use. Defaults to "shq64".
            device_id (int): The ID of the CUDA device to use. Defaults to 0.
            **kwargs: Method-specific parameters (e.g., `planes` for shq64, `k` for t8q64).
        """
        self.batch_size: int = batch_size
        self.method: str = method
        self.device_id: int = device_id
        self.kwargs: dict[str, Any] = kwargs

        # Initialize GPUEncoder if GPU is available, otherwise set to None for CPU fallback.
        self.encoder: GPUEncoder | None = None
        if GPU_AVAILABLE:
            self.encoder = GPUEncoder(device_id=device_id)

    def encode_stream(self, embeddings: Iterable[np.ndarray | list[int] | bytes]) -> Iterator[str]:
        """
        Encode a stream of embeddings using GPU batching.

        This method takes an iterable of embeddings, batches them, and processes
        these batches on the GPU (if available). It yields encoded strings as they
        become available, maintaining a streaming interface.

        Args:
            embeddings (Iterable[Union[np.ndarray, List[int], bytes]]): An iterable of embeddings
                                                                        to be encoded. Each embedding
                                                                        can be a NumPy array, list of ints, or bytes.
            
        Yields:
            str: Encoded strings, one for each input embedding.
        """
        batch: list[np.ndarray | list[int] | bytes] = []

        for embedding in embeddings:
            batch.append(embedding)

            if len(batch) >= self.batch_size:
                # Process the accumulated batch.
                if self.encoder:
                    # Use GPU encoding if encoder is initialized.
                    encoded_batch: list[str] = self._encode_batch_gpu(batch)
                else:
                    # Fallback to CPU encoding if GPU is not available.
                    encoded_batch = [cpu_encode(emb, method=self.method, **self.kwargs)
                                   for emb in batch]

                # Yield each encoded string from the processed batch.
                for encoded in encoded_batch:
                    yield encoded

                batch = [] # Clear the batch for the next set of embeddings.

        # Process any remaining embeddings in the last (possibly incomplete) batch.
        if batch:
            if self.encoder:
                encoded_batch = self._encode_batch_gpu(batch)
            else:
                encoded_batch = [cpu_encode(emb, method=self.method, **self.kwargs)
                               for emb in batch]

            for encoded in encoded_batch:
                yield encoded

    def _encode_batch_gpu(self, batch: list[np.ndarray | list[int] | bytes]) -> list[str]:
        """
        Internal method to dispatch a batch of embeddings to the appropriate GPU encoding function.

        Args:
            batch (List[Union[np.ndarray, List[int], bytes]]): A list of embeddings to encode.

        Returns:
            List[str]: A list of encoded strings.

        Raises:
            ValueError: If an unsupported method is passed for GPU encoding.
        """
        # Ensure the encoder is available before attempting GPU operations.
        if self.encoder is None:
            # This case should ideally not be reached if `encode_stream` checks `self.encoder`.
            # However, it's a safeguard.
            return [cpu_encode(emb, method=self.method, **self.kwargs) for emb in batch]

        # Dispatch to the specific GPU batch encoding method.
        if self.method == "shq64":
            planes: int = self.kwargs.get("planes", 64)
            return self.encoder.batch_simhash_q64_gpu(batch, planes=planes)
        if self.method == "t8q64":
            k: int = self.kwargs.get("k", 8)
            return self.encoder.batch_top_k_q64_gpu(batch, k=k)
        if self.method == "zoq64":
            return self.encoder.batch_z_order_q64_gpu(batch)
        # Fallback to CPU for methods not explicitly supported by GPUEncoder or unknown methods.
        # This ensures robustness even if a method is passed that doesn't have a GPU path.
        return [cpu_encode(emb, method=self.method, **self.kwargs) for emb in batch]


def benchmark_gpu_vs_cpu(
    n_embeddings: int = 1000,
    embedding_dim: int = 768,
    method: str = "shq64",
    **kwargs: Any
) -> dict[str, Any]:
    """
    Benchmarks the performance of GPU encoding against CPU encoding for a given method.

    This function generates synthetic embedding data and measures the time taken
    to encode them using both CPU and GPU (if available) implementations.

    Args:
        n_embeddings (int): The number of embeddings to generate for the benchmark.
                            Defaults to 1000.
        embedding_dim (int): The dimensionality of each embedding. Defaults to 768.
        method (str): The encoding method to test (e.g., "shq64", "t8q64", "zoq64").
                      Defaults to "shq64".
        **kwargs: Method-specific parameters to pass to the encoding functions.
        
    Returns:
        Dict[str, Any]: A dictionary containing the benchmark results, including:
                        - `n_embeddings`: Number of embeddings tested.
                        - `embedding_dim`: Dimension of embeddings.
                        - `method`: Encoding method tested.
                        - `cpu_time`: Time taken for CPU encoding (seconds).
                        - `cpu_throughput`: CPU encoding throughput (embeddings/second).
                        - `gpu_time`: Time taken for GPU encoding (seconds, if GPU available).
                        - `gpu_throughput`: GPU encoding throughput (embeddings/second, if GPU available).
                        - `speedup`: Ratio of CPU time to GPU time (if GPU available).
                        - `results_match`: Boolean indicating if CPU and GPU encoded results are identical (if GPU available).
                        - `gpu_error`: Error message if GPU benchmarking fails.
    """
    import time

    # Generate synthetic test embeddings as uint8 bytes.
    embeddings: list[np.ndarray] = [
        np.random.randint(0, 256, embedding_dim, dtype=np.uint8)
        for _ in range(n_embeddings)
    ]

    results: dict[str, Any] = {"n_embeddings": n_embeddings, "embedding_dim": embedding_dim, "method": method}

    # --- CPU Benchmark ---
    start_time: float = time.time()
    cpu_results: list[str] = [cpu_encode(emb, method=method, **kwargs) for emb in embeddings]
    cpu_time: float = time.time() - start_time
    results["cpu_time"] = cpu_time
    results["cpu_throughput"] = n_embeddings / cpu_time

    # --- GPU Benchmark (if available) ---
    if GPU_AVAILABLE:
        try:
            start_time = time.time()
            gpu_results: list[str] = gpu_encode_batch(embeddings, method=method, **kwargs)
            gpu_time: float = time.time() - start_time

            results["gpu_time"] = gpu_time
            results["gpu_throughput"] = n_embeddings / gpu_time
            results["speedup"] = cpu_time / gpu_time # Calculate speedup factor.
            results["results_match"] = cpu_results == gpu_results # Check if results are identical.
        except Exception as e:
            # Record any error that occurs during GPU benchmarking.
            results["gpu_error"] = str(e)
    else:
        results["gpu_available"] = False # Indicate that GPU was not available for benchmarking.

    return results
