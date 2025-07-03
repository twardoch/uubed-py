"""
Comprehensive input validation for the uubed package.

This module provides a suite of functions designed to rigorously validate
all user inputs, ensuring data integrity and preventing common errors.
It offers detailed error messages and suggestions to guide users in fixing
issues related to encoding methods, embedding formats, method-specific parameters,
batch processing configurations, file paths, memory usage, and GPU parameters.
"""

import sys
from typing import Any, Union, List, Tuple, Optional, Dict, TypeVar
from pathlib import Path
import numpy as np

from .exceptions import (
    UubedValidationError, 
    UubedResourceError,
    validation_error,
    resource_error
)


# Type definitions for embedding inputs to improve readability and type checking.
EmbeddingInput = Union[List[int], np.ndarray, bytes]


def validate_encoding_method(method: str) -> str:
    """
    Validates and normalizes the provided encoding method string.
    
    Args:
        method (str): The encoding method string to validate (e.g., "eq64", "shq64", "auto").
        
    Returns:
        str: The normalized (lowercase, stripped) and validated method name.
        
    Raises:
        UubedValidationError: If the method is not a string, is empty, or is not one of the recognized valid methods.
    """
    if not isinstance(method, str):
        raise validation_error(
            "Encoding method must be a string",
            parameter="method",
            expected="string (eq64, shq64, t8q64, zoq64, mq64, or auto)",
            received=f"{type(method).__name__}"
        )
    
    method = method.lower().strip()
    valid_methods: set[str] = {'eq64', 'shq64', 't8q64', 'zoq64', 'mq64', 'auto'}
    
    if method not in valid_methods:
        raise validation_error(
            f"Unknown encoding method: '{method}'",
            parameter="method",
            expected="one of: " + ", ".join(sorted(valid_methods)),
            received=method
        )
    
    return method


def validate_embedding_input(embedding: EmbeddingInput, method: str) -> np.ndarray:
    """
    Validates and normalizes an embedding input into a NumPy array of `uint8`.
    
    This function handles various input types (list of ints, NumPy array, bytes)
    and performs necessary type conversions and value range checks.
    
    Args:
        embedding (EmbeddingInput): The embedding to validate. Can be a list of integers,
                                    a NumPy array (int or float), or bytes.
        method (str): The encoding method. Used for method-specific dimension validation.
        
    Returns:
        np.ndarray: The normalized embedding as a NumPy array with `dtype=np.uint8`.
        
    Raises:
        UubedValidationError: If the embedding is `None`, empty, has an unsupported type or dtype,
                              or contains values outside the expected range (0-255 for integers,
                              [0, 1] or [0, 255] for floats).
    """
    # Check for None or empty input, which are invalid.
    if embedding is None:
        raise validation_error(
            "Embedding cannot be None",
            parameter="embedding",
            expected="array-like of integers 0-255, or bytes",
            received="None"
        )
    
    # Convert various input types to a NumPy array.
    arr: np.ndarray
    if isinstance(embedding, (list, tuple)):
        if len(embedding) == 0:
            raise validation_error(
                "Embedding cannot be empty",
                parameter="embedding",
                expected="non-empty array-like of integers 0-255",
                received="empty list/tuple"
            )
        try:
            # Attempt to convert list/tuple to uint8 array. This will fail if values are out of range.
            arr = np.array(embedding, dtype=np.uint8)
        except (ValueError, OverflowError) as e:
            raise validation_error(
                f"Cannot convert embedding to uint8 array: {e}. Values must be in range 0-255.",
                parameter="embedding",
                expected="integers in range 0-255",
                received=f"values causing overflow/error: {str(e)}"
            )
    elif isinstance(embedding, bytes):
        if len(embedding) == 0:
            raise validation_error(
                "Embedding cannot be empty",
                parameter="embedding", 
                expected="non-empty bytes",
                received="empty bytes"
            )
        # Convert bytes directly to a uint8 NumPy array.
        arr = np.frombuffer(embedding, dtype=np.uint8)
    elif isinstance(embedding, np.ndarray):
        if embedding.size == 0:
            raise validation_error(
                "Embedding cannot be empty",
                parameter="embedding",
                expected="non-empty numpy array",
                received="empty array"
            )
        
        # Handle different NumPy array dtypes.
        if embedding.dtype != np.uint8:
            if embedding.dtype in [np.float32, np.float64]:
                # Check if float values are within expected ranges ([0, 1] or [0, 255]).
                if np.all((embedding >= 0) & (embedding <= 1)):
                    # Normalize [0, 1] to [0, 255] and convert to uint8.
                    arr = (embedding * 255).astype(np.uint8)
                elif np.all((embedding >= 0) & (embedding <= 255)):
                    # Convert [0, 255] floats directly to uint8.
                    arr = embedding.astype(np.uint8)
                else:
                    raise validation_error(
                        f"Float embedding values must be in range [0, 1] or [0, 255], got range [{embedding.min():.3f}, {embedding.max():.3f}]",
                        parameter="embedding",
                        expected="float values in [0, 1] (normalized) or [0, 255]",
                        received=f"range [{embedding.min():.3f}, {embedding.max():.3f}]"
                    )
            elif np.issubdtype(embedding.dtype, np.integer):
                # Check if integer values are within the 0-255 range.
                if np.any((embedding < 0) | (embedding > 255)):
                    raise validation_error(
                        f"Integer embedding values must be in range [0, 255], got range [{embedding.min()}, {embedding.max()}]",
                        parameter="embedding",
                        expected="integers in range [0, 255]",
                        received=f"range [{embedding.min()}, {embedding.max()}]"
                    )
                # Convert integer array to uint8.
                arr = embedding.astype(np.uint8)
            else:
                raise validation_error(
                    f"Unsupported embedding dtype: {embedding.dtype}",
                    parameter="embedding",
                    expected="uint8, int, or float array",
                    received=f"dtype {embedding.dtype}"
                )
        else:
            # If already uint8, no conversion needed.
            arr = embedding
    else:
        raise validation_error(
            f"Unsupported embedding type: {type(embedding).__name__}",
            parameter="embedding",
            expected="list, tuple, bytes, or numpy array",
            received=f"{type(embedding).__name__}"
        )
    
    # Perform method-specific dimension validation.
    _validate_embedding_dimensions(arr, method)
    
    return arr


def _validate_embedding_dimensions(embedding: np.ndarray, method: str) -> None:
    """
    Validates embedding dimensions against method-specific requirements.
    
    Args:
        embedding (np.ndarray): The NumPy array representing the embedding.
        method (str): The encoding method being used.
        
    Raises:
        UubedValidationError: If the embedding size does not meet the requirements
                              for the specified encoding method.
    """
    if method == 'auto':
        # Dimension validation is skipped for 'auto' method as it adapts to input.
        return
    
    size: int = embedding.size
    
    # Define method-specific dimension requirements (min_size, max_size, and descriptive constraints).
    dimension_requirements: Dict[str, Dict[str, Any]] = {
        'eq64': {
            'min_size': 1,
            'max_size': 100000,  # A reasonable upper limit to prevent excessively large inputs.
            'constraints': "any positive size"
        },
        'shq64': {
            'min_size': 32,  # Minimum size for meaningful SimHash operations.
            'max_size': 50000, # Arbitrary upper limit.
            'constraints': "typically 128, 256, 512, 768, 1024, or 1536"
        },
        't8q64': {
            'min_size': 8,   # Must be at least 'k' (default k=8).
            'max_size': 50000, # Arbitrary upper limit.
            'constraints': "must be larger than k parameter"
        },
        'zoq64': {
            'min_size': 2,   # Minimum for Z-order curve (e.g., 2D point).
            'max_size': 10000, # Arbitrary upper limit.
            'constraints': "preferably powers of 2 or multiples thereof"
        },
        'mq64': {
            'min_size': 1, # Matryoshka can handle various sizes.
            'max_size': 100000, # Arbitrary upper limit.
            'constraints': "any positive size, typically with hierarchical structure"
        }
    }
    
    if method in dimension_requirements:
        req = dimension_requirements[method]
        # Check if embedding size is below the minimum requirement.
        if size < req['min_size']:
            raise validation_error(
                f"Embedding size ({size}) is too small for the '{method}' method.",
                parameter="embedding",
                expected=f"size >= {req['min_size']} ({req['constraints']})",
                received=f"size {size}"
            )
        # Check if embedding size is above the maximum allowed size.
        if size > req['max_size']:
            raise validation_error(
                f"Embedding size ({size}) is too large for the '{method}' method.",
                parameter="embedding", 
                expected=f"size <= {req['max_size']} ({req['constraints']})",
                received=f"size {size}"
            )


def validate_method_parameters(method: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Validates method-specific parameters passed as keyword arguments.
    
    Args:
        method (str): The encoding method for which to validate parameters.
        **kwargs: Arbitrary keyword arguments representing method-specific parameters.
        
    Returns:
        Dict[str, Any]: A dictionary containing the validated and normalized parameters.
        
    Raises:
        UubedValidationError: If any parameter is invalid (e.g., wrong type, out of range),
                              or if unknown parameters are provided for a given method.
    """
    validated_params: Dict[str, Any] = {}
    
    if method == 'shq64':
        planes: int = kwargs.get('planes', 64) # Default value for planes.
        if not isinstance(planes, int):
            raise validation_error(
                "'planes' parameter must be an integer",
                parameter="planes",
                expected="positive integer, multiple of 8",
                received=f"{type(planes).__name__}"
            )
        if planes <= 0:
            raise validation_error(
                "'planes' parameter must be positive",
                parameter="planes",
                expected="positive integer, multiple of 8", 
                received=f"{planes}"
            )
        if planes % 8 != 0:
            raise validation_error(
                "'planes' parameter must be a multiple of 8",
                parameter="planes",
                expected="multiple of 8 (e.g., 64, 128, 256)",
                received=f"{planes}"
            )
        if planes > 1024:
            raise validation_error(
                "'planes' parameter too large",
                parameter="planes",
                expected="<= 1024 for reasonable performance",
                received=f"{planes}"
            )
        validated_params['planes'] = planes
    
    elif method == 't8q64':
        k: int = kwargs.get('k', 8) # Default value for k.
        if not isinstance(k, int):
            raise validation_error(
                "'k' parameter must be an integer",
                parameter="k",
                expected="positive integer < embedding size",
                received=f"{type(k).__name__}"
            )
        if k <= 0:
            raise validation_error(
                "'k' parameter must be positive",
                parameter="k",
                expected="positive integer < embedding size",
                received=f"{k}"
            )
        if k > 1000:
            raise validation_error(
                "'k' parameter too large", 
                parameter="k",
                expected="<= 1000 for reasonable performance",
                received=f"{k}"
            )
        validated_params['k'] = k

    elif method == 'mq64':
        levels: Optional[List[int]] = kwargs.get('levels', None)
        if levels is not None:
            if not isinstance(levels, list) or not all(isinstance(l, int) and l > 0 for l in levels):
                raise validation_error(
                    "'levels' parameter must be a list of positive integers",
                    parameter="levels",
                    expected="list of positive integers",
                    received=f"{type(levels).__name__}"
                )
            if sorted(levels) != levels:
                raise validation_error(
                    "'levels' parameter must be sorted in ascending order",
                    parameter="levels",
                    expected="sorted list of positive integers",
                    received=f"{levels}"
                )
        validated_params['levels'] = levels
    
    # Define known parameters for each method to catch unknown/unsupported arguments.
    known_params: Dict[str, set[str]] = {
        'shq64': {'planes'},
        't8q64': {'k'},
        'mq64': {'levels'},
        'eq64': set(),
        'zoq64': set(),
        'auto': set() # 'auto' method does not have specific parameters.
    }
    
    # Check for any unknown parameters provided for the given method.
    if method in known_params:
        unknown_params: set[str] = set(kwargs.keys()) - known_params[method]
        if unknown_params:
            raise validation_error(
                f"Unknown parameters for '{method}' method: {', '.join(unknown_params)}",
                parameter="kwargs",
                expected=f"valid parameters: {', '.join(known_params[method]) or 'none'}",
                received=f"unknown: {', '.join(unknown_params)}"
            )
    
    return validated_params


def validate_batch_parameters(
    batch_size: Optional[int] = None, 
    max_memory_mb: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validates parameters related to batch processing.
    
    Args:
        batch_size (Optional[int]): The desired size of processing batches. If `None`, no validation is performed.
        max_memory_mb (Optional[int]): The maximum memory allowed for an operation in megabytes.
                                      If `None`, no validation is performed.
        
    Returns:
        Dict[str, Any]: A dictionary containing the validated parameters.
        
    Raises:
        UubedValidationError: If `batch_size` or `max_memory_mb` are invalid (e.g., non-integer, non-positive, too large).
    """
    validated: Dict[str, Any] = {}
    
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise validation_error(
                "'batch_size' must be an integer",
                parameter="batch_size",
                expected="positive integer",
                received=f"{type(batch_size).__name__}"
            )
        if batch_size <= 0:
            raise validation_error(
                "'batch_size' must be positive",
                parameter="batch_size",
                expected="positive integer (typically 100-10000)",
                received=f"{batch_size}"
            )
        if batch_size > 100000:
            raise validation_error(
                "'batch_size' too large",
                parameter="batch_size",
                expected="<= 100000 for reasonable memory usage",
                received=f"{batch_size}"
            )
        validated['batch_size'] = batch_size
    
    if max_memory_mb is not None:
        if not isinstance(max_memory_mb, int):
            raise validation_error(
                "'max_memory_mb' must be an integer",
                parameter="max_memory_mb",
                expected="positive integer (MB)",
                received=f"{type(max_memory_mb).__name__}"
            )
        if max_memory_mb <= 0:
            raise validation_error(
                "'max_memory_mb' must be positive",
                parameter="max_memory_mb",
                expected="positive integer (MB)",
                received=f"{max_memory_mb}"
            )
        validated['max_memory_mb'] = max_memory_mb
    
    return validated


def validate_file_path(
    path: Union[str, Path], 
    check_exists: bool = True,
    check_readable: bool = True,
    check_writable: bool = False
) -> Path:
    """
    Validates a file path with comprehensive checks for existence, readability, and writability.
    
    Args:
        path (Union[str, Path]): The file path to validate. Can be a string or a `pathlib.Path` object.
        check_exists (bool): If `True`, checks if the file or directory exists. Defaults to `True`.
        check_readable (bool): If `True`, checks if the file is readable. Requires `check_exists` to be `True`. Defaults to `True`.
        check_writable (bool): If `True`, checks if the file (or its parent directory if it doesn't exist) is writable. Defaults to `False`.
        
    Returns:
        Path: A `pathlib.Path` object representing the validated file path.
        
    Raises:
        UubedValidationError: If the `path` is not a string or `Path` object.
        UubedResourceError: If the file does not exist, is not a file, is not readable/writable,
                            or if its parent directory does not exist or is not writable.
    """
    if not isinstance(path, (str, Path)):
        raise validation_error(
            "File path must be a string or Path object",
            parameter="path",
            expected="string or pathlib.Path",
            received=f"{type(path).__name__}"
        )
    
    path_obj: Path = Path(path)
    
    # Check if the path exists.
    if check_exists and not path_obj.exists():
        raise resource_error(
            f"File or directory does not exist: '{path_obj}'",
            resource_type="file"
        )
    
    # If checking for existence and readability, ensure it's a file.
    if check_exists and check_readable and not path_obj.is_file():
        raise resource_error(
            f"Path is not a file: '{path_obj}'",
            resource_type="file"
        )
    
    # Check if the file is readable.
    if check_exists and check_readable:
        try:
            # Attempt to open and read a byte to verify readability.
            with open(path_obj, 'rb') as f:
                f.read(1)  
        except PermissionError:
            raise resource_error(
                f"File is not readable: '{path_obj}'",
                resource_type="file",
                suggestion="Check file permissions."
            )
        except Exception as e:
            raise resource_error(
                f"Cannot read file '{path_obj}': {e}",
                resource_type="file",
                suggestion="Ensure the file is not corrupted or locked."
            )
    
    # Check if the file or its parent directory is writable.
    if check_writable:
        if path_obj.exists():
            # If the file exists, check if it's a file and writable.
            if not path_obj.is_file():
                raise resource_error(
                    f"Path exists but is not a file: '{path_obj}'",
                    resource_type="file"
                )
            try:
                # Attempt to open in append mode to check writability without truncating.
                with open(path_obj, 'ab') as f:
                    pass  
            except PermissionError:
                raise resource_error(
                    f"File is not writable: '{path_obj}'",
                    resource_type="file",
                    suggestion="Check file permissions."
                )
        else:
            # If the file does not exist, check if its parent directory is writable.
            parent: Path = path_obj.parent
            if not parent.exists():
                raise resource_error(
                    f"Parent directory does not exist: '{parent}'",
                    resource_type="directory",
                    suggestion="Create the parent directory first."
                )
            if not parent.is_dir():
                raise resource_error(
                    f"Parent path is not a directory: '{parent}'",
                    resource_type="directory"
                )
            try:
                # Create and immediately delete a temporary file to test writability.
                test_file: Path = parent / f".uubed_write_test_{id(path_obj)}"
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                raise resource_error(
                    f"Parent directory is not writable: '{parent}'",
                    resource_type="directory",
                    suggestion="Check directory permissions."
                )
    
    return path_obj


def validate_memory_usage(
    estimated_bytes: int, 
    operation: str = "operation",
    max_bytes: Optional[int] = None
) -> None:
    """
    Validates that an operation's estimated memory usage does not exceed a specified limit.
    
    Args:
        estimated_bytes (int): The estimated memory usage of the operation in bytes.
        operation (str): A descriptive name of the operation being performed (e.g., "encoding").
                         Defaults to "operation".
        max_bytes (Optional[int]): The maximum allowed memory usage in bytes. If `None`,
                                   a default of 1GB is used.
        
    Raises:
        UubedResourceError: If the `estimated_bytes` exceeds `max_bytes`.
    """
    if max_bytes is None:
        max_bytes = 1024 * 1024 * 1024  # Default to 1GB.
    
    if estimated_bytes > max_bytes:
        raise resource_error(
            f"Estimated memory usage ({estimated_bytes // (1024*1024)} MB) is too high for {operation}.",
            resource_type="memory",
            available=f"{max_bytes // (1024*1024)} MB",
            required=f"{estimated_bytes // (1024*1024)} MB",
            suggestion="Try reducing batch_size, embedding dimensions, or using streaming operations to lower memory requirements."
        )


def estimate_memory_usage(
    embedding_count: int, 
    embedding_size: int,
    method: str
) -> int:
    """
    Estimates the memory usage for an encoding operation based on the number of embeddings,
    their size, and the encoding method.
    
    This is a heuristic estimate and may not reflect exact memory consumption.
    
    Args:
        embedding_count (int): The number of embeddings to be processed.
        embedding_size (int): The size of each embedding in bytes (or dimensions if 1 byte per dim).
        method (str): The encoding method being used, which influences memory overhead.
        
    Returns:
        int: The estimated total memory usage in bytes.
    """
    # Base memory consumption for storing the raw embeddings.
    base_memory: int = embedding_count * embedding_size
    
    # Multipliers to account for method-specific overhead (e.g., intermediate data structures, string encoding).
    # These are approximate values.
    method_multipliers: Dict[str, float] = {
        'eq64': 2.0,    # Base64 encoding typically doubles size for string representation.
        'shq64': 1.5,   # SimHash might involve matrix operations, but output is compact.
        't8q64': 1.2,   # Top-k involves sorting/selection, relatively low overhead.
        'zoq64': 1.8,   # Z-order involves bit manipulation, some intermediate storage.
        'mq64': 2.5,    # Matryoshka might have higher overhead due to hierarchical processing.
        'auto': 2.5     # Assume worst-case overhead for auto-detection.
    }
    
    multiplier: float = method_multipliers.get(method, 2.0)
    return int(base_memory * multiplier)


def validate_gpu_parameters(device_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Validates GPU-related parameters and checks for GPU availability.
    
    Args:
        device_id (Optional[int]): The CUDA device ID to validate. If `None`, only checks
                                   for general GPU availability.
        
    Returns:
        Dict[str, Any]: A dictionary containing validated parameters (e.g., `device_id`).
        
    Raises:
        UubedResourceError: If CuPy is not installed, or if the specified `device_id` is invalid
                            or the GPU is inaccessible.
        UubedValidationError: If `device_id` is not an integer or is negative.
    """
    validated: Dict[str, Any] = {}
    
    # Attempt to import cupy to check for GPU availability.
    try:
        import cupy as cp
        gpu_available: bool = True
    except ImportError:
        raise resource_error(
            "GPU acceleration not available. CuPy is not installed.",
            resource_type="gpu",
            suggestion="Install CuPy: `pip install cupy-cuda11x` (or appropriate CUDA version for your system)."
        )
    
    if device_id is not None:
        if not isinstance(device_id, int):
            raise validation_error(
                "'device_id' must be an integer",
                parameter="device_id",
                expected="non-negative integer",
                received=f"{type(device_id).__name__}"
            )
        if device_id < 0:
            raise validation_error(
                "'device_id' must be non-negative",
                parameter="device_id", 
                expected="non-negative integer",
                received=f"{device_id}"
            )
        
        # Check if the specified device_id corresponds to an available GPU.
        try:
            # Get the total number of available CUDA devices.
            device_count: int = cp.cuda.runtime.getDeviceCount()
            if device_id >= device_count:
                raise resource_error(
                    f"GPU device {device_id} not available.",
                    resource_type="gpu",
                    available=f"{device_count} devices",
                    required=f"device {device_id}",
                    suggestion=f"Use a 'device_id' in the range 0-{device_count-1}."
                )
        except Exception as e:
            # Catch any other exceptions during device access (e.g., CUDA driver issues).
            raise resource_error(
                f"Cannot access GPU device {device_id}: {e}",
                resource_type="gpu",
                suggestion="Ensure CUDA drivers are correctly installed and the GPU is functioning."
            ) from e
        
        validated['device_id'] = device_id
    
    return validated