#!/usr/bin/env python3
# this_file: src/uubed/api.py
"""High-level API for uubed encoding with comprehensive validation and error handling."""

from typing import Union, List, Literal, Optional, Any, Dict, Tuple
import numpy as np
from .native_wrapper import (
    q64_encode_native,
    q64_decode_native,
    simhash_q64_native,
    top_k_q64_native,
    z_order_q64_native,
    mq64_encode_native,
    mq64_decode_native,
    is_native_available,
)
from .exceptions import (
    UubedValidationError,
    UubedEncodingError,
    UubedDecodingError,
    encoding_error,
    validation_error
)
from .validation import (
    validate_encoding_method,
    validate_embedding_input,
    validate_method_parameters,
    estimate_memory_usage,
    validate_memory_usage
)
from .config import get_config

EncodingMethod = Literal["eq64", "shq64", "t8q64", "zoq64", "mq64", "auto"]


def encode(
    embedding: Union[List[int], np.ndarray, bytes],
    method: EncodingMethod = "auto",
    **kwargs: Any
) -> str:
    """
    Encodes an embedding vector into a uubed string using the specified method.

    This function provides a high-level interface for encoding, handling input
    validation, method selection (including auto-detection), parameter validation,
    and dispatching to optimized native implementations.

    Args:
        embedding: The input vector to encode. Supported types include:
                   - `List[int]`: A list of integers, typically representing byte values (0-255).
                   - `np.ndarray`: A NumPy array of `uint8` (0-255) or `float` (normalized [0,1] or [0,255]).
                   - `bytes`: A raw byte sequence.
        method: The encoding method to use. Defaults to "auto".
                - "eq64": Lossless encoding, preserving full precision. Ideal for exact reconstruction.
                - "shq64": Similarity Hash QuadB64. A lossy, compact encoding suitable for approximate nearest neighbor search.
                - "t8q64": Top-K QuadB64. A lossy, very compact encoding that retains the `k` most significant values.
                - "zoq64": Z-Order QuadB64. A lossy, compact spatial encoding, useful for preserving locality.
                - "mq64": Matryoshka QuadB64. A hierarchical, lossy encoding for multi-resolution representations.
                - "auto": Automatically selects the most appropriate method based on the embedding's characteristics
                          (e.g., size, sparsity) and configured defaults.
        **kwargs: Method-specific parameters that override any default configurations.
            - `planes` (int, optional): For "shq64", specifies the number of hash planes.
                                        Must be a multiple of 8. Default is 64.
            - `k` (int, optional): For "t8q64", specifies the number of top indices to retain. Default is 8.
            - `levels` (List[int], optional): For "mq64", defines the hierarchical levels for encoding.
                                              Each integer represents the number of dimensions for that level.

    Returns:
        str: The encoded string in the position-safe q64 format.

    Raises:
        UubedValidationError: If any input parameter is invalid (e.g., incorrect embedding format,
                              unsupported method, out-of-range method-specific parameters).
        UubedEncodingError: If an error occurs during the underlying native encoding process.
        UubedResourceError: If the estimated memory usage for the encoding operation is excessive,
                            preventing potential system instability.

    Example:
        >>> import numpy as np
        >>> from uubed.api import encode, decode

        >>> # Example 1: Encoding a uint8 NumPy array with lossless eq64
        >>> embedding_uint8 = np.random.randint(0, 256, 128, dtype=np.uint8)
        >>> encoded_eq64 = encode(embedding_uint8, method="eq64")
        >>> print(f"Encoded (eq64): {encoded_eq64[:50]}...")

        >>> # Example 2: Encoding a float NumPy array (normalized) with similarity hash (shq64)
        >>> embedding_float = np.random.random(256)
        >>> encoded_shq64 = encode(embedding_float, method="shq64", planes=128)
        >>> print(f"Encoded (shq64): {encoded_shq64[:50]}...")

        >>> # Example 3: Encoding a list of integers with Top-K (t8q64)
        >>> embedding_list = [i % 256 for i in range(512)]
        >>> encoded_t8q64 = encode(embedding_list, method="t8q64", k=32)
        >>> print(f"Encoded (t8q64): {encoded_t8q64[:50]}...")

        >>> # Example 4: Automatic method selection
        >>> embedding_auto = np.random.randint(0, 256, 64, dtype=np.uint8)
        >>> encoded_auto = encode(embedding_auto, method="auto")
        >>> print(f"Encoded (auto): {encoded_auto[:50]}...")

        >>> # Example 5: Encoding with Matryoshka (mq64)
        >>> embedding_mq64 = np.random.randint(0, 256, 1024, dtype=np.uint8)
        >>> encoded_mq64 = encode(embedding_mq64, method="mq64", levels=[256, 64, 16])
        >>> print(f"Encoded (mq64): {encoded_mq64[:50]}...")
    """
    try:
        # Step 1: Validate and normalize the requested encoding method.
        # This ensures the 'method' argument is a recognized EncodingMethod literal.
        method = validate_encoding_method(method)

        # Step 2: Handle 'auto' method selection based on configuration.
        # If 'auto' is specified, check if a default method is configured globally.
        if method == "auto":
            config = get_config()
            default_method = config.get("encoding.default_method", "auto")
            if default_method != "auto":
                # If a default is found, use it and re-validate to ensure it's a valid method.
                method = validate_encoding_method(default_method)

        # Step 3: Validate and normalize the input embedding.
        # This crucial step converts various input formats (list, bytes, float array)
        # into a standardized `np.ndarray` of `uint8` type, suitable for native processing.
        embedding_array: np.ndarray = validate_embedding_input(embedding, method)

        # Step 4: Merge configuration defaults with provided keyword arguments.
        # Keyword arguments (`kwargs`) always take precedence over values loaded from the configuration.
        config = get_config()
        if method != "auto": # Apply config parameters only if a specific method is chosen
            config_params: Dict[str, Any] = config.get_encoding_params(method)
            merged_kwargs: Dict[str, Any] = {**config_params, **kwargs}
        else:
            # If method is still "auto" (no global default), only use explicitly provided kwargs.
            merged_kwargs = kwargs

        # Step 5: Validate method-specific parameters.
        # This ensures parameters like 'k' for t8q64 or 'planes' for shq64 are valid for the chosen method.
        validated_params: Dict[str, Any] = validate_method_parameters(method, **merged_kwargs)

        # Step 6: Final dynamic auto-selection if method is still "auto".
        # If no default was configured and "auto" is still active, determine the best method
        # based on the characteristics of the `embedding_array`.
        if method == "auto":
            method = _auto_select_method(embedding_array)

        # Step 7: Perform special validation for 'k' parameter in 't8q64' method.
        # The 'k' value (number of top indices) must be less than the total size of the embedding.
        if method == "t8q64":
            k: int = validated_params.get('k', 8)
            if k >= embedding_array.size:
                raise validation_error(
                    f"k parameter ({k}) must be smaller than embedding size ({embedding_array.size})",
                    parameter="k",
                    expected=f"< {embedding_array.size}",
                    received=f"{k}"
                )

        # Step 8: Estimate and validate memory usage.
        # This proactive check prevents potential `MemoryError` for very large embeddings
        # by ensuring the encoding operation won't consume excessive system resources.
        memory_estimate: int = estimate_memory_usage(1, embedding_array.size, method)
        validate_memory_usage(memory_estimate, f"encoding with {method}")

        # Step 9: Convert the validated NumPy array to bytes.
        # Native functions typically operate on raw byte sequences for efficiency.
        embedding_bytes: bytes = embedding_array.tobytes()

        # Step 10: Dispatch the encoding task to the appropriate native encoder.
        # Each native call is wrapped in a `try-except` block to catch and
        # re-raise any underlying native errors as `UubedEncodingError`,
        # providing consistent and informative error messages.
        try:
            if method == "eq64":
                return q64_encode_native(embedding_bytes)
            elif method == "shq64":
                planes: int = validated_params.get("planes", 64)
                return simhash_q64_native(embedding_bytes, planes=planes)
            elif method == "t8q64":
                k: int = validated_params.get("k", 8)
                return top_k_q64_native(embedding_bytes, k=k)
            elif method == "zoq64":
                return z_order_q64_native(embedding_bytes)
            elif method == "mq64":
                # mq64 encoding supports an optional 'levels' parameter for hierarchical encoding.
                levels: Optional[List[int]] = validated_params.get("levels", None)
                return mq64_encode_native(embedding_bytes, levels)
            else:
                # This case should ideally be unreachable due to prior validation of 'method'.
                raise encoding_error(
                    f"Unsupported encoding method: {method}",
                    method=method,
                    suggestion="Ensure a valid encoding method is selected or configured."
                )
        except Exception as e:
            # Catch any exception originating from native calls and re-raise it
            # as a `UubedEncodingError` for standardized error handling.
            raise encoding_error(
                f"Native encoding failed: {str(e)}",
                method=method,
                suggestion="Check if the native extension is properly installed and compatible with your system."
            ) from e

    except UubedValidationError:
        # Re-raise `UubedValidationError` directly as they are specific and informative.
        raise
    except UubedEncodingError:
        # Re-raise `UubedEncodingError` directly.
        raise
    except Exception as e:
        # Catch any other unexpected exceptions during the encoding process
        # and wrap them in a `UubedEncodingError` for consistent error reporting.
        # Safely access 'method' if it was defined before the exception occurred.
        raise encoding_error(
            f"An unexpected error occurred during encoding: {str(e)}",
            method=method if 'method' in locals() else "unknown",
            suggestion="Please report this issue with the full traceback to the developers."
        ) from e


def _auto_select_method(embedding: np.ndarray) -> str:
    """
    Automatically selects the most appropriate encoding method based on the embedding's characteristics.

    This internal helper function applies a heuristic to determine the optimal encoding method
    when "auto" is specified. The selection is primarily driven by the embedding's size and
    its sparsity (proportion of non-zero elements), aiming to balance compression efficiency
    with information preservation.

    Args:
        embedding: A validated NumPy array (`np.ndarray`) representing the embedding.
                   It is expected to be of `uint8` dtype, as produced by `validate_embedding_input`.

    Returns:
        str: The name of the selected encoding method (e.g., "eq64", "shq64", "t8q64").
             The returned method is guaranteed to be one of the supported `EncodingMethod` literals.

    Notes:
        - For very small embeddings (size <= 16), "shq64" (Similarity Hash) is often preferred for its compactness.
        - For small to medium embeddings (size <= 64), sparsity plays a role: "t8q64" (Top-K) is chosen for sparse
          embeddings, while "shq64" is used for denser ones.
        - For medium embeddings (size <= 256), a more nuanced approach is taken, considering very sparse embeddings
          for "t8q64", common dimensions (128, 256) for "shq64", and falling back to lossless "eq64" otherwise.
        - For large embeddings (size > 256), "eq64" (lossless) is generally prioritized to ensure full information
          retention, unless a specific lossy method is explicitly requested by the user.
        - The sparsity thresholds (e.g., 0.3, 0.2) are heuristic and can be fine-tuned based on empirical performance
          and typical embedding characteristics.
    """
    size: int = embedding.size

    # Calculate sparsity: the proportion of non-zero elements in the embedding.
    # This helps in deciding if a sparse-friendly encoding method like t8q64 is suitable.
    if size > 0: # Prevent division by zero for empty embeddings, though validation should ideally prevent this state.
        sparsity: float = np.count_nonzero(embedding) / size
    else:
        # For an empty embedding, sparsity is effectively 0.0, meaning no non-zero elements.
        sparsity = 0.0

    # Apply heuristic rules for method selection based on embedding size and sparsity.
    if size <= 16:
        # For very small embeddings, SimHash (shq64) offers good compactness.
        return "shq64"
    elif size <= 64:
        # For small embeddings, consider sparsity to choose between Top-K and SimHash.
        if sparsity < 0.3:  # If significantly sparse, Top-K is efficient.
            return "t8q64"
        else:
            # Otherwise, SimHash provides general compression.
            return "shq64"
    elif size <= 256:
        # For medium embeddings, a more detailed decision based on sparsity and common sizes.
        if sparsity < 0.2:  # If very sparse, Top-K is still a strong candidate.
            return "t8q64"
        elif size in [128, 256]:  # For common embedding dimensions, SimHash is often a good default.
            return "shq64"
        else:
            # Fallback to lossless encoding for other medium-sized, non-sparse embeddings.
            return "eq64"
    else:
        # For large embeddings, prioritize lossless encoding to preserve all information.
        return "eq64"


def decode(encoded: str, method: Optional[EncodingMethod] = None) -> bytes:
    """
    Decodes an encoded string back to its original bytes representation.

    Note that only lossless encoding methods (currently "eq64" and "mq64") can be
    fully decoded back to the original bytes. Lossy compression methods
    ("shq64", "t8q64", "zoq64") cannot be accurately decoded to their original form.

    Args:
        encoded: The encoded string in q64 format.
        method: The encoding method that was used to create the `encoded` string.
                If `None`, the method will be auto-detected (currently only "eq64"
                can be reliably auto-detected by the presence of dots).

    Returns:
        bytes: The original byte representation of the embedding.

    Raises:
        UubedValidationError: If the `encoded` input is not a valid string or is empty.
        UubedDecodingError: If the decoding operation fails (e.g., invalid encoded string,
                            or attempting to decode a lossy method).
        NotImplementedError: (Potentially, if a method is specified but its decoder is not implemented).

    Example:
        >>> import numpy as np
        >>> embedding = np.random.randint(0, 256, 64, dtype=np.uint8)
        >>> encoded = encode(embedding, method="eq64")
        >>> decoded_bytes = decode(encoded, method="eq64")
        >>> assert bytes(embedding) == decoded_bytes
    """
    try:
        # Validate that the input 'encoded' is a non-empty string.
        if not isinstance(encoded, str):
            raise validation_error(
                "Encoded input must be a string",
                parameter="encoded",
                expected="string in q64 format",
                received=f"{type(encoded).__name__}"
            )

        if not encoded.strip():
            raise validation_error(
                "Encoded string cannot be empty or whitespace-only",
                parameter="encoded",
                expected="non-empty string in q64 format",
                received="empty or whitespace-only string"
            )

        # Auto-detect the encoding method if not explicitly provided.
        # Currently, only "eq64" has a distinct pattern (dots) for auto-detection.
        if method is None:
            method = _auto_detect_method(encoded)
        else:
            # If a method is provided, validate it against known encoding methods.
            method = validate_encoding_method(method)

        # Check if the detected or provided method supports decoding.
        # Only "eq64" and "mq64" are designed for full round-trip decoding.
        if method not in ("eq64", "mq64"):
            raise UubedDecodingError(
                f"Decoding not supported for '{method}' method.",
                method=method,
                suggestion=f"'{method}' is a lossy compression method. Only 'eq64' and 'mq64' support full decoding back to original bytes."
            )

        # Dispatch the decoding task to the appropriate native decoder.
        # Similar to encoding, native calls are wrapped for robust error handling.
        try:
            decoded_result: Union[bytes, List[int]] # Type hint for decoded_result
            if method == "mq64":
                decoded_result = mq64_decode_native(encoded)
            else: # Assumed to be eq64 if not mq64 and passed previous check
                decoded_result = q64_decode_native(encoded)
            return bytes(decoded_result) # Ensure the result is always bytes

        except Exception as e:
            # Catch any exception from native calls and re-raise as a specific decoding error.
            raise UubedDecodingError(
                f"Native decoding failed: {str(e)}",
                method=method,
                encoded_string=encoded,
                suggestion="Check if the encoded string is valid and matches the specified method. "
                           "Ensure the native extension is properly installed and compatible."
            ) from e

    except UubedValidationError:
        # Re-raise validation errors directly as they are specific and informative.
        raise
    except UubedDecodingError:
        # Re-raise decoding errors directly.
        raise
    except NotImplementedError:
        # Re-raise NotImplementedError if encountered (e.g., for a method not yet fully supported).
        raise
    except Exception as e:
        # Catch any other unexpected exceptions during the decoding process
        # and wrap them in a UubedDecodingError for consistent error handling.
        raise UubedDecodingError(
            f"An unexpected error occurred during decoding: {str(e)}",
            method=method if 'method' in locals() else "unknown", # Safely access 'method'
            encoded_string=encoded if 'encoded' in locals() else "unknown", # Safely access 'encoded'
            suggestion="Please report this issue with the full traceback."
        ) from e


def _auto_detect_method(encoded: str) -> str:
    """
    Attempts to auto-detect the encoding method from the structure of the encoded string.

    Currently, only "eq64" can be reliably auto-detected due to its unique
    characteristic of containing dot separators for position encoding.
    Other methods ("shq64", "t8q64", "zoq64", "mq64") produce similar-looking
    q64 strings without dots, making unambiguous auto-detection difficult
    without additional metadata.

    Args:
        encoded: The encoded string to analyze.

    Returns:
        str: The name of the detected method (e.g., "eq64").

    Raises:
        UubedDecodingError: If the method cannot be reliably auto-detected from the string pattern.
    """
    # "eq64" strings are characterized by the presence of dot separators.
    if "." in encoded:
        return "eq64"

    # For other encoding methods, the string format is not distinct enough
    # to reliably determine the original method without explicit metadata.
    raise UubedDecodingError(
        "Cannot auto-detect encoding method from string pattern.",
        encoded_string=encoded,
        suggestion="Please specify the 'method' parameter explicitly when decoding. "
                   "Auto-detection currently only works for 'eq64' method (which includes dots)."
    )
