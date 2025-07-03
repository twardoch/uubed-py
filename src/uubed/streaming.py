#!/usr/bin/env python3
# this_file: src/uubed/streaming.py
"""Streaming API for encoding large datasets efficiently with comprehensive validation."""

from typing import Iterator, Union, List, Optional, BinaryIO, Iterable, Any, Dict
import numpy as np
from pathlib import Path

from .api import encode, decode, EncodingMethod
from .exceptions import (
    UubedValidationError,
    UubedResourceError,
    validation_error,
    resource_error
)
from .validation import (
    validate_encoding_method,
    validate_batch_parameters,
    validate_file_path,
    validate_memory_usage,
    estimate_memory_usage
)


def encode_stream(
    embeddings: Iterable[Union[bytes, List[int], np.ndarray]],
    method: EncodingMethod = "auto",
    batch_size: int = 1000,
    **kwargs: Any
) -> Iterator[str]:
    """
    Encode a stream of embeddings lazily with comprehensive validation.
    
    This generator yields encoded strings one at a time, allowing
    efficient processing of large datasets without loading everything
    into memory. It maintains constant memory usage regardless of the total
    dataset size by processing data in small batches.
    
    Args:
        embeddings: An iterable (e.g., list, generator) of embedding vectors.
                    Each embedding can be bytes, a list of integers, or a NumPy array.
        method: The encoding method to use. Defaults to "auto" for automatic selection.
        batch_size: The number of embeddings to process in a single batch.
                    Optimizing this value can balance memory usage and processing efficiency.
                    (Default: 1000, valid range: 1-100000).
        **kwargs: Method-specific parameters passed directly to the `encode` function
                  (e.g., `k` for t8q64, `planes` for shq64).
        
    Yields:
        str: An encoded string for each input embedding.
        
    Raises:
        UubedValidationError: If any input parameters are invalid (e.g., `embeddings` is not iterable,
                              `batch_size` is out of range, or an individual embedding is malformed).
        UubedResourceError: If memory usage would be excessive for the estimated batch size,
                            or if an unexpected system resource error occurs during streaming.
        
    Example:
        >>> import numpy as np
        >>> 
        >>> # Process embeddings from a generator
        >>> def embedding_generator():
        ...     for i in range(100000):
        ...         yield np.random.randint(0, 256, 768, dtype=np.uint8)
        ...
        >>> # Memory-efficient streaming processing
        >>> count = 0
        >>> for encoded in encode_stream(embedding_generator(), method="shq64", batch_size=500):
        ...     count += 1
        ...     if count % 1000 == 0:
        ...         print(f"Processed {count} embeddings")
        ...
        >>> # Process existing array without loading into memory
        >>> embeddings = [np.random.randint(0, 256, 256, dtype=np.uint8) for _ in range(1000)]
        >>> encoded_list = list(encode_stream(embeddings, method="t8q64", k=16))
    """
    try:
        # Validate the encoding method and batch size parameters.
        method = validate_encoding_method(method)
        validated_params = validate_batch_parameters(batch_size=batch_size)
        batch_size = validated_params["batch_size"]
        
        # Ensure the 'embeddings' input is not None, as it must be an iterable.
        if embeddings is None:
            raise validation_error(
                "Embeddings iterable cannot be None",
                parameter="embeddings",
                expected="iterable of embeddings",
                received="None"
            )
        
        # Initialize a list to hold the current batch of embeddings.
        batch: List[Union[bytes, List[int], np.ndarray]] = []
        total_processed: int = 0 # Keep track of total embeddings processed for error context.
        
        try:
            # Iterate through the provided embeddings iterable.
            for embedding in embeddings:
                batch.append(embedding)
                
                # When the batch reaches the specified size, process it.
                if len(batch) >= batch_size:
                    # Process each embedding in the current batch.
                    for emb in batch:
                        try:
                            # Encode the individual embedding. The `encode` function handles its own validation.
                            yield encode(emb, method=method, **kwargs)
                            total_processed += 1
                        except Exception as e:
                            # Catch any encoding errors and re-raise with context about the failed embedding.
                            # Note: processed_count is incremented before encode, so it's the *next* embedding number.
                            raise UubedEncodingError(
                                f"Failed to encode embedding #{total_processed}: {str(e)}",
                                suggestion="Check that all embeddings in the stream are valid and conform to expected formats."
                            ) from e
                    batch = [] # Clear the batch after processing.
            
            # After the loop, process any remaining embeddings in the last (possibly incomplete) batch.
            for emb in batch:
                try:
                    yield encode(emb, method=method, **kwargs)
                    total_processed += 1
                except Exception as e:
                    # Catch any encoding errors for remaining embeddings and re-raise with context.
                    raise UubedEncodingError(
                        f"Failed to encode embedding #{total_processed}: {str(e)}",
                        suggestion="Check that all embeddings in the stream are valid and conform to expected formats."
                    ) from e
                    
        except TypeError as e:
            # Specifically catch TypeError if 'embeddings' is not iterable, providing a clearer error message.
            raise validation_error(
                f"Embeddings parameter is not iterable: {str(e)}",
                parameter="embeddings",
                expected="iterable of embeddings (list, generator, etc.)",
                received=f"{type(embeddings).__name__}"
            ) from e
            
    except UubedValidationError:
        # Re-raise validation errors directly as they are specific and informative.
        raise
    except Exception as e:
        # Catch any other unexpected errors during the streaming process
        # and wrap them in a UubedResourceError for consistent error handling.
        raise UubedResourceError(
            f"An unexpected error occurred during streaming encoding: {str(e)}",
            resource_type="stream",
            suggestion="Check input data format, available memory, and ensure the underlying encoder is functioning correctly."
        ) from e


def encode_file_stream(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    method: EncodingMethod = "auto",
    embedding_size: int = 768,
    **kwargs: Any
) -> Iterator[str]:
    """
    Encode embeddings from a binary file in a streaming fashion.
    
    This function reads embeddings from a binary file in chunks, processes them,
    and optionally writes the encoded results to an output file. It is designed
    to handle large files efficiently by maintaining constant memory usage.
    
    Args:
        input_path: The path to the binary file containing embeddings.
                    The file must exist and be readable.
        output_path: An optional path to write the encoded results. If provided,
                     the parent directory must exist and be writable.
        method: The encoding method to use. Defaults to "auto".
        embedding_size: The size of each embedding in bytes. This is crucial for
                        correctly parsing the binary file. Must be a positive integer
                        (typically between 128 and 1536 bytes for common embedding models).
        **kwargs: Method-specific parameters passed to the `encode` function.
        
    Yields:
        str: An encoded string for each embedding read from the input file.
        
    Raises:
        UubedValidationError: If input parameters are invalid (e.g., `embedding_size` is non-positive,
                              or file paths are malformed/inaccessible).
        UubedResourceError: If file access fails (e.g., file not found, permissions issue),
                            if the file size is not a multiple of `embedding_size`,
                            or if an incomplete embedding is read.
        
    Example:
        >>> # Encode embeddings from binary file
        >>> for encoded in encode_file_stream("embeddings.bin", 
        ...                                   method="shq64", 
        ...                                   embedding_size=768):
        ...     print(f"Encoded: {encoded[:50]}...")
        ...
        >>> # Encode and save to output file
        >>> results = list(encode_file_stream("input.bin", 
        ...                                   "output.txt",
        ...                                   method="t8q64", 
        ...                                   embedding_size=256,
        ...                                   k=16))
        >>> print(f"Processed {len(results)} embeddings")
    """
    try:
        # Validate the encoding method.
        method = validate_encoding_method(method)
        
        # Validate the embedding_size parameter.
        if not isinstance(embedding_size, int):
            raise validation_error(
                "embedding_size must be an integer",
                parameter="embedding_size",
                expected="positive integer (bytes per embedding)",
                received=f"{type(embedding_size).__name__}"
            )
        
        if embedding_size <= 0:
            raise validation_error(
                "embedding_size must be positive",
                parameter="embedding_size",
                expected="positive integer (typically 128-1536)",
                received=f"{embedding_size}"
            )
        
        if embedding_size > 100000: # Arbitrary upper limit to prevent extreme values.
            raise validation_error(
                "embedding_size too large",
                parameter="embedding_size",
                expected="<= 100000 for reasonable processing",
                received=f"{embedding_size}"
            )
        
        # Validate input and output file paths.
        input_path = validate_file_path(input_path, check_exists=True, check_readable=True)
        
        if output_path is not None:
            output_path = validate_file_path(output_path, check_exists=False, check_writable=True)
        
        # Perform checks on the input file size to ensure it's consistent with embedding_size.
        file_size: int = input_path.stat().st_size
        if file_size % embedding_size != 0:
            remaining_bytes: int = file_size % embedding_size
            raise UubedResourceError(
                f"File size ({file_size} bytes) not divisible by embedding_size ({embedding_size} bytes).",
                resource_type="file",
                available=f"{file_size} bytes",
                required=f"multiple of {embedding_size} bytes",
                suggestion=f"File has {remaining_bytes} extra bytes. Check if embedding_size is correct or if the file is corrupted."
            )
        
        num_embeddings: int = file_size // embedding_size
        if num_embeddings == 0:
            raise resource_error(
                f"File too small to contain any embeddings (file size: {file_size} bytes, expected embedding size: {embedding_size} bytes).",
                resource_type="file",
                available=f"{file_size} bytes",
                required=f">= {embedding_size} bytes"
            )
        
        # Estimate and validate memory usage per embedding to prevent excessive memory allocation.
        memory_estimate: int = estimate_memory_usage(1, embedding_size, method)  # Per embedding
        validate_memory_usage(memory_estimate, f"file streaming with {method}")
        
        # Initialize a counter for processed embeddings.
        processed_count: int = 0
        
        # Open the input file in binary read mode.
        with open(input_path, "rb") as f:
            if output_path:
                # If an output path is provided, open it in write mode.
                with open(output_path, "w") as out:
                    while True:
                        # Read one embedding's worth of bytes from the file.
                        embedding_bytes_read: bytes = f.read(embedding_size)
                        if not embedding_bytes_read:
                            # Break loop if end of file is reached.
                            break
                        
                        # Check if the read chunk has the expected size.
                        if len(embedding_bytes_read) != embedding_size:
                            raise resource_error(
                                f"Incomplete embedding read at position {processed_count + 1}. Expected {embedding_size} bytes, got {len(embedding_bytes_read)} bytes.",
                                resource_type="file",
                                available=f"{len(embedding_bytes_read)} bytes",
                                required=f"{embedding_size} bytes",
                                suggestion="File may be truncated or embedding_size may be incorrect. Ensure the binary file is correctly formed."
                            )
                        
                        try:
                            # Encode the read embedding bytes.
                            encoded: str = encode(embedding_bytes_read, method=method, **kwargs)
                            # Write the encoded string to the output file, followed by a newline.
                            out.write(encoded + "\n")
                            processed_count += 1
                            yield encoded # Yield the encoded string for external consumption.
                        except Exception as e:
                            # Catch and re-raise any encoding errors with context.
                            raise UubedEncodingError(
                                f"Failed to encode embedding #{processed_count} from file: {str(e)}",
                                suggestion="Check embedding data format within the file and the provided encoding parameters."
                            ) from e
            else:
                # If no output path, just yield the encoded strings without writing to a file.
                while True:
                    embedding_bytes_read = f.read(embedding_size)
                    if not embedding_bytes_read:
                        break
                    
                    if len(embedding_bytes_read) != embedding_size:
                        raise resource_error(
                            f"Incomplete embedding read at position {processed_count + 1}. Expected {embedding_size} bytes, got {len(embedding_bytes_read)} bytes.",
                            resource_type="file",
                            available=f"{len(embedding_bytes_read)} bytes",
                            required=f"{embedding_size} bytes",
                            suggestion="File may be truncated or embedding_size may be incorrect. Ensure the binary file is correctly formed."
                        )
                    
                    try:
                        processed_count += 1
                        yield encode(embedding_bytes_read, method=method, **kwargs)
                    except Exception as e:
                        # Catch and re-raise any encoding errors with context.
                        raise UubedValidationError(
                            f"Failed to encode embedding #{processed_count} from file: {str(e)}",
                            suggestion="Check embedding data format within the file and the provided encoding parameters."
                        ) from e
                        
    except UubedValidationError:
        # Re-raise validation errors directly.
        raise
    except UubedResourceError:
        # Re-raise resource errors directly.
        raise
    except Exception as e:
        # Catch any other unexpected errors during file streaming
        # and wrap them in a UubedResourceError for consistent error handling.
        raise UubedResourceError(
            f"An unexpected error occurred during file streaming: {str(e)}",
            resource_type="file",
            suggestion="Check file permissions, file format, and ensure the underlying encoder is functioning correctly."
        ) from e


def decode_stream(
    encoded_strings: Iterable[str],
    method: Optional[EncodingMethod] = None,
) -> Iterator[bytes]:
    """
    Decode a stream of encoded strings back to their original byte representations.
    
    This generator processes encoded strings one by one, yielding the decoded bytes.
    It relies on the `decode` function for the actual decoding logic and error handling.
    
    Args:
        encoded_strings: An iterable of encoded strings (e.g., from a file or another generator).
        method: The encoding method that was used for encoding. If `None`, the method
                will be auto-detected by the `decode` function (currently only "eq64"
                can be reliably auto-detected).
        
    Yields:
        bytes: The decoded byte representation for each input string.
        
    Note:
        Only "eq64" and "mq64" encoding methods support full lossless decoding.
        Other methods ("shq64", "t8q64", "zoq64") are lossy compressions and cannot
        be accurately decoded back to their original bytes. Attempting to decode
        lossy methods will result in a `UubedDecodingError`.
    """
    # Iterate through each encoded string in the input stream.
    for encoded in encoded_strings:
        # Call the main `decode` function for each string.
        # .strip() is used to remove any leading/trailing whitespace, including newlines,
        # which might be present if reading from a file line by line.
        yield decode(encoded.strip(), method=method)


def batch_encode(
    embeddings: List[Union[bytes, List[int], np.ndarray]],
    method: EncodingMethod = "auto",
    n_workers: Optional[int] = None,
    **kwargs: Any
) -> List[str]:
    """
    Encode a list of embedding vectors in a batch with comprehensive validation and error handling.
    
    This function processes a list of embeddings and returns a list of their encoded string
    representations. It includes checks for input validity, memory usage, and provides
    detailed error messages for failed encodings.
    
    Args:
        embeddings: A list of embedding vectors to encode. Each embedding can be bytes,
                    a list of integers, or a NumPy array. The list must not be empty.
        method: The encoding method to use. Defaults to "auto".
        n_workers: The number of parallel workers to use. Currently, this parameter is
                   ignored, and processing is sequential for reliability and to avoid
                   GIL-related issues with native extensions. Parallel processing will
                   be enabled when the native extension supports releasing the GIL.
        **kwargs: Method-specific parameters passed to the `encode` function.
        
    Returns:
        List[str]: A list of encoded strings, where each string corresponds to an input embedding.
        
    Raises:
        UubedValidationError: If input parameters are invalid (e.g., `embeddings` is not a list,
                              is empty, or contains malformed embeddings).
        UubedResourceError: If estimated memory usage for the batch would be excessive,
                            or if an unexpected system resource error occurs.
        
    Note:
        Current implementation processes embeddings sequentially. Future versions may
        leverage parallel processing when native extensions provide GIL-release capabilities.
        
    Example:
        >>> import numpy as np
        >>> embeddings = [np.random.randint(0, 256, 128, dtype=np.uint8) for _ in range(100)]
        >>> encoded = batch_encode(embeddings, method="shq64", planes=64)
        >>> print(f"Encoded {len(encoded)} embeddings")
    """
    try:
        # Validate the encoding method.
        method = validate_encoding_method(method)
        
        # Validate that 'embeddings' is a non-empty list.
        if not isinstance(embeddings, list):
            raise validation_error(
                "embeddings must be a list",
                parameter="embeddings",
                expected="list of embeddings",
                received=f"{type(embeddings).__name__}"
            )
        
        if len(embeddings) == 0:
            raise validation_error(
                "embeddings list cannot be empty",
                parameter="embeddings",
                expected="non-empty list of embeddings",
                received="empty list"
            )
        
        # Validate the 'n_workers' parameter, though it's currently unused.
        if n_workers is not None:
            if not isinstance(n_workers, int):
                raise validation_error(
                    "n_workers must be an integer",
                    parameter="n_workers",
                    expected="positive integer or None",
                    received=f"{type(n_workers).__name__}"
                )
            if n_workers <= 0:
                raise validation_error(
                    "n_workers must be positive",
                    parameter="n_workers",
                    expected="positive integer",
                    received=f"{n_workers}"
                )
        
        # Estimate and validate memory usage for the entire batch.
        # This assumes all embeddings in the batch are of similar size to the first one.
        if embeddings:
            first_emb: Union[bytes, List[int], np.ndarray] = embeddings[0]
            emb_size: int
            if isinstance(first_emb, np.ndarray):
                emb_size = first_emb.size
            elif isinstance(first_emb, (list, tuple)):
                emb_size = len(first_emb)
            elif isinstance(first_emb, bytes):
                emb_size = len(first_emb)
            else:
                # Fallback for unexpected types, a conservative estimate.
                emb_size = 256  
            
            memory_estimate: int = estimate_memory_usage(len(embeddings), emb_size, method)
            validate_memory_usage(memory_estimate, f"batch encoding {len(embeddings)} embeddings")
        
        # Process each embedding in the list sequentially.
        results: List[str] = []
        for i, emb in enumerate(embeddings):
            try:
                # Encode the individual embedding. The `encode` function handles its own validation.
                encoded: str = encode(emb, method=method, **kwargs)
                results.append(encoded)
            except Exception as e:
                # Catch any encoding errors and re-raise with context about the failed embedding.
                raise UubedEncodingError(
                    f"Failed to encode embedding #{i} of {len(embeddings)}: {str(e)}",
                    suggestion="Check that all embeddings in the batch are valid and uniform in format."
                ) from e
        
        return results
        
    except UubedValidationError:
        # Re-raise validation errors directly.
        raise
    except UubedResourceError:
        # Re-raise resource errors directly.
        raise
    except Exception as e:
        # Catch any other unexpected errors during batch encoding
        # and wrap them in a UubedResourceError for consistent error handling.
        raise UubedResourceError(
            f"An unexpected error occurred during batch encoding: {str(e)}",
            resource_type="memory", # Assuming memory is the most common resource issue for batches.
            suggestion="Check input data format, available memory, and ensure the underlying encoder is functioning correctly."
        ) from e


class StreamingEncoder:
    """
    Context manager for streaming encoding operations.
    
    This class provides a convenient and resource-safe way to encode multiple
    embeddings, optionally writing the results to a file. It ensures proper
    resource management (e.g., file closing) even if errors occur.
    
    Example:
        >>> import numpy as np
        >>> 
        >>> # Encode to file with automatic resource cleanup
        >>> with StreamingEncoder("output.txt", method="shq64", planes=128) as encoder:
        ...     for i in range(1000):
        ...         embedding = np.random.randint(0, 256, 256, dtype=np.uint8)
        ...         encoded = encoder.encode(embedding)
        ...         if i % 100 == 0:
        ...             print(f"Processed {encoder.count} embeddings")
        ...
        >>> # Encode without file output
        >>> with StreamingEncoder(method="t8q64", k=16) as encoder:
        ...     embedding = np.random.randint(0, 256, 512, dtype=np.uint8)
        ...     result = encoder.encode(embedding)
        ...     print(f"Result: {result[:50]}...")
    """
    
    def __init__(
        self,
        output_path: Optional[Union[str, Path]] = None,
        method: EncodingMethod = "auto",
        **kwargs: Any
    ):
        """
        Initializes the StreamingEncoder instance.

        This constructor sets up the encoder with a specified output path (optional),
        encoding method, and any method-specific keyword arguments. It performs
        initial validation of the provided parameters.

        Args:
            output_path: An optional file path (`str` or `Path` object) where the
                         encoded results will be written. If `None`, results are not
                         persisted to a file.
            method: The encoding method to use. Defaults to "auto" for automatic selection.
                    Refer to `uubed.api.encode` for supported methods.
            **kwargs: Arbitrary keyword arguments that are passed directly to the
                      underlying `uubed.api.encode` function. These typically include
                      method-specific parameters like `k` for "t8q64" or `planes` for "shq64".

        Raises:
            UubedValidationError: If any of the input parameters are invalid (e.g., an unknown
                                  `method`, or an `output_path` that is not writable).
        """
        try:
            # Validate the encoding method to ensure it's a recognized type.
            self.method: EncodingMethod = validate_encoding_method(method)

            # Validate and store the output file path. If provided, ensure it's a valid
            # path and that the application has write permissions to it.
            if output_path is not None:
                self.output_path: Optional[Path] = validate_file_path(
                    output_path,
                    check_exists=False,  # We don't expect the file to exist yet
                    check_writable=True  # But we must be able to write to its directory
                )
            else:
                self.output_path = None

            # Store any additional keyword arguments. These will be passed to the `encode` function.
            self.kwargs: Dict[str, Any] = kwargs

            # Initialize internal state variables.
            self.output_file: Optional[BinaryIO] = None  # File handle for writing, initialized in __enter__.
            self.encoded_count: int = 0  # Counter for successfully encoded embeddings.
            self._is_open: bool = False  # Internal flag to track if the encoder is active within a context.

        except UubedValidationError:
            # Re-raise validation errors directly as they are specific and informative.
            raise
        except Exception as e:
            # Catch any other unexpected exceptions during the initialization process
            # and wrap them in a UubedValidationError for consistent error handling.
            raise UubedValidationError(
                f"Failed to initialize StreamingEncoder: {str(e)}",
                suggestion="Check output path permissions, encoding parameters, and ensure valid inputs."
            ) from e

    def __enter__(self) -> "StreamingEncoder":
        """
        Enters the runtime context related to this object.

        This method is automatically called when the `StreamingEncoder` instance
        is used in a `with` statement. It opens the specified output file for writing
        if `output_path` was provided during initialization.

        Returns:
            StreamingEncoder: The instance of the StreamingEncoder itself, allowing
                              it to be assigned to a variable in the `with` statement.

        Raises:
            UubedResourceError: If the output file cannot be opened for writing due to
                                 permissions, disk space, or other file system issues.
        """
        try:
            if self.output_path:
                # Open the file in write mode. 'w' mode creates the file if it doesn't exist
                # or truncates it if it does. This is suitable for writing encoded strings.
                self.output_file = open(self.output_path, "w")
            self._is_open = True  # Mark the encoder as active and ready for use.
            return self
        except Exception as e:
            # Catch any exception that occurs during file opening and re-raise it
            # as a UubedResourceError, providing context about the failure.
            raise resource_error(
                f"Cannot open output file '{self.output_path}': {str(e)}",
                resource_type="file",
                suggestion="Check file permissions, ensure the path is valid, and verify disk space."
            ) from e

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """
        Exits the runtime context related to this object.

        This method is automatically called when exiting a `with` statement,
        regardless of whether an exception occurred within the block. Its primary
        responsibility is to ensure that the output file (if opened) is properly
        closed, releasing system resources.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      `None` if the context was exited normally.
            exc_val: The exception instance that caused the context to be exited.
                     `None` if the context was exited normally.
            exc_tb: The traceback object associated with the exception.
                    `None` if the context was exited normally.
        """
        self._is_open = False  # Mark the encoder as no longer active.
        if self.output_file:
            try:
                self.output_file.close()  # Attempt to gracefully close the file.
            except Exception:
                # Suppress any exceptions that occur during file closing to avoid
                # masking the original exception (if any) from the `with` block.
                pass
            finally:
                self.output_file = None  # Explicitly clear the file handle.

    def encode(self, embedding: Union[bytes, List[int], np.ndarray]) -> str:
        """
        Encodes a single embedding using the configured method.

        This method is the primary way to encode individual embeddings when using
        the `StreamingEncoder` within a `with` statement. If an `output_path` was
        specified during initialization, the resulting encoded string is also
        written to that file, followed by a newline character.

        Args:
            embedding: The embedding vector to encode. Can be a raw `bytes` sequence,
                       a `List[int]` (representing byte values), or a `np.ndarray`.

        Returns:
            str: The encoded string representation of the input embedding.

        Raises:
            UubedValidationError: If the `StreamingEncoder` instance is not currently
                                  active (i.e., not used within a `with` statement),
                                  or if the provided `embedding` itself is invalid.
            UubedResourceError: If an error occurs while writing the encoded string
                                to the output file (e.g., disk full, permissions issue).
            UubedEncodingError: If the underlying `uubed.api.encode` function fails
                                for any reason.
        """
        # Ensure the encoder is currently active and being used within a 'with' block.
        if not self._is_open:
            raise UubedValidationError(
                "StreamingEncoder is not open. It must be used within a 'with' statement.",
                suggestion="Use 'with StreamingEncoder(...) as encoder:' to properly initialize and use the encoder."
            )

        try:
            # Call the top-level `encode` function to perform the actual encoding.
            # This function handles its own input validation and error propagation.
            encoded: str = encode(embedding, method=self.method, **self.kwargs)

            # If an output file was configured, write the encoded string to it.
            if self.output_file:
                try:
                    self.output_file.write(encoded + "\n")
                    # Flush the buffer to ensure data is immediately written to disk.
                    # This is important for real-time streaming and error recovery.
                    self.output_file.flush()
                except Exception as e:
                    # Catch any file writing errors and re-raise them as a resource error.
                    raise resource_error(
                        f"Failed to write encoded data to output file: {str(e)}",
                        resource_type="file",
                        suggestion="Check disk space, file permissions, and ensure the output path is valid."
                    ) from e

            # Increment the counter for successfully encoded embeddings.
            # This counter reflects the number of embeddings for which encoding and writing (if applicable) succeeded.
            self.encoded_count += 1
            return encoded

        except UubedValidationError:
            # Re-raise `UubedValidationError` directly as they are specific and informative.
            raise
        except UubedResourceError:
            # Re-raise `UubedResourceError` directly.
            raise
        except Exception as e:
            # Catch any other unexpected exceptions during the encoding process within the context manager
            # and wrap them in a `UubedEncodingError` for consistent error reporting.
            # The `self.encoded_count` here refers to the count *before* the current embedding was attempted.
            # So, `self.encoded_count` is the index of the *next* embedding to be processed.
            # Therefore, `self.encoded_count` is the correct index for the currently failing embedding.
            raise UubedEncodingError(
                f"An unexpected error occurred while encoding embedding #{self.encoded_count}: {str(e)}",
                method=self.method,
                suggestion="Check the format of the embedding and the encoding parameters. Report this issue if it persists."
            ) from e

    @property
    def count(self) -> int:
        """
        Returns the total number of embeddings successfully encoded by this instance.

        This property provides a convenient way to track the progress of the streaming
        encoding operation.

        Returns:
            int: The total count of embeddings that have been successfully encoded
                 and, if applicable, written to the output file.
        """
        return self.encoded_count

    @property
    def is_open(self) -> bool:
        """
        Checks if the StreamingEncoder is currently active and ready for use.

        This property is useful for determining if the encoder is within a `with` block
        and thus capable of processing embeddings.

        Returns:
            bool: `True` if the encoder is open (i.e., `__enter__` has been called
                  and `__exit__` has not yet been called), `False` otherwise.
        """
        return self._is_open

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves a dictionary of operational statistics for the StreamingEncoder instance.

        This method provides insights into the current state and performance of the encoder.

        Returns:
            Dict[str, Any]: A dictionary containing the following key-value pairs:
                            - `encoded_count` (int): The number of embeddings processed so far.
                            - `method` (EncodingMethod): The encoding method currently in use.
                            - `output_path` (Optional[str]): The string representation of the output file path,
                                                              or `None` if no output file is being used.
                            - `is_open` (bool): Indicates whether the encoder is currently active.
        """
        return {
            "encoded_count": self.encoded_count,
            "method": self.method,
            "output_path": str(self.output_path) if self.output_path else None,
            "is_open": self.is_open
        }