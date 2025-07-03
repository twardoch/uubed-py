#!/usr/bin/env python3
# this_file: src/uubed/matryoshka.py
"""Matryoshka (nested) embedding support for uubed.

Matryoshka embeddings allow multiple representation granularities within
a single embedding vector. This module provides utilities for encoding
different levels of a Matryoshka embedding using uubed's position-safe
algorithms, and for building a basic search index that leverages these
hierarchical representations.

References:
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.13147
"""

from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np
import math

from .api import encode, decode, EncodingMethod
from .exceptions import UubedValidationError, UubedDecodingError


class MatryoshkaEncoder:
    """
    Encoder for Matryoshka embeddings with multiple granularity levels.
    
    Matryoshka embeddings contain multiple nested representations where
    the first k dimensions contain a valid embedding of lower dimensionality.
    This encoder can create separate encodings for each level, allowing
    for progressive refinement or multi-granularity analysis.
    """
    
    def __init__(
        self,
        dimensions: List[int],
        base_method: EncodingMethod = "auto",
        level_methods: Optional[Dict[int, EncodingMethod]] = None
    ):
        """
        Initialize the Matryoshka encoder.
        
        Args:
            dimensions: A list of embedding dimensions that define the granularity levels.
                        These dimensions must be sorted in ascending order.
            base_method: The default encoding method to use for all levels if no
                         specific method is provided for a dimension. Defaults to "auto".
            level_methods: An optional dictionary mapping specific dimension sizes
                           to their preferred encoding methods. This overrides `base_method`
                           for the specified dimensions.
        
        Raises:
            ValueError: If `dimensions` is empty or contains non-positive values.
        """
        # Ensure dimensions are sorted and store them.
        self.dimensions: List[int] = sorted(dimensions)
        self.base_method: EncodingMethod = base_method
        # Use an empty dict if no level_methods are provided.
        self.level_methods: Dict[int, EncodingMethod] = level_methods or {}
        
        # Validate dimensions to ensure they are positive.
        if not self.dimensions:
            raise ValueError("At least one dimension must be specified for MatryoshkaEncoder.")
        
        for dim in self.dimensions:
            if dim <= 0:
                raise ValueError(f"All dimensions must be positive integers, but got {dim}.")
    
    def get_method_for_level(self, dimension: int) -> EncodingMethod:
        """
        Retrieves the appropriate encoding method for a given dimension level.
        
        Args:
            dimension: The dimension level for which to get the encoding method.
            
        Returns:
            EncodingMethod: The encoding method configured for that dimension, or the base method.
        """
        return self.level_methods.get(dimension, self.base_method)
    
    def encode_level(
        self,
        embedding: Union[List[float], np.ndarray],
        dimension: int,
        method: Optional[EncodingMethod] = None,
        **kwargs: Any
    ) -> str:
        """
        Encodes a specific dimension level of a Matryoshka embedding.
        
        This method extracts the relevant portion of the embedding and encodes it
        using the specified or default method. It relies on the main `uubed.api.encode`
        function for the actual encoding, which handles input validation and normalization.
        
        Args:
            embedding: The full embedding vector (list of floats or NumPy array).
            dimension: The dimension level to extract and encode (e.g., 64, 128).
            method: Optional override for the encoding method for this specific level.
                    If `None`, the level-specific or base method from `__init__` is used.
            **kwargs: Additional method-specific parameters passed to `uubed.api.encode`.
            
        Returns:
            str: The encoded string for the specified dimension level.
            
        Raises:
            ValueError: If the requested `dimension` is not configured in the encoder,
                        or if the input `embedding` has fewer dimensions than requested.
        """
        if dimension not in self.dimensions:
            raise ValueError(f"Dimension {dimension} is not a configured level in this MatryoshkaEncoder.")
        
        # Ensure embedding is a NumPy array for consistent slicing.
        if isinstance(embedding, list):
            embedding_array: np.ndarray = np.array(embedding)
        else:
            embedding_array = embedding
        
        # Validate that the embedding has enough dimensions for the requested level.
        if embedding_array.shape[0] < dimension:
            raise ValueError(
                f"Input embedding has {embedding_array.shape[0]} dimensions, "
                f"but encoding level {dimension} was requested. Embedding must be at least {dimension} dimensions long."
            )
        
        # Extract the sub-embedding for the current dimension level.
        level_embedding: np.ndarray = embedding_array[:dimension]
        
        # Determine the encoding method for this level.
        # Explicitly provided method takes precedence, then level-specific, then base method.
        encoding_method: EncodingMethod = method or self.get_method_for_level(dimension)
        
        # Call the main `encode` function. It handles its own input validation and normalization.
        return encode(level_embedding, method=encoding_method, **kwargs)
    
    def encode_all_levels(
        self,
        embedding: Union[List[float], np.ndarray],
        method: Optional[EncodingMethod] = None,
        **kwargs: Any
    ) -> Dict[int, str]:
        """
        Encodes the input embedding at all configured dimension levels.
        
        Args:
            embedding: The full embedding vector (list of floats or NumPy array).
            method: Optional override for the encoding method for all levels.
                    If `None`, level-specific or base methods are used.
            **kwargs: Additional method-specific parameters passed to `uubed.api.encode`.
            
        Returns:
            Dict[int, str]: A dictionary where keys are dimension levels and values
                            are the corresponding encoded strings.
        """
        results: Dict[int, str] = {}
        
        # Iterate through each configured dimension level and encode the corresponding sub-embedding.
        for dim in self.dimensions:
            # Determine the encoding method for the current level.
            encoding_method: EncodingMethod = method or self.get_method_for_level(dim)
            results[dim] = self.encode_level(
                embedding, dim, method=encoding_method, **kwargs
            )
        
        return results
    
    def decode_level(
        self,
        encoded: str,
        dimension: int,
        method: Optional[EncodingMethod] = None
    ) -> np.ndarray:
        """
        Decodes an encoded string back to a NumPy array for a specific dimension level.
        
        Args:
            encoded: The encoded string to decode.
            dimension: The original dimension level of the embedding that was encoded.
            method: Optional override for the encoding method used. If `None`, auto-detection
                    or level-specific method is used.
            
        Returns:
            np.ndarray: The decoded embedding vector as a NumPy array of `uint8`.
            
        Raises:
            NotImplementedError: If the encoding method used is lossy and does not support full decoding.
            UubedDecodingError: If the decoding operation fails.
        
        Note:
            Only lossless encoding methods (currently `eq64` and `mq64`) support full decoding
            back to the original byte representation. Attempting to decode lossy methods
            will raise a `NotImplementedError`.
        """
        # Determine the encoding method for this level.
        encoding_method: EncodingMethod = method or self.get_method_for_level(dimension)
        
        # Check if the method supports decoding.
        if encoding_method not in ("eq64", "mq64"):
            raise NotImplementedError(
                f"Decoding is not supported for the '{encoding_method}' method. "
                "Only 'eq64' and 'mq64' provide lossless encoding and full decoding capability."
            )
        
        # Decode the string to bytes using the main `decode` function.
        decoded_bytes: bytes = decode(encoded, method=encoding_method)
        # Convert the bytes back to a NumPy array of unsigned 8-bit integers.
        return np.frombuffer(decoded_bytes, dtype=np.uint8)
    
    def get_encoding_stats(
        self,
        embedding: Union[List[float], np.ndarray],
        **kwargs: Any
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculates and returns encoding statistics for all configured dimension levels.
        
        This provides insights into the compression ratio and encoded length for each level.
        
        Args:
            embedding: The full embedding vector (list of floats or NumPy array).
            **kwargs: Additional method-specific parameters passed to `uubed.api.encode`.
            
        Returns:
            Dict[int, Dict[str, Any]]: A dictionary where keys are dimension levels and values
                                       are dictionaries containing statistics for that level,
                                       such as method used, encoded length, byte size, original
                                       byte size, compression ratio, and a sample of the encoded string.
        """
        stats: Dict[int, Dict[str, Any]] = {}
        
        for dim in self.dimensions:
            # Get the method for the current dimension level.
            method: EncodingMethod = self.get_method_for_level(dim)
            # Encode the level to get the encoded string.
            encoded: str = self.encode_level(embedding, dim, **kwargs)
            
            # Calculate original and encoded byte sizes.
            original_bytes: int = dim  # Assuming 1 byte per dimension for uint8 representation.
            encoded_bytes: int = len(encoded.encode('utf-8')) # Get byte length of the UTF-8 encoded string.
            
            # Calculate compression ratio.
            compression_ratio: float = original_bytes / encoded_bytes if encoded_bytes > 0 else 0.0
            
            stats[dim] = {
                "method": method,
                "encoded_length": len(encoded),
                "encoded_bytes": encoded_bytes,
                "original_bytes": original_bytes,
                "compression_ratio": compression_ratio,
                "encoded_sample": encoded[:20] + "..." if len(encoded) > 20 else encoded
            }
        
        return stats


class MatryoshkaSearchIndex:
    """
    A basic search index for Matryoshka embeddings, supporting progressive refinement.
    
    This index allows storing embeddings at multiple granularity levels and performing
    searches that can start with lower-dimensional representations and progressively
    refine to higher dimensions for more accurate results. Note that the similarity
    metric used (`_calculate_string_similarity`) is a simple demonstration and not
    suitable for robust vector similarity search in production environments.
    """
    
    def __init__(
        self,
        encoder: MatryoshkaEncoder,
        enable_progressive_search: bool = True
    ):
        """
        Initialize the Matryoshka search index.
        
        Args:
            encoder: An instance of `MatryoshkaEncoder` configured with the desired
                     dimension levels and encoding methods.
            enable_progressive_search: If `True`, enables the progressive refinement
                                       search strategy. If `False` or if only one
                                       dimension is configured, search defaults to
                                       the highest dimension.
        """
        self.encoder: MatryoshkaEncoder = encoder
        self.enable_progressive_search: bool = enable_progressive_search
        # Stores encoded embeddings: dimension -> {embedding_id: encoded_string}
        self.index: Dict[int, Dict[str, str]] = {}
        # Stores original metadata associated with each embedding_id.
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
    def add_embedding(
        self,
        embedding_id: str,
        embedding: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[int, str]:
        """
        Adds an embedding to the index at all configured dimension levels.
        
        Args:
            embedding_id: A unique string identifier for the embedding.
            embedding: The full embedding vector (list of floats or NumPy array).
            metadata: Optional dictionary of additional metadata to store with the embedding.
            **kwargs: Encoding parameters passed to the `MatryoshkaEncoder`.
            
        Returns:
            Dict[int, str]: A dictionary of encoded strings for each dimension level.
        """
        # Encode the full embedding at all configured levels.
        encoded_levels: Dict[int, str] = self.encoder.encode_all_levels(embedding, **kwargs)
        
        # Store the encoded strings in the index, organized by dimension level.
        for dim, encoded in encoded_levels.items():
            if dim not in self.index:
                self.index[dim] = {} # Initialize dictionary for this dimension if it doesn't exist.
            self.index[dim][embedding_id] = encoded
        
        # Store any associated metadata.
        if metadata:
            self.metadata[embedding_id] = metadata
        
        return encoded_levels
    
    def search_level(
        self,
        query_embedding: Union[List[float], np.ndarray],
        dimension: int,
        top_k: int = 10,
        method: Optional[EncodingMethod] = None,
        **kwargs: Any
    ) -> List[Tuple[str, str, float]]:
        """
        Searches for similar embeddings at a specific dimension level.
        
        This method encodes the query embedding at the specified dimension and
        then compares it against all indexed embeddings at that same dimension
        using a simple string similarity metric.
        
        Args:
            query_embedding: The query embedding vector (list of floats or NumPy array).
            dimension: The dimension level to perform the search on.
            top_k: The number of top similar results to return. Defaults to 10.
            method: Optional override for the encoding method for the query embedding.
            **kwargs: Encoding parameters passed to the `MatryoshkaEncoder`.
            
        Returns:
            List[Tuple[str, str, float]]: A list of tuples, where each tuple contains:
                                         (embedding_id, encoded_string_at_dimension, similarity_score).
                                         Results are sorted by similarity score in descending order.
        """
        if dimension not in self.index:
            # If the requested dimension is not in the index, return an empty list.
            return []
        
        # Encode the query embedding at the specified dimension level.
        query_encoded: str = self.encoder.encode_level(
            query_embedding, dimension, method=method, **kwargs
        )
        
        # Perform a simple string similarity comparison against all indexed embeddings
        # at this dimension level. This is for demonstration purposes only.
        results: List[Tuple[str, str, float]] = []
        for emb_id, encoded_str in self.index[dimension].items():
            similarity: float = self._calculate_string_similarity(query_encoded, encoded_str)
            results.append((emb_id, encoded_str, similarity))
        
        # Sort the results by similarity score in descending order and return the top_k.
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def progressive_search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        initial_candidates: int = 100,
        final_top_k: int = 10,
        **kwargs: Any
    ) -> List[Tuple[str, Dict[int, str], float]]:
        """
        Performs a progressive search starting from the lowest dimension level.
        
        This strategy refines candidates through successive dimension levels,
        aiming to improve search accuracy while potentially reducing computation
        by filtering out less relevant candidates early.
        
        Args:
            query_embedding: The query embedding vector (list of floats or NumPy array).
            initial_candidates: The number of top candidates to retrieve from the first (lowest)
                                dimension level. Defaults to 100.
            final_top_k: The final number of top similar results to return after all
                         refinement stages. Defaults to 10.
            **kwargs: Encoding parameters passed to the `MatryoshkaEncoder`.
            
        Returns:
            List[Tuple[str, Dict[int, str], float]]: A list of tuples, where each tuple contains:
                                                     (embedding_id, all_encodings_for_id, final_similarity_score).
                                                     Results are sorted by final similarity score in descending order.
        """
        # If progressive search is disabled or only one dimension is configured,
        # fall back to a simple search on the highest dimension.
        if not self.enable_progressive_search or len(self.encoder.dimensions) < 2:
            highest_dim: int = max(self.encoder.dimensions)
            results_single_level = self.search_level(query_embedding, highest_dim, final_top_k, **kwargs)
            # Format results to match the progressive search output structure.
            return [(r[0], {highest_dim: r[1]}, r[2]) for r in results_single_level]
        
        current_candidates: set[str] = set() # Set to store candidate embedding IDs.
        
        # Iterate through dimension levels, from lowest to highest.
        for i, dim in enumerate(self.encoder.dimensions):
            if i == 0:
                # First level: Perform an initial broad search to get candidates.
                results_first_level = self.search_level(
                    query_embedding, dim, initial_candidates, **kwargs
                )
                current_candidates = {r[0] for r in results_first_level}
            else:
                # Subsequent levels: Refine the current set of candidates.
                if not current_candidates:
                    # If no candidates remain, stop further refinement.
                    break
                
                candidates_scores: List[Tuple[str, float]] = []
                # Encode the query embedding at the current dimension level.
                query_encoded_current_dim: str = self.encoder.encode_level(query_embedding, dim, **kwargs)
                
                # Score only the current candidates at this dimension level.
                for candidate_id in current_candidates:
                    if candidate_id in self.index[dim]:
                        encoded_candidate_str: str = self.index[dim][candidate_id]
                        similarity: float = self._calculate_string_similarity(query_encoded_current_dim, encoded_candidate_str)
                        candidates_scores.append((candidate_id, similarity))
                
                # Sort candidates by similarity and select a reduced set for the next level.
                candidates_scores.sort(key=lambda x: x[1], reverse=True)
                # Dynamically adjust the number of candidates to keep.
                keep_count: int = min(len(candidates_scores), max(final_top_k, initial_candidates // (2 ** i)))
                current_candidates = {cs[0] for cs in candidates_scores[:keep_count]}
        
        # Final scoring at the highest dimension level for the remaining candidates.
        highest_dim: int = max(self.encoder.dimensions)
        final_results: List[Tuple[str, Dict[int, str], float]] = []
        query_encoded_highest_dim: str = self.encoder.encode_level(query_embedding, highest_dim, **kwargs)
        
        for candidate_id in current_candidates:
            if candidate_id in self.index[highest_dim]:
                # Collect all encodings for this final candidate across all dimensions.
                all_encodings_for_candidate: Dict[int, str] = {}
                for dim_level in self.encoder.dimensions:
                    if candidate_id in self.index[dim_level]:
                        all_encodings_for_candidate[dim_level] = self.index[dim_level][candidate_id]
                
                # Calculate the final similarity score using the highest dimension.
                encoded_final_candidate: str = self.index[highest_dim][candidate_id]
                final_similarity: float = self._calculate_string_similarity(query_encoded_highest_dim, encoded_final_candidate)
                final_results.append((candidate_id, all_encodings_for_candidate, final_similarity))
        
        # Sort the final results by similarity and return the top_k.
        final_results.sort(key=lambda x: x[2], reverse=True)
        return final_results[:final_top_k]
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculates a simple character-wise similarity between two strings.
        
        This function is intended for demonstration purposes within the MatryoshkaSearchIndex
        and is NOT a suitable metric for true vector embedding similarity. It computes
        the proportion of matching characters at corresponding positions.
        
        Args:
            str1: The first string.
            str2: The second string.
            
        Returns:
            float: A similarity score between 0.0 and 1.0. Returns 0.0 if strings have different lengths.
        """
        if len(str1) != len(str2):
            return 0.0 # Strings of different lengths are considered completely dissimilar.
        
        # Count matching characters at the same positions.
        matches: int = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        # Calculate similarity as the ratio of matches to total length.
        return matches / len(str1)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics about the current state of the search index.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                            - `total_embeddings`: The total number of unique embeddings indexed.
                            - `dimensions`: The configured dimension levels of the encoder.
                            - `levels`: A nested dictionary with statistics for each dimension level,
                                        including the count of embeddings at that level and the
                                        encoding method used.
        """
        stats: Dict[str, Any] = {
            "total_embeddings": len(self.metadata),
            "dimensions": self.encoder.dimensions,
            "levels": {}
        }
        
        for dim in self.encoder.dimensions:
            if dim in self.index:
                stats["levels"][dim] = {
                    "count": len(self.index[dim]),
                    "method": self.encoder.get_method_for_level(dim)
                }
        
        return stats


def create_adaptive_matryoshka_encoder(
    max_dimension: int,
    num_levels: int = 4,
    progression: str = "exponential"
) -> MatryoshkaEncoder:
    """
    Creates a `MatryoshkaEncoder` instance with adaptively selected dimension levels.
    
    This function helps in setting up a `MatryoshkaEncoder` by automatically determining
    the intermediate dimension levels based on a specified progression type.
    It also suggests appropriate encoding methods for each level based on common heuristics.
    
    Args:
        max_dimension: The maximum dimension of the full embedding. This will be the highest
                       dimension level in the generated encoder.
        num_levels: The desired number of intermediate dimension levels to create.
                    Defaults to 4.
        progression: The strategy for spacing the dimension levels:
                     - "linear": Dimensions are evenly spaced.
                     - "exponential": Dimensions increase exponentially.
                     - "powers_of_2": Dimensions are powers of 2 (e.g., 64, 128, 256).
                     Defaults to "exponential".
            
    Returns:
        MatryoshkaEncoder: A configured instance of `MatryoshkaEncoder`.
        
    Raises:
        ValueError: If an unknown `progression` type is specified.
    """
    dimensions: List[int] = []
    
    if progression == "linear":
        if num_levels > 1:
            step: float = max_dimension / num_levels
            dimensions = [int(round(step * (i + 1))) for i in range(num_levels)]
        else:
            dimensions = [max_dimension]
    elif progression == "exponential":
        if num_levels > 1:
            # Calculate the base for exponential progression.
            # Ensures that the last dimension is `max_dimension`.
            base: float = math.pow(max_dimension, 1 / num_levels)
            dimensions = [int(round(base ** (i + 1))) for i in range(num_levels)]
        else:
            dimensions = [max_dimension]
    elif progression == "powers_of_2":
        # Calculate starting power to ensure max_dimension is included or approximated.
        # This logic aims to generate powers of 2 that are relevant to the max_dimension.
        if max_dimension <= 0: # Handle edge case for max_dimension
            raise ValueError("max_dimension must be positive for 'powers_of_2' progression.")

        # Find the largest power of 2 less than or equal to max_dimension
        current_power_of_2 = 2**(int(math.log2(max_dimension)))
        
        # Generate dimensions by going downwards from current_power_of_2
        temp_dims = []
        while current_power_of_2 >= 1 and len(temp_dims) < num_levels:
            temp_dims.append(current_power_of_2)
            current_power_of_2 //= 2
        dimensions = sorted(temp_dims) # Sort to ensure ascending order

        # Ensure max_dimension is always included if it's not already the highest power of 2
        if max_dimension not in dimensions:
            dimensions.append(max_dimension)
            dimensions.sort()

    else:
        raise ValueError(f"Unknown progression type: {progression}. Expected 'linear', 'exponential', or 'powers_of_2'.")
    
    # Ensure dimensions are unique, sorted, and do not exceed max_dimension.
    # Using set to remove duplicates, then sorting.
    dimensions = sorted(list(set(d for d in dimensions if d <= max_dimension)))
    
    # Adaptive method selection based on dimension size heuristics.
    level_methods: Dict[int, EncodingMethod] = {}
    for dim in dimensions:
        if dim <= 64:
            level_methods[dim] = "shq64"  # Compact for small dimensions.
        elif dim <= 256:
            level_methods[dim] = "t8q64"  # Sparse for medium dimensions.
        else:
            level_methods[dim] = "eq64"   # Full precision for large dimensions.
    
    return MatryoshkaEncoder(dimensions, level_methods=level_methods)
