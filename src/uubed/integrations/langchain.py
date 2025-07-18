#!/usr/bin/env python3
# this_file: src/uubed/integrations/langchain.py
"""LangChain integration for uubed encoding.

This module provides LangChain-compatible components for encoding embeddings
using uubed's position-safe algorithms. It aims to facilitate the use of
uubed-encoded strings within LangChain pipelines, particularly for scenarios
where compact, position-safe representations of embeddings are beneficial.

**Key Components:**
- `UubedEncoder`: A document transformer that encodes numerical embeddings
  into uubed strings and optionally adds them to document metadata.
- `UubedEmbeddings`: A wrapper around existing LangChain `Embeddings` models
  to automatically apply uubed encoding to generated numerical embeddings.
- `UubedDocumentProcessor`: A utility class for batch processing documents,
  generating their embeddings, and applying uubed encoding before storage
  in vector databases.
- `create_uubed_retriever`: A helper function to configure LangChain `VectorStore`
  instances to use uubed encoding for query embeddings during search operations.

**Purpose:**
- **Compact Storage:** Enable storing embeddings in a more compact, text-based format
  within vector databases or other storage solutions.
- **Position Safety:** Leverage uubed's position-safe properties for enhanced
  search capabilities or data integrity.
- **Seamless Integration:** Provide a straightforward way to incorporate uubed
  encoding into existing LangChain workflows without extensive modifications.

**Note on Dependencies:**
This module requires `langchain` to be installed. If it's not found, an `ImportError`
will be raised with instructions on how to install it.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    # Import necessary LangChain components. These are optional dependencies.
    from langchain.embeddings.base import Embeddings
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings as EmbeddingsProtocol
    from langchain.vectorstores.base import VectorStore
except ImportError:
    # Raise an ImportError if LangChain is not installed, guiding the user.
    raise ImportError(
        "LangChain is required for this integration. "
        "Install it with: `pip install langchain`"
    )

from ..api import EncodingMethod, decode, encode
from ..streaming import encode_stream


class UubedEncoder:
    """
    A LangChain-compatible document transformer that encodes embeddings using uubed.

    This transformer is designed to be integrated into LangChain processing pipelines.
    It takes a list of `Document` objects and their corresponding numerical embeddings,
    encodes these embeddings into uubed's position-safe string format, and then
    optionally adds the encoded string to the document's metadata. This is useful
    for preparing documents for storage in vector databases where a compact and
    position-safe representation of embeddings is desired.

    **Usage:**
    Typically used after an embedding model has generated numerical embeddings for documents.

    **Example:**
    ```python
    from langchain.schema import Document
    from uubed.integrations.langchain import UubedEncoder
    import numpy as np

    # Sample documents and their numerical embeddings
    docs = [
        Document(page_content="Hello world", metadata={"source": "test"}),
        Document(page_content="Another document", metadata={"source": "example"}),
    ]
    # Assume these embeddings are generated by a LangChain Embeddings model
    embeddings_data = [
        np.random.rand(768).tolist(), # Example 768-dim embedding
        np.random.rand(768).tolist(),
    ]

    # Initialize the encoder
    uubed_encoder = UubedEncoder(method="shq64", metadata_key="uubed_hash")

    # Encode documents
    encoded_docs = uubed_encoder.encode_documents(docs, embeddings_data)

    for doc in encoded_docs:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Encoded Hash: {doc.metadata.get(\'uubed_hash\')[:10]}...\n")
    ```
    """

    def __init__(
        self,
        method: EncodingMethod = "auto",
        add_to_metadata: bool = True,
        metadata_key: str = "uubed_encoded",
        **encoder_kwargs: Any
    ):
        """
        Initializes the UubedEncoder instance.

        Args:
            method (EncodingMethod): The uubed encoding method to use (e.g., "shq64", "eq64", "auto").
                                     This method will be applied when encoding numerical embeddings.
                                     Defaults to "auto".
            add_to_metadata (bool): If `True`, the generated uubed-encoded string will be added
                                    to the `Document`'s metadata dictionary under the specified `metadata_key`.
                                    If `False`, the original documents are returned without modification to metadata.
                                    Defaults to `True`.
            metadata_key (str): The key to use in the document's metadata dictionary for storing the
                                encoded string. This is only relevant if `add_to_metadata` is `True`.
                                Defaults to "uubed_encoded".
            **encoder_kwargs (Any): Additional keyword arguments to pass directly to the
                                    `uubed.api.encode` function. These can include method-specific
                                    parameters like `k` for "t8q64" or `planes` for "shq64".
        """
        self.method: EncodingMethod = method
        self.add_to_metadata: bool = add_to_metadata
        self.metadata_key: str = metadata_key
        self.encoder_kwargs: dict[str, Any] = encoder_kwargs

    def encode_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]]
    ) -> list[Document]:
        """
        Encodes a list of numerical embedding vectors and optionally integrates them into document metadata.

        This method processes a list of LangChain `Document` objects and their corresponding
        numerical embeddings. For each pair, it generates a uubed-encoded string from the
        numerical embedding. If `add_to_metadata` is `True` (configured during initialization),
        a new `Document` object is created with the uubed string added to its metadata.

        Args:
            documents (List[Document]): A list of LangChain `Document` objects.
            embeddings (List[List[float]]): A list of numerical embedding vectors (as lists of floats),
                                            where each embedding corresponds to a document in the `documents` list.

        Returns:
            List[Document]: A new list of `Document` objects. If `add_to_metadata` is `True`, each
                            document's metadata will include the uubed-encoded string under `metadata_key`.
                            Otherwise, the original documents are returned without metadata modification.

        Raises:
            ValueError: If the number of `documents` does not match the number of `embeddings`.
            UubedValidationError: If an individual embedding is malformed or cannot be encoded.
            UubedEncodingError: If an error occurs during the uubed encoding process.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})."
            )

        encoded_docs: list[Document] = []
        # Iterate through documents and their corresponding embeddings using `zip` for parallel processing.
        for doc, embedding_list in zip(documents, embeddings, strict=False):
            # Convert the embedding list to a NumPy array. Using `float32` is common for embeddings.
            # The `uubed.api.encode` function is designed to handle this conversion and normalization internally.
            embedding_array: np.ndarray = np.array(embedding_list, dtype=np.float32)

            # Encode the numerical embedding into a uubed string using the configured method and keyword arguments.
            encoded_string: str = self.encode_embedding(embedding_list) # Use encode_embedding for consistency and error handling

            if self.add_to_metadata:
                # Create a copy of the document's metadata to avoid modifying the original `Document` object in place.
                new_metadata: dict[str, Any] = doc.metadata.copy()
                new_metadata[self.metadata_key] = encoded_string
                # Create a new `Document` object with the updated metadata.
                new_doc: Document = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
                encoded_docs.append(new_doc)
            else:
                # If not configured to add to metadata, simply append the original document to the result list.
                encoded_docs.append(doc)

        return encoded_docs

    def encode_embedding(self, embedding: list[float]) -> str:
        """
        Encodes a single numerical embedding vector into a uubed string.

        This is a helper method used by `encode_documents` and `UubedEmbeddings`
        to perform the actual uubed encoding of a single numerical vector.

        Args:
            embedding (List[float]): A single embedding vector as a list of floats.

        Returns:
            str: The uubed-encoded string representation of the embedding.

        Raises:
            UubedValidationError: If the input `embedding` is malformed or cannot be encoded.
            UubedEncodingError: If an error occurs during the uubed encoding process.
        """
        # Convert the embedding list to a NumPy array. `uubed.api.encode` handles normalization
        # from float to uint8 internally based on the configured method.
        embedding_array: np.ndarray = np.array(embedding, dtype=np.float32)

        # Encode the embedding using the configured method and kwargs.
        try:
            return encode(embedding_array, method=self.method, **self.encoder_kwargs)
        except Exception as e:
            # Catch any exception from the underlying `encode` function and re-raise it
            # as a more specific `UubedEncodingError` or `UubedValidationError` for clarity.
            if isinstance(e, (UubedValidationError, UubedEncodingError)):
                raise  # Re-raise directly if it's already a uubed-specific error.
            raise UubedEncodingError(f"Failed to encode embedding: {e}") from e


def create_uubed_retriever(
    vectorstore: VectorStore,
    embeddings: Embeddings,
    method: EncodingMethod = "shq64",
    search_type: str = "similarity",
    search_kwargs: dict[str, Any] | None = None,
) -> VectorStore:
    """
    Configures a LangChain `VectorStore` to use uubed encoding for query embeddings.

    This function is designed to integrate uubed encoding into the search mechanism
    of a LangChain `VectorStore`. It wraps the provided `embeddings` model with
    `UubedEmbeddings` to ensure that any query embeddings generated for search
    operations are first processed by uubed. This is particularly useful for vector
    stores that can leverage uubed's compact or position-safe properties during search.

    **Important Considerations:**
    - The `return_encoded` parameter of the internal `UubedEmbeddings` wrapper is
      set to `False`. This is because most vector stores expect numerical embeddings
      for their internal similarity calculations, even if the underlying data might
      be stored as uubed strings.
    - This function modifies the `vectorstore` object in place by setting its
      `embedding_function` attribute.

    Args:
        vectorstore (VectorStore): The base LangChain `VectorStore` instance to configure.
                                   This object will have its `embedding_function` updated.
        embeddings (Embeddings): The LangChain `Embeddings` model to use for generating
                                 numerical embeddings before uubed encoding is applied.
        method (EncodingMethod): The uubed encoding method to use for encoding query embeddings
                                 during search operations. Defaults to "shq64".
        search_type (str): The type of search to perform (e.g., "similarity", "mmr").
                           This parameter is passed directly to `vectorstore.as_retriever()`
                           when the retriever is created. Defaults to "similarity".
        search_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to
                                                  `vectorstore.as_retriever()`. These can include
                                                  parameters like `k` (number of results) or `filter`
                                                  (metadata filtering). Defaults to `None`.

    Returns:
        VectorStore: The configured `VectorStore` instance, ready to be used as a retriever.
                     Note that this is the same `vectorstore` object passed as input, but modified.

    Example:
        ```python
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        from uubed.integrations.langchain import create_uubed_retriever

        # Initialize a base vector store (e.g., Chroma) and an embedding model
        my_vectorstore = Chroma()
        my_embeddings = OpenAIEmbeddings()

        # Create a retriever that uses uubed encoding for queries
        uubed_retriever = create_uubed_retriever(
            my_vectorstore,
            my_embeddings,
            method="shq64",
            search_kwargs={"k": 5} # Retrieve top 5 results
        )

        # Now, `uubed_retriever` can be used in LangChain chains or agents.
        # When `uubed_retriever.get_relevant_documents()` is called, the query
        # will first be embedded numerically, then uubed-encoded, and then
        # passed to the underlying vector store's search mechanism.
        ```
    """
    # Wrap the base embeddings with `UubedEmbeddings`. Crucially, `return_encoded` is set to `False`
    # here. This is because the `VectorStore` typically expects numerical embeddings for its internal
    # similarity calculations, even if the underlying data might be stored as uubed strings.
    uubed_embeddings_wrapper: UubedEmbeddings = UubedEmbeddings(
        embeddings,
        method=method,
        return_encoded=False  # Vector stores usually expect numerical embeddings for search
    )

    # Assign the wrapped embeddings as the embedding function for the vector store.
    # This modifies the `vectorstore` object in place, ensuring all subsequent embedding
    # operations (e.g., for queries) go through the `uubed_embeddings_wrapper`.
    vectorstore.embedding_function = uubed_embeddings_wrapper

    # Return the vector store configured as a retriever with specified search settings.
    # `search_kwargs or {}` ensures that if `search_kwargs` is `None`, an empty dictionary is used.
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs or {}  # Ensure search_kwargs is a dict if None is passed.
    )


class UubedDocumentProcessor:
    """
    A utility class for processing documents with uubed encoding within LangChain pipelines.

    This processor is designed to be used before documents are added to a vector database
    or other storage solutions. It takes raw LangChain `Document` objects, generates their
    numerical embeddings using a provided `Embeddings` model, and then encodes these
    numerical embeddings into uubed strings. Optionally, it adds these uubed-encoded
    strings to the document metadata. This prepares documents for efficient storage
    and retrieval in systems that can leverage uubed's properties.

    **Usage:**
    ```python
    from langchain.schema import Document
    from langchain.embeddings import OpenAIEmbeddings
    from uubed.integrations.langchain import UubedDocumentProcessor

    # Initialize your embedding model
    embedder = OpenAIEmbeddings()

    # Initialize the document processor
    processor = UubedDocumentProcessor(
        embeddings=embedder,
        method="eq64",
        batch_size=50 # Process 50 documents at a time
    )

    # Sample documents
    docs_to_process = [
        Document(page_content="The quick brown fox jumps over the lazy dog."),
        Document(page_content="Never underestimate the power of a good book."),
        # ... more documents
    ]

    # Process the documents
    processed_docs = processor.process(docs_to_process)

    for doc in processed_docs:
        print(f"Original Content: {doc.page_content[:30]}...")
        print(f"Uubed Encoded: {doc.metadata.get('uubed_encoded')[:10]}...")
        print("---")
    ```
    """

    def __init__(
        self,
        embeddings: Embeddings,
        method: EncodingMethod = "auto",
        batch_size: int = 100,
        **encoder_kwargs: Any
    ):
        """
        Initializes the UubedDocumentProcessor.

        Args:
            embeddings (Embeddings): The LangChain `Embeddings` model to use for generating
                                     numerical embeddings from document content. This model
                                     should be capable of embedding text into numerical vectors.
            method (EncodingMethod): The uubed encoding method to apply to the numerical embeddings.
                                     Defaults to "auto".
            batch_size (int): The number of documents to process in a single batch. This parameter
                              influences memory usage and the efficiency of calls to the underlying
                              embedding model. Defaults to 100.
            **encoder_kwargs (Any): Additional keyword arguments to pass to the `UubedEncoder`
                                    constructor. These will then be passed to `uubed.api.encode`.
        """
        self.embeddings: Embeddings = embeddings
        # Initialize a `UubedEncoder` instance to handle the actual encoding of embeddings.
        # The `add_to_metadata` and `metadata_key` for this encoder will be its defaults
        # unless explicitly overridden via `encoder_kwargs` if `UubedEncoder` supported it.
        self.encoder: UubedEncoder = UubedEncoder(method=method, **encoder_kwargs)
        self.batch_size: int = batch_size

    def process(self, documents: list[Document]) -> list[Document]:
        """
        Processes a list of documents by generating numerical embeddings and applying uubed encoding.

        This method iterates through the input `documents` in batches. For each batch,
        it extracts the text content, generates numerical embeddings using the configured
        `embeddings` model, and then uses the internal `UubedEncoder` to convert these
        numerical embeddings into uubed strings. The uubed strings are then (optionally)
        added to the document metadata.

        Args:
            documents (List[Document]): A list of LangChain `Document` objects to process.

        Returns:
            List[Document]: A new list of `Document` objects. Each document in this list
                            will have its metadata updated (if configured in `UubedEncoder`)
                            to include the uubed-encoded string of its embedding.
        """
        processed_docs: list[Document] = []

        # Process documents in batches for efficiency and to manage memory usage.
        for i in range(0, len(documents), self.batch_size):
            batch: list[Document] = documents[i:i + self.batch_size]

            # Extract page content from the batch of documents to generate embeddings.
            texts: list[str] = [doc.page_content for doc in batch]
            # Generate numerical embeddings for the batch of texts using the base embedding model.
            embeddings: list[list[float]] = self.embeddings.embed_documents(texts)

            # Encode these numerical embeddings into uubed strings and integrate them back into the documents.
            # The `encode_documents` method of `UubedEncoder` handles adding to metadata.
            encoded_batch: list[Document] = self.encoder.encode_documents(batch, embeddings)
            processed_docs.extend(encoded_batch)

        return processed_docs

    async def aprocess(self, documents: list[Document]) -> list[Document]:
        """
        Asynchronously processes a list of documents by generating embeddings and applying uubed encoding.

        This method is the asynchronous counterpart to `process`. It performs the same
        batch processing, numerical embedding generation, and uubed encoding, but it
        uses the asynchronous `aembed_documents` method of the underlying `embeddings` model.

        Args:
            documents (List[Document]): A list of LangChain `Document` objects to process.

        Returns:
            List[Document]: A new list of `Document` objects, where each document's
                            metadata (if configured in `UubedEncoder`) includes the
                            uubed-encoded string of its embedding.
        """
        processed_docs: list[Document] = []

        # Process documents in batches asynchronously.
        for i in range(0, len(documents), self.batch_size):
            batch: list[Document] = documents[i:i + self.batch_size]

            # Extract page content for asynchronous embedding generation.
            texts: list[str] = [doc.page_content for doc in batch]
            # Asynchronously generate numerical embeddings for the batch.
            embeddings: list[list[float]] = await self.embeddings.aembed_documents(texts)

            # Encode these embeddings and integrate them back into the documents.
            encoded_batch: list[Document] = self.encoder.encode_documents(batch, embeddings)
            processed_docs.extend(encoded_batch)

        return processed_docs


def create_uubed_retriever(
    vectorstore: VectorStore,
    embeddings: Embeddings,
    method: EncodingMethod = "shq64",
    search_type: str = "similarity",
    search_kwargs: dict[str, Any] | None = None,
) -> VectorStore:
    """
    Creates a LangChain `VectorStore` configured to use uubed encoding for searches.
    
    This function is particularly useful for vector stores that can perform searches
    based on string representations (e.g., by storing uubed-encoded strings directly)
    or when uubed's position-safe properties are desired for similarity matching.
    
    Args:
        vectorstore (VectorStore): The base LangChain `VectorStore` instance to configure.
        embeddings (Embeddings): The LangChain `Embeddings` model to use for generating
                                 numerical embeddings before uubed encoding.
        method (EncodingMethod): The uubed encoding method to use for encoding query embeddings
                                 during search operations. Defaults to "shq64".
        search_type (str): The type of search to perform (e.g., "similarity", "mmr").
                           Passed to `vectorstore.as_retriever()`. Defaults to "similarity".
        search_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to
                                                  `vectorstore.as_retriever()`. Defaults to `None`.
        
    Returns:
        VectorStore: The configured `VectorStore` instance, ready to be used as a retriever.
                     Note: This function modifies the `vectorstore` object in place by setting
                     its `embedding_function` attribute.
        
    Example:
        >>> from langchain.vectorstores import Chroma
        >>> from langchain.embeddings import OpenAIEmbeddings
        >>> 
        >>> # Initialize a base vector store and embedding model.
        >>> vectorstore = Chroma()
        >>> embeddings = OpenAIEmbeddings()
        >>> 
        >>> # Create a retriever that uses uubed encoding for queries.
        >>> retriever = create_uubed_retriever(
        ...     vectorstore,
        ...     embeddings,
        ...     method="shq64"
        ... )
    """
    # Wrap the base embeddings with UubedEmbeddings. Crucially, `return_encoded` is False
    # here because the vectorstore typically expects numerical embeddings for its internal
    # similarity calculations, even if the underlying data might be stored as strings.
    uubed_embeddings_wrapper: UubedEmbeddings = UubedEmbeddings(
        embeddings,
        method=method,
        return_encoded=False # Vector stores usually expect numerical embeddings
    )

    # Assign the wrapped embeddings as the embedding function for the vector store.
    # This modifies the vectorstore object in place.
    vectorstore.embedding_function = uubed_embeddings_wrapper

    # Return the vector store configured as a retriever with specified search settings.
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs or {} # Ensure search_kwargs is a dict.
    )


class UubedDocumentProcessor:
    """
    A utility class for processing documents with uubed encoding within LangChain pipelines.

    This processor is designed to be used before documents are added to a vector database
    or other storage solutions. It takes raw LangChain `Document` objects, generates their
    numerical embeddings using a provided `Embeddings` model, and then encodes these
    numerical embeddings into uubed strings. Optionally, it adds these uubed-encoded
    strings to the document metadata. This prepares documents for efficient storage
    and retrieval in systems that can leverage uubed's properties.

    **Usage:**
    ```python
    from langchain.schema import Document
    from langchain.embeddings import OpenAIEmbeddings
    from uubed.integrations.langchain import UubedDocumentProcessor

    # Initialize your embedding model
    embedder = OpenAIEmbeddings()

    # Initialize the document processor
    processor = UubedDocumentProcessor(
        embeddings=embedder,
        method="eq64",
        batch_size=50 # Process 50 documents at a time
    )

    # Sample documents
    docs_to_process = [
        Document(page_content="The quick brown fox jumps over the lazy dog."),
        Document(page_content="Never underestimate the power of a good book."),
        # ... more documents
    ]

    # Process the documents
    processed_docs = processor.process(docs_to_process)

    for doc in processed_docs:
        print(f"Original Content: {doc.page_content[:30]}...")
        print(f"Uubed Encoded: {doc.metadata.get('uubed_encoded')[:10]}...")
        print("---")
    ```
    """

    def __init__(
        self,
        embeddings: Embeddings,
        method: EncodingMethod = "auto",
        batch_size: int = 100,
        **encoder_kwargs: Any
    ):
        """
        Initializes the UubedDocumentProcessor.

        Args:
            embeddings (Embeddings): The LangChain `Embeddings` model to use for generating
                                     numerical embeddings from document content. This model
                                     should be capable of embedding text into numerical vectors.
            method (EncodingMethod): The uubed encoding method to apply to the numerical embeddings.
                                     Defaults to "auto".
            batch_size (int): The number of documents to process in a single batch. This parameter
                              influences memory usage and the efficiency of calls to the underlying
                              embedding model. Defaults to 100.
            **encoder_kwargs (Any): Additional keyword arguments to pass to the `UubedEncoder`
                                    constructor. These will then be passed to `uubed.api.encode`.
        """
        self.embeddings: Embeddings = embeddings
        # Initialize a `UubedEncoder` instance to handle the actual encoding of embeddings.
        # The `add_to_metadata` and `metadata_key` for this encoder will be its defaults
        # unless explicitly overridden via `encoder_kwargs` if `UubedEncoder` supported it.
        self.encoder: UubedEncoder = UubedEncoder(method=method, **encoder_kwargs)
        self.batch_size: int = batch_size

    def process(self, documents: list[Document]) -> list[Document]:
        """
        Processes a list of documents by generating numerical embeddings and applying uubed encoding.

        This method iterates through the input `documents` in batches. For each batch,
        it extracts the text content, generates numerical embeddings using the configured
        `embeddings` model, and then uses the internal `UubedEncoder` to convert these
        numerical embeddings into uubed strings. The uubed strings are then (optionally)
        added to the document metadata.

        Args:
            documents (List[Document]): A list of LangChain `Document` objects to process.

        Returns:
            List[Document]: A new list of `Document` objects. Each document in this list
                            will have its metadata updated (if configured in `UubedEncoder`)
                            to include the uubed-encoded string of its embedding.
        """
        processed_docs: list[Document] = []

        # Process documents in batches for efficiency and to manage memory usage.
        for i in range(0, len(documents), self.batch_size):
            batch: list[Document] = documents[i:i + self.batch_size]

            # Extract page content from the batch of documents to generate embeddings.
            texts: list[str] = [doc.page_content for doc in batch]
            # Generate numerical embeddings for the batch of texts using the base embedding model.
            embeddings: list[list[float]] = self.embeddings.embed_documents(texts)

            # Encode these numerical embeddings into uubed strings and integrate them back into the documents.
            # The `encode_documents` method of `UubedEncoder` handles adding to metadata.
            encoded_batch: list[Document] = self.encoder.encode_documents(batch, embeddings)
            processed_docs.extend(encoded_batch)

        return processed_docs

    async def aprocess(self, documents: list[Document]) -> list[Document]:
        """
        Asynchronously processes a list of documents by generating embeddings and applying uubed encoding.

        This method is the asynchronous counterpart to `process`. It performs the same
        batch processing, numerical embedding generation, and uubed encoding, but it
        uses the asynchronous `aembed_documents` method of the underlying `embeddings` model.

        Args:
            documents (List[Document]): A list of LangChain `Document` objects to process.

        Returns:
            List[Document]: A new list of `Document` objects, where each document's
                            metadata (if configured in `UubedEncoder`) includes the
                            uubed-encoded string of its embedding.
        """
        processed_docs: list[Document] = []

        # Process documents in batches asynchronously.
        for i in range(0, len(documents), self.batch_size):
            batch: list[Document] = documents[i:i + self.batch_size]

            # Extract page content for asynchronous embedding generation.
            texts: list[str] = [doc.page_content for doc in batch]
            # Asynchronously generate numerical embeddings for the batch.
            embeddings: list[list[float]] = await self.embeddings.aembed_documents(texts)

            # Encode these embeddings and integrate them back into the documents.
            encoded_batch: list[Document] = self.encoder.encode_documents(batch, embeddings)
            processed_docs.extend(encoded_batch)

        return processed_docs
