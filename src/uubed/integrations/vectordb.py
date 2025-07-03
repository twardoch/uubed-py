#!/usr/bin/env python3
# this_file: src/uubed/integrations/vectordb.py
"""Vector database integrations for uubed.

This module provides abstract and concrete connector implementations for popular
vector databases, enabling seamless integration of uubed's position-safe encoding
with vector storage and retrieval systems. It allows users to store uubed-encoded
strings alongside numerical vectors for enhanced search capabilities or compact storage.

**Purpose:**
To offer a standardized interface for interacting with various vector databases
while automatically handling the encoding of numerical embedding vectors into
uubed's compact, position-safe string format. This facilitates:
- **Compact Storage:** Storing embeddings more efficiently in text-based fields.
- **Hybrid Search:** Enabling potential future hybrid search strategies where
  uubed codes can be used for filtering or initial approximate retrieval.
- **Interoperability:** Providing a consistent way to manage embeddings across
  different vector database backends.

**Supported Databases (via concrete connector classes):**
- Pinecone (`PineconeConnector`)
- Weaviate (`WeaviateConnector`)
- Qdrant (`QdrantConnector`)
- ChromaDB (`ChromaConnector`)

**Usage:**
Users can instantiate a specific connector via the `get_connector` factory function
and then use its methods (`connect`, `create_collection`, `insert_vectors`, `search`)
without needing to manually encode/decode uubed strings for each operation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from abc import ABC, abstractmethod

from ..api import encode, decode, EncodingMethod
from ..streaming import encode_stream
from ..exceptions import UubedValidationError, UubedResourceError, validation_error, resource_error


class VectorDBConnector(ABC):
    """Abstract base class for all vector database connectors.

    This class defines the common interface that all concrete vector database
    connectors must implement. It provides a standardized way to interact with
    different vector databases for operations such as connecting, creating collections,
    inserting vectors, and performing searches. It also includes a utility method
    for encoding vectors using uubed, which is shared across all implementations.

    Subclasses are expected to implement the abstract methods (`connect`,
    `create_collection`, `insert_vectors`, `search`) according to the specifics
    of their respective vector database APIs.
    """
    
    def __init__(self, encoding_method: EncodingMethod = "auto", **encoding_kwargs: Any):
        """
        Initializes the base connector with a default uubed encoding method and keyword arguments.

        This constructor is called by subclasses to set up the common configuration
        for uubed encoding that will be used when inserting vectors into the database.

        Args:
            encoding_method (EncodingMethod): The uubed encoding method to use for vectors
                                              before inserting them into the database.
                                              Defaults to "auto" for automatic selection.
            **encoding_kwargs (Any): Additional keyword arguments to pass directly to the
                                     `uubed.api.encode` function when encoding vectors.
                                     These can include method-specific parameters like
                                     `k` for "t8q64" or `planes` for "shq64".
        """
        self.encoding_method: EncodingMethod = encoding_method
        self.encoding_kwargs: Dict[str, Any] = encoding_kwargs
    
    @abstractmethod
    def connect(self, **kwargs: Any) -> Any:
        """Abstract method: Connects to the underlying vector database.

        Subclasses must implement this method to establish a connection to their
        specific database client or service. The return type can vary depending
        on the database client library.

        Args:
            **kwargs (Any): Connection-specific parameters (e.g., API keys, URLs, hosts,
                            authentication credentials). These are passed directly to the
                            database client's connection method.

        Returns:
            Any: The connected client object, a connection handle, or a boolean
                 indicating the success of the connection. The exact return type
                 depends on the specific database implementation.
        """
        pass
    
    @abstractmethod
    def create_collection(self, name: str, **kwargs: Any) -> Any:
        """Abstract method: Creates a new collection, index, or class in the database.

        Subclasses must implement this method to define and initialize a storage
        unit for vectors within their respective databases. This might involve
        specifying vector dimensions, similarity metrics, or other schema details.

        Args:
            name (str): The name of the collection/index/class to create.
            **kwargs (Any): Collection-specific parameters (e.g., `dimension` of vectors,
                            `metric` for similarity search, `vectorizer` settings).
                            These are passed directly to the database's collection creation method.

        Returns:
            Any: The created collection/index object, a reference to it, or a boolean
                 indicating the success of the creation. The exact return type depends
                 on the specific database implementation.
        """
        pass
    
    @abstractmethod
    def insert_vectors(self, vectors: List[Union[List[float], np.ndarray]], metadata: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> None:
        """Abstract method: Inserts vectors into the database.

        Subclasses must implement this method to handle the batch insertion of
        numerical embedding vectors along with their associated metadata.
        The uubed-encoded string of each vector should typically be stored as part
        of the metadata or payload.

        Args:
            vectors (List[Union[List[float], np.ndarray]]): A list of numerical embedding vectors
                                                              to be inserted. Each vector can be
                                                              a `List[float]` or a `np.ndarray`.
            metadata (Optional[List[Dict[str, Any]]]): An optional list of dictionaries,
                                                        where each dictionary contains metadata
                                                        for the corresponding vector. If provided,
                                                        its length must match the `vectors` list.
                                                        Defaults to `None`.
            **kwargs (Any): Insertion-specific parameters (e.g., `batch_size` for batching inserts,
                            `ids` for specifying unique identifiers for each vector).
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: Union[List[float], np.ndarray], top_k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        """Abstract method: Searches for similar vectors in the database.

        Subclasses must implement this method to perform a similarity search
        against the stored vectors using a given query vector. The results should
        typically include the ID, similarity score, and any associated metadata
        (including the uubed-encoded string).

        Args:
            query_vector (Union[List[float], np.ndarray]): The numerical query embedding vector.
                                                              Can be a `List[float]` or a `np.ndarray`.
            top_k (int): The maximum number of top similar results to retrieve.
                         Defaults to 10.
            **kwargs (Any): Search-specific parameters (e.g., `filters` to narrow down the search,
                            `include_values` to retrieve the original vector values).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents
                                  a search result. Each result typically includes:
                                  - `id`: The unique identifier of the matched vector.
                                  - `score`: The similarity score between the query and the matched vector.
                                  - `metadata` or `payload`: A dictionary containing associated metadata,
                                    which should include the `uubed_encoded` string.
                                  - `uubed_encoded`: The uubed-encoded string of the matched vector.
        """
        pass
    
    def encode_vector(self, vector: Union[List[float], np.ndarray]) -> str:
        """
        Encodes a numerical vector using the configured uubed encoding method.

        This utility method is used internally by concrete connector implementations
        to convert numerical embedding vectors into their uubed string representation
        before insertion into the database. It leverages the `uubed.api.encode` function,
        which handles various input types and performs necessary internal normalizations
        (e.g., converting float arrays to uint8 bytes).

        Args:
            vector (Union[List[float], np.ndarray]): The numerical embedding vector to encode.
                                                      Can be a `List[float]` or a `np.ndarray`.

        Returns:
            str: The uubed-encoded string representation of the vector.

        Raises:
            UubedValidationError: If the input `vector` is invalid or cannot be encoded.
            UubedEncodingError: If an error occurs during the uubed encoding process.
        """
        # The `uubed.api.encode` function is robust and handles various input types
        # and normalizations (e.g., float to uint8). We pass the vector directly to it.
        try:
            return encode(vector, method=self.encoding_method, **self.encoding_kwargs)
        except UubedValidationError as e:
            raise UubedValidationError(f"Failed to encode vector for database insertion: {e}") from e
        except UubedEncodingError as e:
            raise UubedEncodingError(f"Failed to encode vector for database insertion: {e}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during vector encoding: {e}") from e


class PineconeConnector(VectorDBConnector):
    """
    Connector for Pinecone vector database, integrating uubed encoding.

    This concrete implementation of `VectorDBConnector` provides the necessary
    methods to interact with Pinecone. It automatically handles the encoding
    of numerical embedding vectors into uubed's position-safe string format
    and stores this string in the metadata of each vector within Pinecone.
    This enables compact representation and potential use in hybrid search scenarios.

    **Requirements:**
    - The `pinecone-client` Python package must be installed (`pip install pinecone-client`).
    - A running Pinecone instance and valid API credentials are required for connection.

    **Key Features:**
    - **Automatic uubed Encoding:** Embeddings are automatically encoded using the
      configured `encoding_method` and `encoding_kwargs` before insertion.
    - **Metadata Storage:** The uubed-encoded string and the encoding method used
      are stored in the vector's metadata under the keys `uubed_encoded` and
      `encoding_method` respectively.
    - **Batch Insertion:** Supports efficient batch insertion of vectors.
    - **Standard Search Interface:** Provides a `search` method compatible with the
      `VectorDBConnector` abstract interface.

    **Example Usage:**
    ```python
    import numpy as np
    from uubed.integrations.vectordb import PineconeConnector

    # Initialize the connector with a specific uubed encoding method
    connector = PineconeConnector(encoding_method="shq64")

    # Connect to Pinecone (replace with your actual API key and environment)
    # Ensure you have a Pinecone API key and environment configured.
    try:
        connector.connect(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
        print("Successfully connected to Pinecone.")
    except ImportError:
        print("Pinecone client not installed. Please run: pip install pinecone-client")
        exit()
    except Exception as e:
        print(f"Failed to connect to Pinecone: {e}")
        exit()

    # Create a new Pinecone index (collection) if it doesn't exist
    index_name = "my-uubed-embeddings"
    vector_dimension = 768 # Example dimension for your embeddings
    try:
        index = connector.create_collection(index_name, dimension=vector_dimension, metric="cosine")
        print(f"Index '{index_name}' ready.")
    except Exception as e:
        print(f"Failed to create/get index: {e}")
        exit()

    # Generate some dummy vectors and metadata for insertion
    vectors_to_insert = [np.random.rand(vector_dimension).tolist() for _ in range(10)]
    metadatas_to_insert = [{'source': f'doc_{i}'} for i in range(10)]
    ids_to_insert = [f'id_{i}' for i in range(10)]

    # Insert vectors with automatic uubed encoding in metadata
    print("Inserting vectors...")
    connector.insert_vectors(vectors_to_insert, metadata=metadatas_to_insert, ids=ids_to_insert)
    print(f"Inserted {len(vectors_to_insert)} vectors.")

    # Search for similar vectors
    query_vector = np.random.rand(vector_dimension).tolist()
    print("Searching for similar vectors...")
    results = connector.search(query_vector, top_k=3)

    print("Search Results:")
    for res in results:
        print(f"  ID: {res['id']}, Score: {res['score']:.4f}, Encoded: {res['uubed_encoded'][:10]}..., Source: {res['metadata'].get('source')}")
    ```
    """
    
    def __init__(self, encoding_method: EncodingMethod = "shq64", **encoding_kwargs: Any):
        """
        Initializes the PineconeConnector.

        Args:
            encoding_method (EncodingMethod): The uubed encoding method to use for vectors.
                                              Defaults to "shq64". This method will be used
                                              when `encode_vector` is called internally.
            **encoding_kwargs (Any): Additional keyword arguments to pass to the uubed encoder
                                     during vector encoding (e.g., `planes` for SimHash).
        """
        super().__init__(encoding_method, **encoding_kwargs)
        self.client: Optional[Any] = None  # Pinecone client instance, initialized upon connection.
        self.index: Optional[Any] = None   # Pinecone index object, initialized upon collection creation.
    
    def connect(self, api_key: str, environment: str, **kwargs: Any) -> bool:
        """
        Connects to the Pinecone service.

        This method initializes the Pinecone client with the provided API key and environment.
        It raises an `ImportError` if the `pinecone-client` package is not found.

        Args:
            api_key (str): Your Pinecone API key.
            environment (str): The Pinecone environment (e.g., "us-east-1-aws").
            **kwargs (Any): Additional keyword arguments to pass to `pinecone.init`
                            (e.g., `project_name`).

        Returns:
            bool: `True` if the connection is successfully established.

        Raises:
            ImportError: If the `pinecone-client` package is not installed.
            Exception: Any other exceptions raised by the Pinecone client during initialization.
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone client required: `pip install pinecone-client`")
        
        # Initialize the Pinecone connection. This sets up the global Pinecone context.
        pinecone.init(api_key=api_key, environment=environment, **kwargs)
        self.client = pinecone  # Store the pinecone module as the client.
        return True
    
    def create_collection(self, name: str, dimension: int, metric: str = "cosine", **kwargs: Any) -> Any:
        """
        Creates a new Pinecone index (collection) if it doesn't already exist.

        If an index with the given `name` already exists, this method will simply
        connect to it. Otherwise, it will create a new index with the specified
        `dimension` and `metric`.

        Args:
            name (str): The name of the index to create or connect to.
            dimension (int): The dimensionality of the vectors to be stored in the index.
                             This is a required parameter for creating a new index.
            metric (str): The similarity metric to use for the index (e.g., "cosine",
                          "euclidean", "dotproduct"). Defaults to "cosine".
            **kwargs (Any): Additional keyword arguments to pass to `pinecone.create_index`
                            (e.g., `shards`, `replicas`, `pod_type`).

        Returns:
            Any: The Pinecone `Index` object representing the created or connected index.

        Raises:
            RuntimeError: If the connector is not connected to Pinecone (i.e., `connect()`
                          has not been called successfully).
            Exception: Any other exceptions raised by the Pinecone client during index creation.
        """
        if not self.client:
            raise RuntimeError("Not connected to Pinecone. Call `connect()` first.")
        
        # Check if the index already exists to avoid recreation errors and unnecessary operations.
        if name not in self.client.list_indexes():
            self.client.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                **kwargs
            )
        
        # Connect to the specific index. This returns an Index object that can be used for data operations.
        self.index = self.client.Index(name)
        return self.index
    
    def insert_vectors(
        self,
        vectors: List[Union[List[float], np.ndarray]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Inserts vectors into the Pinecone index, with uubed encoding stored in metadata.

        This method prepares the vectors for insertion by generating unique IDs (if not
        provided) and encoding each numerical vector into a uubed string. The uubed-encoded
        string and the encoding method are then added to the vector's metadata before
        batch upserting them into Pinecone.

        Args:
            vectors (List[Union[List[float], np.ndarray]]): A list of numerical embedding vectors to insert.
                                                              Each vector can be a `List[float]` or a `np.ndarray`.
            metadata (Optional[List[Dict[str, Any]]]): An optional list of metadata dictionaries,
                                                        one for each vector. If provided, its length
                                                        must match the `vectors` list. Defaults to `None`.
            batch_size (int): The number of vectors to upsert in a single batch. Larger batches
                              can improve performance but require more memory. Defaults to 100.
            ids (Optional[List[str]]): An optional list of unique string IDs for each vector.
                                       If `None`, sequential IDs (e.g., "vec_0", "vec_1") will be generated.
                                       If provided, its length must match the `vectors` list.
        
        Raises:
            RuntimeError: If no Pinecone index is selected (i.e., `create_collection()`
                          has not been called successfully).
            ValueError: If the number of provided `ids` or `metadata` dictionaries does not
                        match the number of `vectors`.
            Exception: Any other exceptions raised by the Pinecone client during the upsert operation.
        """
        if not self.index:
            raise RuntimeError("No Pinecone index selected. Call `create_collection()` first.")
        
        # Initialize metadata list if not provided, ensuring it matches the vectors' length.
        if metadata is None:
            metadata = [{} for _ in vectors]
        
        # Validate that the lengths of input lists are consistent.
        if ids is not None and len(ids) != len(vectors):
            raise ValueError("Number of provided IDs must match the number of vectors.")
        if len(metadata) != len(vectors):
            raise ValueError("Number of provided metadata dictionaries must match the number of vectors.")

        # Process vectors in batches for efficient upsert operations.
        # Pinecone's upsert method is optimized for batching.
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids else None
            
            # Prepare data in the format expected by Pinecone's upsert method:
            # List of (id, vector_values, metadata_dict) tuples.
            upsert_data: List[Tuple[str, List[float], Dict[str, Any]]] = []
            for j, (vector, meta) in enumerate(zip(batch_vectors, batch_metadata)):
                # Generate a unique ID if not explicitly provided for the current batch item.
                vector_id: str = batch_ids[j] if batch_ids else f"vec_{i+j}"
                
                # Encode the numerical vector into a uubed string using the configured method.
                encoded_uubed: str = self.encode_vector(vector)
                
                # Create a copy of the original metadata and add the uubed-encoded string
                # and the encoding method to it. This ensures the original metadata is not modified.
                meta_with_encoding: Dict[str, Any] = meta.copy()
                meta_with_encoding["uubed_encoded"] = encoded_uubed
                meta_with_encoding["encoding_method"] = self.encoding_method
                
                # Ensure the vector is a list of floats, as required by Pinecone's API.
                vector_list: List[float] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                
                upsert_data.append((vector_id, vector_list, meta_with_encoding))
            
            # Perform the batch upsert operation to Pinecone.
            self.index.upsert(vectors=upsert_data)
    
    def search(self, query_vector: Union[List[float], np.ndarray], top_k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Searches the Pinecone index for similar vectors.

        This method performs a similarity search using the provided `query_vector`
        and retrieves the `top_k` most similar results. It ensures that metadata,
        including the uubed-encoded string, is included in the results.

        Args:
            query_vector (Union[List[float], np.ndarray]): The numerical query embedding vector.
                                                              Can be a `List[float]` or a `np.ndarray`.
            top_k (int): The number of top similar results to retrieve. Defaults to 10.
            **kwargs (Any): Additional keyword arguments to pass to Pinecone's `index.query` method
                            (e.g., `filter` for metadata filtering, `include_values` to retrieve
                            the original vector values, `include_metadata`).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a search result.
                                  Each result includes the `id`, `score`, `metadata` (original and uubed-related),
                                  and the `uubed_encoded` string for convenience.

        Raises:
            RuntimeError: If no Pinecone index is selected (i.e., `create_collection()`
                          has not been called successfully).
            Exception: Any other exceptions raised by the Pinecone client during the query operation.
        """
        if not self.index:
            raise RuntimeError("No Pinecone index selected. Call `create_collection()` first.")
        
        # Ensure the query vector is a list of floats, as required by Pinecone's API.
        query_vector_list: List[float] = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        # Perform the query against the Pinecone index.
        # `include_metadata=True` is crucial to retrieve the stored uubed_encoded string.
        results = self.index.query(
            vector=query_vector_list,
            top_k=top_k,
            include_metadata=True, # Always include metadata to retrieve uubed_encoded string.
            **kwargs
        )
        
        # Format the results into a list of dictionaries for consistent output.
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {}),
                "uubed_encoded": match.get("metadata", {}).get("uubed_encoded")
            }
            for match in results["matches"]
        ]


class WeaviateConnector(VectorDBConnector):
    """
    Connector for Weaviate vector database, integrating uubed encoding.

    This connector allows storing data objects with their vectors and properties
    in Weaviate, automatically adding a uubed-encoded string to the properties.

    Requires: `pip install weaviate-client`

    Example:
        >>> connector = WeaviateConnector(encoding_method="t8q64", k=16)
        >>> connector.connect(url="http://localhost:8080")
        >>> connector.create_collection("Documents", properties=["content", "title"])
        >>> 
        >>> # Insert data objects with vectors and properties
        >>> vectors_to_insert = [np.random.randn(768) for _ in range(10)]
        >>> properties_to_insert = [{'content': f'doc {i}', 'title': f'Title {i}'} for i in range(10)]
        >>> connector.insert_vectors(vectors_to_insert, properties_to_insert)
        >>> 
        >>> # Search for similar vectors
        >>> query = np.random.randn(768)
        >>> results = connector.search(query, top_k=3)
        >>> for res in results:
        ...     print(f"Content: {res['properties']['content']}, Certainty: {res['certainty']:.4f}")
    """
    
    def __init__(self, encoding_method: EncodingMethod = "t8q64", **encoding_kwargs: Any):
        """
        Initializes the WeaviateConnector.

        Args:
            encoding_method (EncodingMethod): The uubed encoding method to use. Defaults to "t8q64".
            **encoding_kwargs (Any): Additional keyword arguments for the uubed encoder.
        """
        super().__init__(encoding_method, **encoding_kwargs)
        self.client: Optional[Any] = None # Weaviate client instance.
        self.class_name: Optional[str] = None # Currently selected class name.
    
    def connect(self, url: str = "http://localhost:8080", **kwargs: Any) -> bool:
        """
        Connects to the Weaviate instance.

        Args:
            url (str): The URL of the Weaviate instance. Defaults to "http://localhost:8080".
            **kwargs (Any): Additional keyword arguments for `weaviate.Client`.

        Returns:
            bool: `True` if the connection is successful and the client is ready.

        Raises:
            ImportError: If the `weaviate-client` package is not installed.
        """
        try:
            import weaviate
        except ImportError:
            raise ImportError("Weaviate client required: `pip install weaviate-client`")
        
        self.client = weaviate.Client(url, **kwargs)
        return self.client.is_ready()
    
    def create_collection(self, name: str, properties: List[str], **kwargs: Any) -> bool:
        """
        Creates a new Weaviate class (collection) if it doesn't already exist.

        Args:
            name (str): The name of the Weaviate class to create.
            properties (List[str]): A list of property names (strings) for the class.
            **kwargs (Any): Additional keyword arguments for the class schema (e.g., `vectorizer`).

        Returns:
            bool: `True` if the class is created or already exists.

        Raises:
            RuntimeError: If not connected to Weaviate.
        """
        if not self.client:
            raise RuntimeError("Not connected to Weaviate. Call `connect()` first.")
        
        # Define the schema for the new class, including uubed-specific properties.
        class_schema: Dict[str, Any] = {
            "class": name,
            "properties": [
                {"name": prop, "dataType": ["text"]} for prop in properties
            ] + [
                {"name": "uubed_encoded", "dataType": ["text"]},
                {"name": "encoding_method", "dataType": ["text"]},
            ],
            **kwargs
        }
        
        # Check if the class already exists before attempting to create it.
        if not self.client.schema.exists(name):
            self.client.schema.create_class(class_schema)
        
        self.class_name = name
        return True
    
    def insert_vectors(
        self,
        vectors: List[Union[List[float], np.ndarray]],
        properties: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Inserts vectors and their associated properties into the Weaviate database.

        Each vector's uubed-encoded string is automatically added to its properties.

        Args:
            vectors (List[Union[List[float], np.ndarray]]): A list of numerical embedding vectors.
            properties (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                contains the properties for a corresponding vector.
            batch_size (int): The number of data objects to insert in a single batch. Defaults to 100.

        Raises:
            RuntimeError: If not connected to Weaviate or no class is selected.
            ValueError: If the number of vectors does not match the number of properties.
        """
        if not self.client or not self.class_name:
            raise RuntimeError("Not connected to Weaviate or no class selected. Call `connect()` and `create_collection()` first.")
        
        if len(vectors) != len(properties):
            raise ValueError("Number of vectors must match number of properties.")

        # Use Weaviate's batch context manager for efficient insertions.
        with self.client.batch as batch:
            batch.batch_size = batch_size
            
            for vector, props in zip(vectors, properties):
                # Encode the vector using uubed and add to properties.
                encoded_uubed: str = self.encode_vector(vector)
                props_with_encoding: Dict[str, Any] = props.copy()
                props_with_encoding["uubed_encoded"] = encoded_uubed
                props_with_encoding["encoding_method"] = self.encoding_method
                
                # Ensure vector is a list of floats for Weaviate.
                vector_list: List[float] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                
                batch.add_data_object(
                    data_object=props_with_encoding,
                    class_name=self.class_name,
                    vector=vector_list
                )
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the Weaviate database for similar vectors.

        Args:
            query_vector (Union[List[float], np.ndarray]): The numerical query embedding vector.
            top_k (int): The number of top similar results to retrieve. Defaults to 10.
            where_filter (Optional[Dict[str, Any]]): An optional Weaviate `where` filter to apply to the search.
                                                    Defaults to `None`.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a search result
                                  with properties, certainty, distance, and the uubed-encoded string.

        Raises:
            RuntimeError: If not connected to Weaviate or no class is selected.
        """
        if not self.client or not self.class_name:
            raise RuntimeError("Not connected to Weaviate or no class selected. Call `connect()` and `create_collection()` first.")
        
        # Ensure query vector is a list of floats for Weaviate.
        query_vector_list: List[float] = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        # Build the Weaviate query.
        query = self.client.query.get(self.class_name).with_near_vector({
            "vector": query_vector_list
        }).with_limit(top_k).with_additional(["certainty", "distance"])
        
        if where_filter:
            query = query.with_where(where_filter)
        
        results = query.do()
        
        # Extract and format the search results.
        return [
            {
                "properties": item,
                "certainty": item.get("_additional", {}).get("certainty"),
                "distance": item.get("_additional", {}).get("distance"),
                "uubed_encoded": item.get("uubed_encoded")
            }
            for item in results["data"]["Get"][self.class_name]
        ]


class QdrantConnector(VectorDBConnector):
    """
    Connector for Qdrant vector database, integrating uubed encoding.

    This connector allows storing and searching vectors in Qdrant, automatically
    adding a uubed-encoded string to the payload of each point.

    Requires: `pip install qdrant-client`

    Example:
        >>> connector = QdrantConnector(encoding_method="zoq64")
        >>> connector.connect(host="localhost", port=6333)
        >>> connector.create_collection("my_embeddings", vector_size=768)
        >>> 
        >>> # Insert vectors with automatic uubed encoding in payload
        >>> vectors_to_insert = [np.random.randn(768) for _ in range(100)]
        >>> payloads_to_insert = [{'source': 'document', 'page': i} for i in range(100)]
        >>> connector.insert_vectors(vectors_to_insert, payloads_to_insert)
        >>> 
        >>> # Search for similar vectors
        >>> query = np.random.randn(768)
        >>> results = connector.search(query, top_k=5)
        >>> for res in results:
        ...     print(f"ID: {res['id']}, Score: {res['score']:.4f}, Source: {res['payload']['source']}")
    """
    
    def __init__(self, encoding_method: EncodingMethod = "zoq64", **encoding_kwargs: Any):
        """
        Initializes the QdrantConnector.

        Args:
            encoding_method (EncodingMethod): The uubed encoding method to use. Defaults to "zoq64".
            **encoding_kwargs (Any): Additional keyword arguments for the uubed encoder.
        """
        super().__init__(encoding_method, **encoding_kwargs)
        self.client: Optional[Any] = None # Qdrant client instance.
        self.collection_name: Optional[str] = None # Currently selected collection name.
    
    def connect(self, host: str = "localhost", port: int = 6333, **kwargs: Any) -> bool:
        """
        Connects to the Qdrant service.

        Args:
            host (str): The host address of the Qdrant instance. Defaults to "localhost".
            port (int): The port of the Qdrant instance. Defaults to 6333.
            **kwargs (Any): Additional keyword arguments for `QdrantClient`.

        Returns:
            bool: `True` if the connection is successful.

        Raises:
            ImportError: If the `qdrant-client` package is not installed.
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("Qdrant client required: `pip install qdrant-client`")
        
        self.client = QdrantClient(host=host, port=port, **kwargs)
        return True
    
    def create_collection(self, name: str, vector_size: int, distance: str = "Cosine", **kwargs: Any) -> bool:
        """
        Creates a new Qdrant collection if it doesn't already exist.

        Args:
            name (str): The name of the collection to create.
            vector_size (int): The dimensionality of the vectors to be stored in the collection.
            distance (str): The similarity metric to use ("Cosine", "Euclidean", "Dot"). Defaults to "Cosine".
            **kwargs (Any): Additional keyword arguments for `client.recreate_collection`.

        Returns:
            bool: `True` if the collection is created or already exists.

        Raises:
            RuntimeError: If not connected to Qdrant.
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant. Call `connect()` first.")
        
        from qdrant_client.http.models import VectorParams, Distance
        
        # Map string distance names to Qdrant Distance enums.
        distance_map: Dict[str, Distance] = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        
        # Use `recreate_collection` for simplicity in examples, but be aware it deletes existing data.
        # For production, consider `create_collection` (which fails if exists) or `update_collection`.
        try:
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE) # Default to Cosine if unknown.
                ),
                **kwargs
            )
        except Exception:
            # If recreate_collection fails (e.g., due to network issues, not just existence),
            # we assume it might already exist or handle it silently for this example.
            # In a real application, more specific error handling would be needed.
            pass
        
        self.collection_name = name
        return True
    
    def insert_vectors(
        self,
        vectors: List[Union[List[float], np.ndarray]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        ids: Optional[List[Union[int, str]]] = None
    ) -> None:
        """
        Inserts vectors into the Qdrant collection, with uubed encoding stored in payloads.

        Args:
            vectors (List[Union[List[float], np.ndarray]]): A list of numerical embedding vectors.
            payloads (Optional[List[Dict[str, Any]]]): An optional list of payload dictionaries,
                                                        one for each vector. Defaults to `None`.
            batch_size (int): The number of points to upsert in a single batch. Defaults to 100.
            ids (Optional[List[Union[int, str]]]): Optional list of unique IDs for each point. If `None`,
                                                   sequential integer IDs will be generated.

        Raises:
            RuntimeError: If not connected to Qdrant or no collection is selected.
            ValueError: If the number of provided IDs or payloads does not match the number of vectors.
        """
        if not self.client or not self.collection_name:
            raise RuntimeError("Not connected to Qdrant or no collection selected. Call `connect()` and `create_collection()` first.")
        
        from qdrant_client.http.models import PointStruct
        
        if payloads is None:
            payloads = [{} for _ in vectors]
        
        if ids is not None and len(ids) != len(vectors):
            raise ValueError("Number of provided IDs must match the number of vectors.")

        # Process vectors in batches for efficient upsert operations.
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids else None
            
            points: List[PointStruct] = []
            for j, (vector, payload) in enumerate(zip(batch_vectors, batch_payloads)):
                # Generate a unique ID if not provided.
                point_id: Union[int, str] = batch_ids[j] if batch_ids else i + j
                
                # Encode the vector using uubed and add to payload.
                encoded_uubed: str = self.encode_vector(vector)
                payload_with_encoding: Dict[str, Any] = payload.copy()
                payload_with_encoding["uubed_encoded"] = encoded_uubed
                payload_with_encoding["encoding_method"] = self.encoding_method
                
                # Ensure vector is a list of floats for Qdrant.
                vector_list: List[float] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector_list,
                    payload=payload_with_encoding
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Searches the Qdrant collection for similar vectors.

        Args:
            query_vector (Union[List[float], np.ndarray]): The numerical query embedding vector.
            top_k (int): The number of top similar results to retrieve. Defaults to 10.
            score_threshold (Optional[float]): A minimum score threshold for results. Defaults to `None`.
            **kwargs (Any): Additional keyword arguments for `client.search` (e.g., `query_filter`).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a search result
                                  with ID, score, payload, and the uubed-encoded string.

        Raises:
            RuntimeError: If not connected to Qdrant or no collection is selected.
        """
        if not self.client or not self.collection_name:
            raise RuntimeError("Not connected to Qdrant or no collection selected. Call `connect()` and `create_collection()` first.")
        
        # Ensure query vector is a list of floats for Qdrant.
        query_vector_list: List[float] = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector_list,
            limit=top_k,
            score_threshold=score_threshold,
            **kwargs
        )
        
        # Extract and format the search results.
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                "uubed_encoded": hit.payload.get("uubed_encoded")
            }
            for hit in results
        ]


class ChromaConnector(VectorDBConnector):
    """
    Connector for ChromaDB, integrating uubed encoding.

    This connector allows storing documents and their vectors in ChromaDB,
    automatically adding a uubed-encoded string to the metadata.

    Requires: `pip install chromadb`

    Example:
        >>> connector = ChromaConnector(encoding_method="eq64")
        >>> connector.connect() # For in-memory client
        >>> # connector.connect(path="./chroma_db") # For persistent client
        >>> connector.create_collection("my_documents")
        >>> 
        >>> # Insert documents with vectors and metadata
        >>> docs_to_insert = ["This is doc 1", "This is doc 2"]
        >>> vectors_to_insert = [np.random.randn(768), np.random.randn(768)]
        >>> metadatas_to_insert = [{'source': 'web'}, {'source': 'pdf'}]
        >>> ids_to_insert = ["doc1", "doc2"]
        >>> connector.insert_vectors(vectors_to_insert, docs_to_insert, metadatas_to_insert, ids_to_insert)
        >>> 
        >>> # Search for similar vectors
        >>> query = np.random.randn(768)
        >>> results = connector.search(query, top_k=1)
        >>> for res in results:
        ...     print(f"Document: {res['document']}, Distance: {res['distance']:.4f}")
    """
    
    def __init__(self, encoding_method: EncodingMethod = "eq64", **encoding_kwargs: Any):
        """
        Initializes the ChromaConnector.

        Args:
            encoding_method (EncodingMethod): The uubed encoding method to use. Defaults to "eq64".
            **encoding_kwargs (Any): Additional keyword arguments for the uubed encoder.
        """
        super().__init__(encoding_method, **encoding_kwargs)
        self.client: Optional[Any] = None # ChromaDB client instance.
        self.collection: Optional[Any] = None # ChromaDB collection object.
    
    def connect(self, path: Optional[str] = None, **kwargs: Any) -> bool:
        """
        Connects to the ChromaDB instance.

        Args:
            path (Optional[str]): The path to the ChromaDB directory for a persistent client.
                                  If `None`, an in-memory client is used. Defaults to `None`.
            **kwargs (Any): Additional keyword arguments for `chromadb.PersistentClient` or `chromadb.Client`.

        Returns:
            bool: `True` if the connection is successful.

        Raises:
            ImportError: If the `chromadb` package is not installed.
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaDB required: `pip install chromadb`")
        
        if path:
            self.client = chromadb.PersistentClient(path=path, **kwargs)
        else:
            self.client = chromadb.Client(**kwargs)
        
        return True
    
    def create_collection(self, name: str, **kwargs: Any) -> Any:
        """
        Creates a new ChromaDB collection if it doesn't already exist.

        Args:
            name (str): The name of the collection to create.
            **kwargs (Any): Additional keyword arguments for `client.create_collection`.

        Returns:
            Any: The ChromaDB `Collection` object.

        Raises:
            RuntimeError: If not connected to ChromaDB.
        """
        if not self.client:
            raise RuntimeError("Not connected to ChromaDB. Call `connect()` first.")
        
        try:
            self.collection = self.client.create_collection(name=name, **kwargs)
        except Exception:
            # If create_collection fails (e.g., due to collection already existing),
            # try to get the existing collection.
            self.collection = self.client.get_collection(name=name)
        
        return self.collection
    
    def insert_vectors(
        self,
        vectors: List[Union[List[float], np.ndarray]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Inserts vectors, documents, and metadata into the ChromaDB collection.

        Each vector's uubed-encoded string is automatically added to its metadata.

        Args:
            vectors (List[Union[List[float], np.ndarray]]): A list of numerical embedding vectors.
            documents (List[str]): A list of document strings corresponding to the vectors.
            metadatas (Optional[List[Dict[str, Any]]]): An optional list of metadata dictionaries,
                                                         one for each vector. Defaults to `None`.
            ids (Optional[List[str]]): Optional list of unique IDs for each entry. If `None`,
                                       sequential IDs (`doc_0`, `doc_1`, ...) will be generated.

        Raises:
            RuntimeError: If no collection is selected.
            ValueError: If the number of vectors, documents, metadatas, or IDs do not match.
        """
        if not self.collection:
            raise RuntimeError("No ChromaDB collection selected. Call `create_collection()` first.")
        
        if metadatas is None:
            metadatas = [{} for _ in vectors]
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(vectors))]
        
        # Validate consistent lengths across all input lists.
        if not (len(vectors) == len(documents) == len(metadatas) == len(ids)):
            raise ValueError("All input lists (vectors, documents, metadatas, ids) must have the same length.")

        enhanced_metadatas: List[Dict[str, Any]] = []
        enhanced_vectors: List[List[float]] = []
        
        for vector, metadata in zip(vectors, metadatas):
            # Encode the vector using uubed and add to metadata.
            encoded_uubed: str = self.encode_vector(vector)
            enhanced_metadata: Dict[str, Any] = metadata.copy()
            enhanced_metadata["uubed_encoded"] = encoded_uubed
            enhanced_metadata["encoding_method"] = self.encoding_method
            enhanced_metadatas.append(enhanced_metadata)
            
            # Ensure vector is a list of floats for ChromaDB.
            vector_list: List[float] = vector.tolist() if isinstance(vector, np.ndarray) else vector
            enhanced_vectors.append(vector_list)
        
        self.collection.add(
            embeddings=enhanced_vectors,
            documents=documents,
            metadatas=enhanced_metadatas,
            ids=ids
        )
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Searches the ChromaDB collection for similar vectors.

        Args:
            query_vector (Union[List[float], np.ndarray]): The numerical query embedding vector.
            top_k (int): The number of top similar results to retrieve. Defaults to 10.
            where (Optional[Dict[str, Any]]): An optional ChromaDB `where` filter to apply to the search.
                                             Defaults to `None`.
            **kwargs (Any): Additional keyword arguments for `collection.query`.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a search result
                                  with ID, document, metadata, distance, and the uubed-encoded string.

        Raises:
            RuntimeError: If no collection is selected.
        """
        if not self.collection:
            raise RuntimeError("No ChromaDB collection selected. Call `create_collection()` first.")
        
        # Ensure query vector is a list of floats for ChromaDB.
        query_vector_list: List[float] = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        results = self.collection.query(
            query_embeddings=[query_vector_list],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"], # Always include these for comprehensive results.
            **kwargs
        )
        
        # Extract and format the search results.
        # ChromaDB returns results in a nested list structure, so we flatten it.
        return [
            {
                "id": id_,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "uubed_encoded": meta.get("uubed_encoded") # Extract the uubed_encoded string.
            }
            for id_, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]


def get_connector(db_type: str, **kwargs: Any) -> VectorDBConnector:
    """
    Factory function to retrieve an appropriate `VectorDBConnector` instance.

    This function simplifies the instantiation of vector database connectors
    by providing a centralized entry point.

    Args:
        db_type (str): The type of database connector to retrieve (e.g., "pinecone", "weaviate", "qdrant", "chroma").
        **kwargs (Any): Additional keyword arguments to pass to the constructor of the specific connector.

    Returns:
        VectorDBConnector: An instance of the requested `VectorDBConnector` subclass.

    Raises:
        ValueError: If an unknown or unsupported `db_type` is specified.
    """
    connectors: Dict[str, type[VectorDBConnector]] = {
        "pinecone": PineconeConnector,
        "weaviate": WeaviateConnector,
        "qdrant": QdrantConnector,
        "chroma": ChromaConnector,
    }
    
    db_type_lower = db_type.lower()
    if db_type_lower not in connectors:
        raise ValueError(f"Unknown or unsupported database type: '{db_type}'. Expected one of: {', '.join(sorted(connectors.keys()))}.")
    
    return connectors[db_type_lower](**kwargs)
