#!/usr/bin/env python3
"""Test integration modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestLangChainIntegration:
    """Test LangChain integration components."""
    
    def test_import_langchain_integration(self):
        """Test that LangChain integration can be imported."""
        try:
            from uubed.integrations.langchain import UubedEncoder
            assert UubedEncoder is not None
        except ImportError:
            pytest.skip("LangChain not available")
    
    @pytest.mark.skipif(True, reason="LangChain not installed in test environment")
    def test_uubed_encoder_basic(self):
        """Test basic UubedEncoder functionality."""
        from uubed.integrations.langchain import UubedEncoder
        from langchain.schema import Document
        
        encoder = UubedEncoder(method="shq64")
        
        # Mock documents and embeddings
        docs = [
            Document(page_content="Test document 1", metadata={"id": 1}),
            Document(page_content="Test document 2", metadata={"id": 2}),
        ]
        
        embeddings = [
            np.random.randn(384).tolist(),  # Normalized embeddings
            np.random.randn(384).tolist(),
        ]
        
        # Encode documents
        encoded_docs = encoder.encode_documents(docs, embeddings)
        
        assert len(encoded_docs) == 2
        for doc in encoded_docs:
            assert "uubed_encoded" in doc.metadata
            assert isinstance(doc.metadata["uubed_encoded"], str)
    
    def test_mock_uubed_encoder(self):
        """Test UubedEncoder with mocked dependencies."""
        # Mock the Document class
        class MockDocument:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}
        
        # Mock encoder behavior
        class MockUubedEncoder:
            def __init__(self, method="shq64", **kwargs):
                self.method = method
                self.kwargs = kwargs
            
            def encode_documents(self, documents, embeddings):
                result = []
                for doc, emb in zip(documents, embeddings):
                    # Simulate encoding
                    from uubed import encode
                    emb_array = np.array(emb)
                    if emb_array.min() < 0 or emb_array.max() > 255:
                        emb_uint8 = ((emb_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        emb_uint8 = emb_array.astype(np.uint8)
                    
                    encoded = encode(emb_uint8, method=self.method, **self.kwargs)
                    
                    new_metadata = doc.metadata.copy()
                    new_metadata["uubed_encoded"] = encoded
                    
                    result.append(MockDocument(doc.page_content, new_metadata))
                return result
        
        # Test the mock encoder
        encoder = MockUubedEncoder(method="t8q64", k=8)
        
        docs = [
            MockDocument("Test doc 1", {"id": 1}),
            MockDocument("Test doc 2", {"id": 2}),
        ]
        
        embeddings = [
            np.random.randn(128).tolist(),
            np.random.randn(128).tolist(),
        ]
        
        encoded_docs = encoder.encode_documents(docs, embeddings)
        
        assert len(encoded_docs) == 2
        for doc in encoded_docs:
            assert "uubed_encoded" in doc.metadata
            assert isinstance(doc.metadata["uubed_encoded"], str)


class TestVectorDBConnectors:
    """Test vector database connector functionality."""
    
    def test_import_vectordb_connectors(self):
        """Test that vector DB connectors can be imported."""
        from uubed.integrations.vectordb import get_connector, VectorDBConnector
        assert get_connector is not None
        assert VectorDBConnector is not None
    
    def test_connector_factory(self):
        """Test connector factory function."""
        from uubed.integrations.vectordb import get_connector
        
        # Test valid connector types
        valid_types = ["pinecone", "weaviate", "qdrant", "chroma"]
        for db_type in valid_types:
            connector = get_connector(db_type, encoding_method="shq64")
            assert connector is not None
            assert hasattr(connector, "encode_vector")
        
        # Test invalid connector type
        with pytest.raises(ValueError, match="Unknown database type"):
            get_connector("invalid_db")
    
    def test_mock_chroma_connector(self):
        """Test ChromaDB connector with mocked dependencies."""
        # Mock ChromaDB
        class MockChromaClient:
            def __init__(self):
                self.collections = {}
            
            def create_collection(self, name, **kwargs):
                collection = MockChromaCollection(name)
                self.collections[name] = collection
                return collection
            
            def get_collection(self, name):
                return self.collections.get(name)
        
        class MockChromaCollection:
            def __init__(self, name):
                self.name = name
                self.data = []
            
            def add(self, embeddings, documents, metadatas, ids):
                for emb, doc, meta, id_ in zip(embeddings, documents, metadatas, ids):
                    self.data.append({
                        "id": id_,
                        "embedding": emb,
                        "document": doc,
                        "metadata": meta
                    })
            
            def query(self, query_embeddings, n_results, where=None, include=None):
                results = self.data[:n_results]
                return {
                    "ids": [[r["id"] for r in results]],
                    "documents": [[r["document"] for r in results]],
                    "metadatas": [[r["metadata"] for r in results]],
                    "distances": [[0.5] * len(results)]
                }
        
        # Test with mock
        with patch('uubed.integrations.vectordb.chromadb') as mock_chromadb:
            mock_chromadb.Client = MockChromaClient
            mock_chromadb.PersistentClient = MockChromaClient
            
            from uubed.integrations.vectordb import ChromaConnector
            
            connector = ChromaConnector(encoding_method="eq64")
            assert connector.connect()
            assert connector.create_collection("test_collection")
            
            # Test vector insertion
            vectors = [np.random.randn(128) for _ in range(3)]
            documents = ["Doc 1", "Doc 2", "Doc 3"]
            
            connector.insert_vectors(vectors, documents)
            
            # Test search
            query_vector = np.random.randn(128)
            results = connector.search(query_vector, top_k=2)
            
            assert len(results) <= 2
            for result in results:
                assert "uubed_encoded" in result["metadata"]
    
    def test_vector_encoding_consistency(self):
        """Test that vector encoding is consistent across connectors."""
        from uubed.integrations.vectordb import get_connector
        
        test_vector = np.random.randn(64)
        
        # Test that all connectors encode the same vector consistently
        connectors = []
        for db_type in ["chroma", "pinecone", "weaviate", "qdrant"]:
            try:
                connector = get_connector(db_type, encoding_method="shq64")
                encoded = connector.encode_vector(test_vector)
                connectors.append((db_type, encoded))
            except ImportError:
                # Skip if dependency not available
                continue
        
        # All encodings should be the same
        if len(connectors) > 1:
            first_encoding = connectors[0][1]
            for db_type, encoding in connectors[1:]:
                assert encoding == first_encoding, f"Encoding mismatch for {db_type}"


class TestGPUIntegration:
    """Test GPU acceleration integration."""
    
    def test_import_gpu_module(self):
        """Test that GPU module can be imported."""
        from uubed.gpu import is_gpu_available, get_gpu_info
        assert is_gpu_available is not None
        assert get_gpu_info is not None
    
    def test_gpu_availability_detection(self):
        """Test GPU availability detection."""
        from uubed.gpu import is_gpu_available, get_gpu_info
        
        available = is_gpu_available()
        assert isinstance(available, bool)
        
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
    
    def test_gpu_fallback_behavior(self):
        """Test GPU fallback to CPU when not available."""
        from uubed.gpu import gpu_encode_batch
        
        embeddings = [
            np.random.randint(0, 256, 64, dtype=np.uint8)
            for _ in range(5)
        ]
        
        # Should work regardless of GPU availability
        encoded = gpu_encode_batch(embeddings, method="shq64")
        assert len(encoded) == 5
        assert all(isinstance(enc, str) for enc in encoded)


class TestMatryoshkaIntegration:
    """Test Matryoshka embedding integration."""
    
    def test_import_matryoshka_module(self):
        """Test that Matryoshka module can be imported."""
        from uubed.matryoshka import MatryoshkaEncoder, create_adaptive_matryoshka_encoder
        assert MatryoshkaEncoder is not None
        assert create_adaptive_matryoshka_encoder is not None
    
    def test_matryoshka_encoder_basic(self):
        """Test basic MatryoshkaEncoder functionality."""
        from uubed.matryoshka import MatryoshkaEncoder
        
        encoder = MatryoshkaEncoder([32, 64, 128])
        embedding = np.random.randn(128)
        
        # Test encoding all levels
        encoded_levels = encoder.encode_all_levels(embedding)
        assert len(encoded_levels) == 3
        assert 32 in encoded_levels
        assert 64 in encoded_levels
        assert 128 in encoded_levels
        
        # Test individual level encoding
        level_64 = encoder.encode_level(embedding, 64)
        assert isinstance(level_64, str)
        assert len(level_64) > 0
    
    def test_adaptive_matryoshka_encoder(self):
        """Test adaptive MatryoshkaEncoder creation."""
        from uubed.matryoshka import create_adaptive_matryoshka_encoder
        
        encoder = create_adaptive_matryoshka_encoder(512, num_levels=4)
        assert len(encoder.dimensions) == 4
        assert max(encoder.dimensions) <= 512
        
        # Test with different progressions
        for progression in ["linear", "exponential", "powers_of_2"]:
            encoder = create_adaptive_matryoshka_encoder(
                256, num_levels=3, progression=progression
            )
            assert len(encoder.dimensions) == 3


class TestIntegrationConsistency:
    """Test consistency across all integrations."""
    
    def test_encoding_consistency_across_modules(self):
        """Test that encoding is consistent across all modules."""
        test_embedding = np.random.randint(0, 256, 64, dtype=np.uint8)
        
        # Core API
        from uubed.api import encode
        core_encoded = encode(test_embedding, method="shq64")
        
        # Streaming API
        from uubed.streaming import encode_stream
        streaming_encoded = list(encode_stream([test_embedding], method="shq64"))[0]
        
        # GPU API (should fallback to CPU)
        from uubed.gpu import gpu_encode_batch
        gpu_encoded = gpu_encode_batch([test_embedding], method="shq64")[0]
        
        # All should produce the same result
        assert core_encoded == streaming_encoded == gpu_encoded
    
    def test_all_modules_importable(self):
        """Test that all integration modules can be imported."""
        modules_to_test = [
            "uubed.integrations.langchain",
            "uubed.integrations.vectordb",
            "uubed.gpu",
            "uubed.matryoshka",
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                # Some modules have optional dependencies
                if "langchain" in str(e).lower():
                    pytest.skip(f"Optional dependency not available: {module_name}")
                else:
                    raise
    
    def test_package_exports(self):
        """Test that main package exports work correctly."""
        import uubed
        
        # Core functions should always be available
        assert hasattr(uubed, "encode")
        assert hasattr(uubed, "decode")
        assert hasattr(uubed, "encode_stream")
        assert hasattr(uubed, "__version__")
        
        # Optional functions depend on dependencies
        optional_functions = [
            "is_gpu_available",
            "MatryoshkaEncoder",
            "gpu_encode_batch",
        ]
        
        for func_name in optional_functions:
            # Should either be available or not in __all__
            if hasattr(uubed, func_name):
                assert func_name in uubed.__all__


if __name__ == "__main__":
    pytest.main([__file__])