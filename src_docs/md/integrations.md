---
title: Integrations
---

# Integrations

uubed strings are printable ASCII and safe in any field that accepts text. The sections below show the idiomatic pattern for each system.

---

## Elasticsearch / OpenSearch

Store the encoded string as a `keyword` field alongside the raw vector field. uubed strings never share accidental substrings, so `term` and `prefix` queries work cleanly.

```python
from elasticsearch import Elasticsearch
import numpy as np
from uubed import encode

es = Elasticsearch("http://localhost:9200")

# Index mapping
es.indices.create(index="embeddings", body={
    "mappings": {
        "properties": {
            "vector": {"type": "dense_vector", "dims": 768},
            "eq64": {"type": "keyword"},        # lossless, for exact lookup
            "shq64": {"type": "keyword"},       # SimHash, for similarity filter
        }
    }
})

# Index a document
emb = np.random.randint(0, 256, 768, dtype=np.uint8)
es.index(index="embeddings", document={
    "vector": emb.tolist(),
    "eq64": encode(emb, method="eq64"),
    "shq64": encode(emb, method="shq64", planes=64),
})

# Exact keyword lookup — zero false positives thanks to position-safe encoding
es.search(index="embeddings", body={
    "query": {"term": {"eq64": encode(query_emb, method="eq64")}}
})
```

---

## pgvector (PostgreSQL)

Add a `TEXT` column alongside the `vector` column. Use the `shq64` value for a fast pre-filter before the expensive KNN scan.

```sql
-- Schema
ALTER TABLE embeddings
  ADD COLUMN eq64  TEXT,
  ADD COLUMN shq64 TEXT;

-- Index for similarity pre-filter
CREATE INDEX ON embeddings (shq64);
```

```python
import psycopg2
import numpy as np
from uubed import encode

conn = psycopg2.connect("dbname=mydb")
cur = conn.cursor()

emb = np.random.randint(0, 256, 768, dtype=np.uint8)
cur.execute(
    "INSERT INTO embeddings (vec, eq64, shq64) VALUES (%s, %s, %s)",
    (emb.tolist(), encode(emb, method="eq64"), encode(emb, method="shq64", planes=64))
)
conn.commit()
```

```sql
-- Two-phase retrieval: SimHash pre-filter → KNN re-rank
SELECT id, vec <-> query_vec AS dist
FROM embeddings
WHERE shq64 = $1          -- fast SimHash equality filter
ORDER BY dist
LIMIT 10;
```

---

## Apache Solr

Use a `string` field (not `text_general`) so tokenisation is skipped. Add a `copyField` to a `text_general` field only if you want prefix search.

```xml
<!-- schema.xml -->
<field name="eq64"  type="string" indexed="true" stored="true"/>
<field name="shq64" type="string" indexed="true" stored="true"/>
```

```python
import pysolr
import numpy as np
from uubed import encode

solr = pysolr.Solr("http://localhost:8983/solr/embeddings")

emb = np.random.randint(0, 256, 768, dtype=np.uint8)
solr.add([{
    "id": "doc1",
    "eq64": encode(emb, method="eq64"),
    "shq64": encode(emb, method="shq64", planes=64),
}])

# Exact term query — safe because position-safe encoding prevents false prefix matches
results = solr.search(f'eq64:"{encode(query_emb, method="eq64")}"')
```

---

## LangChain

Wrap any LangChain `Embeddings` implementation to automatically encode outputs.

```python
from langchain.embeddings import OpenAIEmbeddings
from uubed.integrations.langchain import UubedEmbeddings

base = OpenAIEmbeddings()
uubed_emb = UubedEmbeddings(base, method="shq64", planes=64, return_encoded=True)

docs_encoded = uubed_emb.embed_documents(["Hello, world!"])
# docs_encoded[0] is a uubed shq64 string ready for keyword storage
```

---

## Vector databases (Pinecone / Weaviate / Qdrant / ChromaDB)

Store the uubed string in the metadata payload alongside the raw vector, enabling hybrid filtering.

```python
from uubed.integrations.vectordb import get_connector
import numpy as np

connector = get_connector("pinecone", encoding_method="t8q64", k=8)
connector.connect(api_key="YOUR_KEY", environment="us-east1-gcp")
connector.create_collection("my-index", dimension=768)

vectors = [np.random.randint(0, 256, 768, dtype=np.uint8) for _ in range(100)]
# uubed strings injected automatically into metadata
connector.insert_vectors(vectors)
```
