from services.embedder import Embedder
from services.milvus_client import MilvusClient

MILVUS_HOST = "standalone"
MILVUS_PORT = 19530
COLLECTION_NAME = "documents"

milvus_client = MilvusClient(
    host=MILVUS_HOST,
    port=MILVUS_PORT,
)

embedder = Embedder(
    model_name="intfloat/multilingual-e5-base",
    batch_size=32,
)

def semantic_search(query: str, top_k: int = 5):
    query_embedding = embedder.encode_query(query)

    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        query_vectors=[query_embedding],
        top_k=top_k,
    )

    return results[0]
