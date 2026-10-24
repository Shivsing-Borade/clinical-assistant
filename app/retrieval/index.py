import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantIndex:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if not self.url or not self.api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY not provided")
            
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine"):
        """Creates a collection in Qdrant."""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.from_name(distance))
        )

    def upsert_chunks(self, collection_name: str, chunks: List[Dict[str, Any]]):
        """
        Upserts chunks with embeddings and metadata into Qdrant.
        Expects chunks to have 'chunk_id', 'content', 'embedding', and 'metadata'.
        """
        points = []
        for chunk in chunks:
            points.append(
                models.PointStruct(
                    id=chunk['chunk_id'], # Note: Qdrant expects UUID or Integer. We will handle this in dry-run.
                    vector=chunk['embedding'],
                    payload=chunk['metadata']
                )
            )
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def get_collection_info(self, collection_name: str):
        return self.client.get_collection(collection_name)
