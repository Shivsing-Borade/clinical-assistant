import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """Integration with Qdrant Vector Database for child chunk storage and retrieval."""

    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def ensure_collection(self):
        """Ensures the collection exists with correct dimensions and distance metric."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)

            # Check if dimensions match
            current_dim = collection_info.config.params.vectors.size
            if current_dim != settings.EMBEDDING_OUTPUT_DIMENSION:
                raise ValueError(
                    f"Collection '{self.collection_name}' exists but has dimension {current_dim}, "
                    f"which mismatches configured dimension {settings.EMBEDDING_OUTPUT_DIMENSION}."
                )
            logger.info(f"Collection '{self.collection_name}' already exists and is compatible.")

        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                logger.info(f"Creating collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=settings.EMBEDDING_OUTPUT_DIMENSION,
                        distance=models.Distance.COSINE
                    )
                )
            else:
                raise e

    def upsert_chunks(self, chunks: List[Any], vectors: List[List[float]]):
        """Upserts child chunks and their vectors into Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors.")

        points = []
        for chunk, vector in zip(chunks, vectors):
            import uuid
            # Create a deterministic UUID from the chunk_id string
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)

            # Payload includes all provenance metadata
            if hasattr(chunk.metadata, "model_dump"):
                metadata_dict = chunk.metadata.model_dump()
            else:
                metadata_dict = chunk.metadata

            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                **metadata_dict
            }

            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Successfully upserted {len(points)} points to Qdrant.")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a dense-vector similarity search using query_points.
        Returns a list of results containing content, score, and metadata.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        formatted_results = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.get("content", "")
            chunk_id = payload.get("chunk_id", "unknown")
            metadata = {k: v for k, v in payload.items() if k not in ["content", "chunk_id"]}

            formatted_results.append({
                "chunk_id": chunk_id,
                "content": content,
                "score": point.score,
                "metadata": metadata
            })

        return formatted_results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a full-text keyword search on the 'content' field.
        Returns a list of results containing content, score, and metadata.
        """
        # In qdrant-client 1.20.0, the recommended API for full-text search on a
        # payload index is using TextQuery with the 'using' parameter specifying the field.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=models.TextQuery(text=query),
            using="content",
            limit=top_k,
            with_payload=True
        )

        formatted_results = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.get("content", "")
            chunk_id = payload.get("chunk_id", "unknown")
            metadata = {k: v for k, v in payload.items() if k not in ["content", "chunk_id"]}

            formatted_results.append({
                "chunk_id": chunk_id,
                "content": content,
                "score": point.score,
                "metadata": metadata
            })

        return formatted_results
