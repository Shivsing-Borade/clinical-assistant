import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.schemas.chunks import Chunk

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """Integration with Qdrant Vector Database for child chunk storage."""

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
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

    def upsert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]):
        """Upserts child chunks and their vectors into Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors.")

        points = []
        for chunk, vector in zip(chunks, vectors):
            # Use stable point ID (chunk_id)
            # Qdrant points IDs can be UUID or integer. If chunk_id is string, we can use it as is if it's a valid UUID,
            # or we can hash it to a UUID.
            import uuid
            # Create a deterministic UUID from the chunk_id string
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)

            # Payload includes all provenance metadata
            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                **chunk.metadata.model_dump()
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
