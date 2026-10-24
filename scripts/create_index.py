# one time index creation
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings

client = QdrantClient(
      url=settings.QDRANT_URL,
      api_key=settings.QDRANT_API_KEY
  )

client.create_payload_index(
      collection_name=settings.QDRANT_COLLECTION,
      field_name="content",
      field_schema=models.TextIndexParams(
          type="text",
      ),
  )
print("Full-text index on 'content' created successfully.")