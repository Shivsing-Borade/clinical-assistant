import logging
from app.config import settings
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose():
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

        # 1. Server Version
        try:
            # Using .info() which is available in QdrantClient
            info_resp = client.info()
            logger.info(f"Qdrant Server Version: {info_resp.version}")
        except Exception as e:
            logger.info(f"Could not retrieve server info: {e}")

        # 2. Inspect Collection
        collection_name = settings.QDRANT_COLLECTION
        coll_info = client.get_collection(collection_name=collection_name)

        # Inspecting indexes specifically
        # The info object structure: coll_info.config.params.payload_index_config
        indexes = getattr(coll_info.config.params, 'payload_index_config', {})
        logger.info(f"Payload Index Config: {indexes}")

        content_index = indexes.get("content")
        if content_index:
            logger.info(f"Index on 'content' found: {content_index}")
        else:
            logger.info("No index found on 'content' field.")

    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")

if __name__ == "__main__":
    diagnose()
