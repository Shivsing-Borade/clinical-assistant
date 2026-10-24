import logging
from app.config import settings
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_index():
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

        collection_name = settings.QDRANT_COLLECTION
        logger.info(f"Inspecting collection: {collection_name}")

        info = client.get_collection(collection_name=collection_name)

        # payload_index_config is typically in info.payload_schema or info.config.params.payload_index_config
        # In recent versions, it's often part of the config
        indexes = getattr(info.config.params, 'payload_index_config', {})

        if not indexes:
            logger.info("No payload indexes found in the collection.")
            return False

        content_index = indexes.get("content")
        if content_index:
            logger.info(f"Found index on 'content' field: {content_index}")
            # Check if it is a text index
            # The structure varies by version, but usually contains 'type' or is a TextIndexParams object
            if "text" in str(content_index).lower():
                logger.info("The 'content' field has a text index suitable for MatchText.")
                return True
            else:
                logger.info("The 'content' field has an index, but it may not be a full-text index.")
                return False
        else:
            logger.info("No index found on the 'content' field.")
            return False

    except Exception as e:
        logger.error(f"Error inspecting collection: {e}")
        return False

if __name__ == "__main__":
    check_index()
