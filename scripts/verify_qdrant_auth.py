import logging
import sys
from app.retrieval.vector_store import QdrantVectorStore
from app.config import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_connection():
    try:
        logger.info("Initializing QdrantVectorStore...")
        store = QdrantVectorStore()
        collection_name = store.collection_name
        logger.info(f"Verifying collection: {collection_name}")

        collection_info = store.client.get_collection(collection_name=collection_name)

        # Log the actual structure of the config for diagnosis
        logger.info(f"Collection Config Params: {collection_info.config.params}")

        vectors_config = collection_info.config.params.vectors

        # Diagnostic check for the 'empty dict' issue
        if isinstance(vectors_config, dict):
            logger.info(f"Vectors config is a dict: {vectors_config}")
            # Some versions of the client might wrap the VectorParams in a way that doesn't
            # match standard dict access or has specific key names.
            if not vectors_config:
                # If it's an empty dict but the request was 200 OK, the client
                # might be failing to deserialize the response correctly into the Pydantic model.
                logger.error("Vector configuration dictionary is empty. This might be a deserialization issue.")
                return False

            first_key = list(vectors_config.keys())[0]
            config = vectors_config[first_key]
            dim = config.size
            dist = config.distance
        else:
            dim = vectors_config.size
            dist = vectors_config.distance

        logger.info(f"Successfully connected to Qdrant.")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Dimension: {dim} (Expected: {settings.EMBEDDING_OUTPUT_DIMENSION})")
        logger.info(f"Distance: {dist}")

        if dim != settings.EMBEDDING_OUTPUT_DIMENSION:
            logger.error(f"Dimension mismatch! Found {dim}, expected {settings.EMBEDDING_OUTPUT_DIMENSION}")
            return False

        return True

    except Exception as e:
        logger.error(f"Connection verification failed: {e}")
        return False

if __name__ == "__main__":
    if verify_connection():
        print("VERIFICATION_SUCCESS")
        sys.exit(0)
    else:
        print("VERIFICATION_FAILURE")
        sys.exit(1)
