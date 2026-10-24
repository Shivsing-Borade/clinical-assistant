import json
import logging
import argparse
from typing import List
from app.config import settings
from app.schemas.chunks import Chunk
from app.embeddings.gemini_embedding_2 import GeminiEmbeddingModel
from app.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_processed_chunks(file_path: str) -> List[Chunk]:
    logger.info(f"Loading chunks from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = [Chunk(**item) for item in data]
    logger.info(f"Loaded {len(chunks)} chunks.")
    return chunks

def run_indexing(dry_run: bool = True):
    """
    Indexes processed child chunks into Qdrant.
    """
    # 1. Load child chunks
    children_path = f"{settings.DATA_PROCESSED_DIR}/children.json"
    try:
        chunks = load_processed_chunks(children_path)
    except Exception as e:
        logger.error(f"Failed to load child chunks: {e}")
        return

    # 2. Validate metadata for all chunks
    missing_meta = []
    for i, chunk in enumerate(chunks):
        if not chunk.metadata.parent_id:
            missing_meta.append(f"Chunk {chunk.chunk_id} (index {i}) missing parent_id")

    if missing_meta:
        logger.error(f"Found {len(missing_meta)} chunks with missing metadata:")
        for msg in missing_meta[:10]:
            logger.error(msg)
        if len(missing_meta) > 10:
            logger.error(f"... and {len(missing_meta)-10} more.")
        return

    if dry_run:
        logger.info("--- DRY RUN MODE ---")
        logger.info(f"Validated {len(chunks)} child chunks for indexing.")
        logger.info(f"Target Embedding Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"Target Dimension: {settings.EMBEDDING_OUTPUT_DIMENSION}")
        logger.info(f"Target Qdrant Collection: {settings.QDRANT_COLLECTION}")
        logger.info("Dry run complete. No API calls or DB writes performed.")
        return

    # Real indexing flow
    try:
        # Initialize components
        embedder = GeminiEmbeddingModel()
        vector_store = QdrantVectorStore()

        # Ensure collection is ready
        vector_store.ensure_collection()

        # Process in batches to avoid API limits/memory issues
        batch_size = 50
        total_upserted = 0
        failures = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            try:
                logger.info(f"Embedding batch {i//batch_size + 1} ({i} to {min(i+batch_size, len(chunks))})...")
                vectors = embedder.embed_texts(texts)
                vector_store.upsert_chunks(batch, vectors)
                total_upserted += len(batch)
            except Exception as e:
                logger.error(f"Failed to process batch starting at {i}: {e}")
                failures += len(batch)

        logger.info(f"Indexing complete. Total upserted: {total_upserted}, Total failures: {failures}")

    except Exception as e:
        logger.error(f"Critical error during indexing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index processed chunks into Qdrant.")
    parser.add_argument("--execute", action="store_true", help="Actually run indexing. Defaults to dry-run.")
    args = parser.parse_args()

    run_indexing(dry_run=not args.execute)
