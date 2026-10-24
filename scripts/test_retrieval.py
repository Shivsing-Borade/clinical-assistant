import sys
import argparse
import logging
from app.config import settings
from app.embeddings.gemini_embedding_2 import GeminiEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test retrieval from clinical guidelines.")
    parser.add_argument("query", type=str, help="The clinical question to search for.")
    parser.add_argument("--top_k", type=int, default=settings.RETRIEVAL_TOP_K, help="Number of results to return.")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense", help="Retrieval mode (default: dense).")

    args = parser.parse_args()

    try:
        # Initialize components
        embedding_provider = GeminiEmbeddingProvider()
        vector_store = QdrantVectorStore()

        if args.mode == "dense":
            retriever = DenseRetriever(embedding_provider, vector_store)
        else:
            retriever = HybridRetriever(embedding_provider, vector_store)

        logger.info(f"Querying in {args.mode} mode: {args.query}")
        results = retriever.retrieve(args.query, top_k=args.top_k)

        if not results:
            print("\nNo results found.")
            return

        print(f"\nTop {len(results)} results ({args.mode} mode):\n" + "="*60)
        for i, res in enumerate(results, 1):
            score = res['score']
            content = res['content']
            meta = res['metadata']

            print(f"Result {i} | Score: {score:.4f}")
            print(f"Source: {meta.get('source_file', 'Unknown')} | Page: {meta.get('page_start', 'N/A')}")
            print(f"Section: {meta.get('section_path', 'N/A')}")
            print(f"Content: {content[:200]}...")
            print("-" * 60)

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
