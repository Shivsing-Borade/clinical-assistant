import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # General
    PROJECT_NAME: str = "Clinical Knowledge Assistant"

    # LLM / Embeddings
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "google")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    EMBEDDING_OUTPUT_DIMENSION: int = int(os.getenv("EMBEDDING_OUTPUT_DIMENSION", "768"))

    # Vector DB (Qdrant)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "clinical_guidelines")

    # Retrieval
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"

    # Data Paths
    DATA_RAW_DIR: str = "data/raw"
    DATA_INTERIM_DIR: str = "data/interim"
    DATA_PROCESSED_DIR: str = "data/processed"
    MANIFESTS_DIR: str = "manifests"

    class Config:
        env_file = ".env"

settings = Settings()
