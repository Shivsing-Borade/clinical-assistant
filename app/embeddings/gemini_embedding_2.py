import logging
from typing import List
from google import genai
from google.genai import types
from app.embeddings.base import BaseEmbeddingModel
from app.config import settings

logger = logging.getLogger(__name__)

class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Implementation of embeddings using Google Gemini Embedding 2."""

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in environment variables.")

        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model_id = settings.EMBEDDING_MODEL
        self._dimension = settings.EMBEDDING_OUTPUT_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single chunk of text."""
        if not text or not text.strip():
            # Handle empty text by returning a zero vector or raising error
            # In clinical context, we should probably log this and avoid indexing
            logger.warning("Attempted to embed empty text.")
            return [0.0] * self._dimension

        try:
            result = self.client.models.embed_content(
                model=self._model_id,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self._dimension)
            )

            vector = result.embeddings[0].values

            if len(vector) != self._dimension:
                raise ValueError(f"Expected vector dimension {self._dimension}, got {len(vector)}")

            return vector

        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of text chunks."""
        # Google GenAI SDK often supports batching.
        # If not available or limited, we implement with retries/chunking.
        # For now, we iterate to ensure strict dimension validation per chunk.
        return [self.embed_text(t) for t in texts]
