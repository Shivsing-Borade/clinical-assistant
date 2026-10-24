import os
import logging
from typing import List
from .base import EmbeddingProvider
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)

class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.model = settings.EMBEDDING_MODEL # gemini-embedding-2
        self.dimension = settings.EMBEDDING_OUTPUT_DIMENSION # 768

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Empty or whitespace-only text cannot be embedded")

        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension)
            )
            vector = result.embeddings[0].values

            if len(vector) != self.dimension:
                raise ValueError(f"Expected vector dimension {self.dimension}, got {len(vector)}")

            return vector
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
