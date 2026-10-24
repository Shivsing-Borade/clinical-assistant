from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string of text into a vector."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings into a list of vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the output vectors."""
        pass
