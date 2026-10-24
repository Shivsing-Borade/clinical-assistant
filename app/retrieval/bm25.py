import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter

class LocalBM25Retriever:
    """
    A lightweight, local BM25 implementation for keyword retrieval.
    Calculates scores based on term frequency, inverse document frequency,
    and document length normalization.
    """

    def __init__(self, corpus: List[Dict[str, Any]]):
        """
        Initialize BM25 retriever with a corpus of documents.
        corpus: List of dicts containing 'chunk_id', 'content', and 'metadata'.
        """
        self.corpus = corpus
        self.avg_dl = 0.0
        self.idf = {}
        self.doc_lengths = []
        self.tokenized_corpus = []

        # Parameters
        self.k1 = 1.5
        self.b = 0.75

        self._prepare_corpus()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase and alphanumeric split."""
        return re.findall(r'\w+', text.lower())

    def _prepare_corpus(self):
        """Builds the IDF and document length statistics."""
        total_docs = len(self.corpus)
        if total_docs == 0:
            return

        df = Counter()
        sum_dl = 0

        for doc in self.corpus:
            tokens = self._tokenize(doc['content'])
            self.tokenized_corpus.append(tokens)

            # Document length
            dl = len(tokens)
            self.doc_lengths.append(dl)
            sum_dl += dl

            # Document frequency
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        self.avg_dl = sum_dl / total_docs

        # Precompute IDF: log( (N - n + 0.5) / (n + 0.5) ) + 1
        # Added +1 to prevent negative IDF for very common terms
        for token, n in df.items():
            self.idf[token] = math.log((total_docs - n + 0.5) / (n + 0.5)) + 1.0

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Ranks documents based on BM25 score.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            dl = self.doc_lengths[i]

            # Term frequency in document
            tf_map = Counter(doc_tokens)

            for token in query_tokens:
                if token in self.idf:
                    tf = tf_map[token]
                    idf = self.idf[token]
                    # BM25 Formula: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_dl)))
                    denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
                    score += idf * (tf * (self.k1 + 1)) / denom

            scores.append((score, i))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        # Format top_k results
        results = []
        for score, idx in scores[:top_k]:
            if score <= 0:
                break

            doc = self.corpus[idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                "score": score,
                "metadata": doc["metadata"]
            })

        return results
