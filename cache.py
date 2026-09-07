"""
Semantic cache.
================
Deliberately simple: no FAISS, no external vector DB. We embed every
incoming prompt with a small local sentence-transformer model, keep all
embeddings in a plain in-memory list, and compare a new prompt against every
cached entry with cosine similarity computed by hand in numpy. If anything is
similar enough, we return its stored response instantly instead of calling
an Ollama model at all.

This is O(n) per lookup (n = cache size), which is completely fine for an
MVP / demo-sized cache (dozens to low thousands of entries) and keeps the
whole thing inspectable in about 60 lines.
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine similarity cutoff above which a cached response is considered a
# match for a new prompt. Started at 0.92: with all-MiniLM-L6-v2, genuine
# paraphrases of the same question ("What is the capital of France?" vs.
# "What's France's capital city?") land around 0.90-0.98, while
# related-but-different prompts ("capital of France" vs "capital of
# Germany") land noticeably lower (~0.6-0.75). 0.92 sits just above that gap
# so near-duplicates hit while merely-related prompts still fall through to
# a real model call. Tune this constant directly if that trade-off needs
# to shift (lower = more hits but higher risk of returning a stale/wrong
# answer for a subtly different question; higher = safer but fewer hits).
SIMILARITY_THRESHOLD = 0.92


@dataclass
class CacheEntry:
    prompt: str
    response: str
    model_used: str
    similarity: float


class SemanticCache:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model = SentenceTransformer(model_name)
        self._embeddings: list[np.ndarray] = []
        self._prompts: list[str] = []
        self._responses: list[str] = []
        self._models_used: list[str] = []

    def embed(self, prompt: str) -> np.ndarray:
        # normalize_embeddings=True gives unit vectors, so cosine similarity
        # reduces to a plain dot product below.
        return self._model.encode(prompt, normalize_embeddings=True)

    def find(self, prompt: str) -> tuple[CacheEntry | None, np.ndarray]:
        """
        Embed `prompt` and look for a cached entry above SIMILARITY_THRESHOLD.
        Always returns the computed embedding too, so the caller can store it
        via `add()` without re-embedding on a cache miss.
        """
        query_embedding = self.embed(prompt)

        if not self._embeddings:
            return None, query_embedding

        # Stack cached embeddings into one matrix and take a single dot
        # product against the query - vectorized cosine similarity since
        # every embedding is already unit-normalized.
        matrix = np.vstack(self._embeddings)
        similarities = matrix @ query_embedding

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            entry = CacheEntry(
                prompt=self._prompts[best_idx],
                response=self._responses[best_idx],
                model_used=self._models_used[best_idx],
                similarity=best_score,
            )
            return entry, query_embedding

        return None, query_embedding

    def add(
        self, prompt: str, embedding: np.ndarray, response: str, model_used: str
    ) -> None:
        self._embeddings.append(embedding)
        self._prompts.append(prompt)
        self._responses.append(response)
        self._models_used.append(model_used)

    def __len__(self) -> int:
        return len(self._prompts)


if __name__ == "__main__":
    # Quick manual sanity check - run `python cache.py` to eyeball similarity
    # scores between paraphrases and unrelated prompts.
    cache = SemanticCache()
    cache.add(
        "What is the capital of France?",
        cache.embed("What is the capital of France?"),
        "The capital of France is Paris.",
        "qwen2.5:1.5b",
    )

    for test_prompt in [
        "What's France's capital city?",  # paraphrase -> should hit
        "Can you tell me the capital of France",  # paraphrase -> should hit
        "What is the capital of Germany?",  # related but different -> should miss
        "How do I bake sourdough bread?",  # unrelated -> should miss
    ]:
        entry, _ = cache.find(test_prompt)
        if entry:
            print(f"HIT  ({entry.similarity:.3f})  {test_prompt!r} -> {entry.prompt!r}")
        else:
            print(f"MISS           {test_prompt!r}")
