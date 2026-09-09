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

import asyncio
import json
import os
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Where the cache persists itself to disk. Two files, not one: embeddings go
# in a numpy .npz (they're the only part that benefits from numpy's binary
# format), and the parallel text fields go in a plain JSON sidecar. No
# pickle - a corrupt or tampered pickle file can execute arbitrary code on
# load; JSON + npz can't.
CACHE_STATE_PATH = "cache_state"

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

    async def save(self, path: str = CACHE_STATE_PATH) -> int:
        """
        Persist the cache to <path>.npz (embeddings, stacked into one
        matrix) and <path>.json (the parallel prompt/response/model_used
        lists). Called after every new entry (see main.py's chat()), not
        just on shutdown - see the module-level note on why - so this
        needs to not block the event loop on every single request.
        Returns the number of entries saved (0 and no files written if the
        cache is empty).

        np.savez()/json.dump() are both synchronous, blocking file I/O -
        run directly on the event loop (as this method used to, before it
        was async), every one of these calls stalls every other in-flight
        coroutine (including a different request's own rate-limit check
        or cache lookup) for however long the write takes. Fixed by
        snapshotting the data to save synchronously first (cheap - a numpy
        stack and three list copies, no I/O, and nothing here awaits
        anything, so no other coroutine can mutate self._embeddings/
        _prompts/etc. mid-snapshot), then handing the actual slow disk
        writes to a worker thread via asyncio.to_thread() so the event
        loop stays free while they run.

        Debouncing/batching saves instead (write every N entries, or on a
        timer) was the other option, but was rejected: it directly
        conflicts with why this method is called after every single entry
        in the first place (see cache_state persistence's own module note -
        a shutdown-only or infrequent save defeats surviving a crash,
        which is the case that matters most at this project's scale). A
        threaded write keeps that guarantee - every entry still gets
        persisted immediately - while just moving the blocking part off
        the loop.
        """
        if not self._embeddings:
            return 0
        matrix = np.vstack(self._embeddings)
        prompts = list(self._prompts)
        responses = list(self._responses)
        models_used = list(self._models_used)

        def _write() -> None:
            np.savez(f"{path}.npz", embeddings=matrix)
            with open(f"{path}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"prompts": prompts, "responses": responses, "models_used": models_used},
                    f,
                )

        await asyncio.to_thread(_write)
        return len(prompts)

    def load(self, path: str = CACHE_STATE_PATH) -> int:
        """
        Reload a previously saved cache from <path>.npz / <path>.json, if
        both exist. Called on startup. Returns the number of entries
        loaded (0 if no saved state was found, leaving the cache empty
        exactly as it was before this method existed).
        """
        npz_path, json_path = f"{path}.npz", f"{path}.json"
        if not (os.path.exists(npz_path) and os.path.exists(json_path)):
            return 0

        with np.load(npz_path) as data:
            matrix = data["embeddings"]
        with open(json_path, encoding="utf-8") as f:
            text_data = json.load(f)

        self._embeddings = list(matrix)
        self._prompts = text_data["prompts"]
        self._responses = text_data["responses"]
        self._models_used = text_data["models_used"]
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
