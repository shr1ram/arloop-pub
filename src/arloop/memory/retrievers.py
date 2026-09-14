"""Retrievers: query(text, bank, k) -> [(Case, score)].

    bm25           Okapi BM25 (Lucene variant) — the lexical, exact contrast
    embedding      cosine over a local pinned embedding model
    hybrid         bm25 + embedding fused by Reciprocal Rank Fusion
    hybrid-rerank  hybrid recall pool -> cross-encoder ordering (the grid's arm)

The loop trace-logs every retrieval; that log is the retrieval audit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

from arloop.memory.bank import Bank
from arloop.memory.cases import Case

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RRF_K = 60

_WORD = re.compile(r"[a-z0-9_]{3,}")


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def indexed_text(case: Case) -> str:
    """What retrievers MATCH against: the written source-run context plus the
    content. Injection uses case.content only; the context is surface, not
    payload."""
    ctx = case.meta.get("retrieval_context", "")
    return f"{ctx}\n{case.content}" if ctx else case.content


class BM25Retriever:
    """Okapi BM25 (Lucene variant: idf strictly positive) over the index text;
    zero-overlap cases are never returned; ties break by case id."""
    name = "bm25"
    K1 = 1.2
    B = 0.75

    def query(self, text: str, bank: Bank, k: int) -> list[tuple[Case, float]]:
        """Rank the bank's cases by BM25 against the query text."""
        n = len(bank.cases)
        if n == 0:
            return []
        tfs = [Counter(_terms(indexed_text(c))) for c in bank.cases]
        lengths = [sum(tf.values()) for tf in tfs]
        avgdl = (sum(lengths) / n) or 1.0
        df: Counter[str] = Counter()
        for tf in tfs:
            df.update(tf.keys())

        query_terms = set(_terms(text))
        scored = []
        for case, tf, dl in zip(bank.cases, tfs, lengths):
            score = 0.0
            for t in query_terms:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                denom = tf[t] + self.K1 * (1 - self.B + self.B * dl / avgdl)
                score += idf * tf[t] * (self.K1 + 1) / denom
            if score > 0:
                scored.append((case, round(score, 6)))
        scored.sort(key=lambda cs: (-cs[1], cs[0].id))
        return scored[:k]


class EmbeddingRetriever:
    """Semantic nearest-k by cosine. The case-vector index is built once per
    (bank, model), cached in memory and on disk beside the bank."""
    name = "embedding"
    EMBED_CHUNK = 8

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, embed_fn=None,
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._embed_fn = embed_fn
        self._model = None
        self._cache: dict[str, tuple[list[str], list[list[float]]]] = {}

    def _embed(self, texts: list[str], is_query: bool) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts, is_query)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]

    def _index_path(self, bank: Bank) -> Path | None:
        if bank.dir is None:
            return None
        return bank.dir / f"embeddings-{self.model_name.replace('/', '--')}.json"

    @staticmethod
    def _ids(bank: Bank) -> list[str]:
        return [hashlib.sha256(indexed_text(c).encode()).hexdigest()[:12]
                for c in bank.cases]

    def _read_index(self, path: Path, ids: list[str]) -> list[list[float]] | None:
        """Vectors from a valid index file, else None (corrupt, partial, other
        model or other cases all mean 'not usable')."""
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        vectors = data.get("vectors") if isinstance(data, dict) else None
        if (isinstance(data, dict) and data.get("model") == self.model_name
                and data.get("ids") == ids and isinstance(vectors, list)):
            return vectors
        return None

    def _build_index(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.EMBED_CHUNK):
            vectors.extend(self._embed(texts[i:i + self.EMBED_CHUNK], is_query=False))
        return vectors

    def _write_index(self, path: Path, ids: list[str],
                     vectors: list[list[float]]) -> None:
        # unique tmp per writer + replace: readers see old-or-new, never partial
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"model": self.model_name, "ids": ids,
                                   "vectors": vectors}))
        os.replace(tmp, path)

    def _bank_vectors(self, bank: Bank) -> list[list[float]]:
        ids = self._ids(bank)
        cached = self._cache.get(bank.bank_id)
        if cached and cached[0] == ids:
            return cached[1]
        path = self._index_path(bank)
        texts = [indexed_text(c) for c in bank.cases]
        vectors = None
        if path is not None and path.exists():
            vectors = self._read_index(path, ids)
        if vectors is None:
            vectors = self._build_index(texts)
            if path is not None:
                self._write_index(path, ids, vectors)
        self._cache[bank.bank_id] = (ids, vectors)
        return vectors

    def embed_bank(self, bank: Bank) -> None:
        """Build and persist the on-disk index (build_corpus precomputes it so
        no cell pays for a whole-bank embed at query time)."""
        self._bank_vectors(bank)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return dot / norm if norm else 0.0

    def query(self, text: str, bank: Bank, k: int) -> list[tuple[Case, float]]:
        """Rank the bank's cases by cosine similarity to the query text."""
        if not bank.cases:
            return []
        vectors = self._bank_vectors(bank)
        q = self._embed([text], is_query=True)[0]
        scored = [(case, round(self._cosine(q, v), 6))
                  for case, v in zip(bank.cases, vectors)]
        scored.sort(key=lambda cs: (-cs[1], cs[0].id))
        return scored[:k]


class HybridRetriever:
    """Reciprocal Rank Fusion over the BM25 and embedding rankings:
    score(case) = sum_i 1/(rrf_k + rank_i), rank 1-based, only over lists where
    the case appears. Rank-based, so the two retrievers need no score
    calibration; ties break by case id."""
    name = "hybrid"

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, embed_fn=None,
                 rrf_k: int = RRF_K, device: str = "cpu"):
        self._bm25 = BM25Retriever()
        self._embedding = EmbeddingRetriever(model_name, embed_fn=embed_fn,
                                             device=device)
        self.rrf_k = rrf_k

    def query(self, text: str, bank: Bank, k: int) -> list[tuple[Case, float]]:
        """Fuse the two full rankings by RRF."""
        if not bank.cases:
            return []
        fused: dict[str, float] = {}
        by_id = bank.by_id
        for retriever in (self._bm25, self._embedding):
            for rank, (case, _) in enumerate(
                    retriever.query(text, bank, len(bank.cases)), start=1):
                fused[case.id] = fused.get(case.id, 0.0) + 1.0 / (self.rrf_k + rank)
        scored = [(by_id[cid], round(score, 6)) for cid, score in fused.items()]
        scored.sort(key=lambda cs: (-cs[1], cs[0].id))
        return scored[:k]


class CrossScorer:
    """Lazy local cross-encoder; tests inject ce_fn(query, texts) -> scores and
    never load the model."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, ce_fn=None,
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._ce_fn = ce_fn
        self._ce = None

    def scores(self, query: str, texts: list[str]) -> list[float]:
        """Cross-encoder relevance scores for (query, text) pairs."""
        if self._ce_fn is not None:
            return self._ce_fn(query, texts)
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            from arloop.memory.model_server import cross_encoder_kwargs
            self._ce = CrossEncoder(self.model_name, device=self.device,
                                    **cross_encoder_kwargs(self.device))
        return [float(x) for x in self._ce.predict([(query, t) for t in texts])]


class HybridRerankRetriever:
    """hybrid RRF recall pool -> cross-encoder ordering. The base contributes
    RECALL only (its top rerank_pool candidates); the cross-encoder owns the
    final order, and the pool bound also bounds cross-encoder cost per query."""
    name = "hybrid-rerank"

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL,
                 reranker_model: str = DEFAULT_RERANKER_MODEL, embed_fn=None,
                 ce_fn=None, rerank_pool: int = 64, rrf_k: int = RRF_K,
                 device: str = "cpu"):
        self._base = HybridRetriever(model_name, embed_fn=embed_fn, rrf_k=rrf_k,
                                     device=device)
        self.reranker_model = reranker_model
        self.rerank_pool = rerank_pool
        self._scorer = CrossScorer(reranker_model, ce_fn=ce_fn, device=device)

    def query(self, text: str, bank: Bank, k: int) -> list[tuple[Case, float]]:
        """Rerank the hybrid pool by cross-encoder score (sigmoid, desc)."""
        pool = self._base.query(text, bank, self.rerank_pool)
        if not pool:
            return []
        ces = self._scorer.scores(text, [indexed_text(c) for c, _ in pool])
        scored = [(c, round(1.0 / (1.0 + math.exp(-s)), 6))
                  for (c, _), s in zip(pool, ces)]
        scored.sort(key=lambda cs: (-cs[1], cs[0].id))
        return scored[:k]
