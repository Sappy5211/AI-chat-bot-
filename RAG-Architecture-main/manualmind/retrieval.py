from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import os
import re
from typing import List

from langchain.schema import Document

from manualmind.ingestion import VECTORSTORE_DIR


logger = logging.getLogger(__name__)

STORE_PATH = VECTORSTORE_DIR / "manualmind_store.json"

_embedding_model = None
_EMBEDDING_MODEL_NAME = os.getenv("MANUALMIND_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@dataclass
class RetrievalResult:
    documents: List[Document]
    scores: List[float]
    is_not_found: bool


DEFAULT_MIN_SIMILARITY_SCORE = float(os.getenv("MANUALMIND_MIN_SIMILARITY_SCORE", "0.35"))
KEYWORD_FALLBACK_MIN_SCORE = float(os.getenv("MANUALMIND_KEYWORD_MIN_SCORE", "0.15"))
SEMANTIC_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", _EMBEDDING_MODEL_NAME)
            _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        except ImportError:
            logger.warning("sentence-transformers is not installed. Falling back to keyword retrieval.")
            _embedding_model = None
        except Exception as exc:
            logger.warning("Embedding model load failed: %s", exc)
            _embedding_model = None
    return _embedding_model


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    model = _get_embedding_model()
    if model is None:
        return None
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [embedding.tolist() for embedding in embeddings]


def _embed_single(text: str) -> list[float] | None:
    embeddings = _embed_texts([text])
    if embeddings is None:
        return None
    return embeddings[0]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(0.0, min(1.0, dot))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_score(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    content_token_set = set(content_tokens)
    content_len = len(content_tokens)
    k1 = 1.2
    b = 0.75
    avg_dl = 200

    score = 0.0
    for query_token in query_tokens:
        if query_token in content_token_set:
            term_frequency = content_tokens.count(query_token)
            tf_norm = (term_frequency * (k1 + 1)) / (
                term_frequency + k1 * (1 - b + b * content_len / avg_dl)
            )
            score += tf_norm

    max_possible = len(query_tokens) * (k1 + 1)
    return min(1.0, score / max_possible) if max_possible > 0 else 0.0


def should_treat_as_not_found(
    documents: List[Document],
    scores: List[float],
    threshold: float = DEFAULT_MIN_SIMILARITY_SCORE,
) -> bool:
    has_content = any(doc.page_content.strip() for doc in documents)
    best_score = max(scores) if scores else None
    below_threshold = best_score is not None and best_score < threshold
    return not documents or not has_content or below_threshold


@lru_cache(maxsize=1)
def _read_store(store_path: str) -> List[dict]:
    path = STORE_PATH if store_path == str(STORE_PATH) else None
    if path is None or not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_store() -> List[dict]:
    if not STORE_PATH.exists():
        return []
    return _read_store(str(STORE_PATH))


def _write_store(rows: List[dict]) -> None:
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    STORE_PATH.write_text(json.dumps(rows), encoding="utf-8")
    _read_store.cache_clear()


def store_documents(documents: List[Document]) -> None:
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    existing_rows = _load_store()
    incoming_file_ids = {
        doc.metadata.get("file_id")
        for doc in documents
        if doc.metadata.get("file_id")
    }
    if incoming_file_ids:
        existing_rows = [
            row
            for row in existing_rows
            if row.get("metadata", {}).get("file_id") not in incoming_file_ids
        ]

    embeddings = _embed_texts([doc.page_content for doc in documents]) if documents else None
    for index, doc in enumerate(documents):
        row = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        if embeddings is not None:
            row["embedding"] = embeddings[index]
        existing_rows.append(row)

    _write_store(existing_rows)


def remove_documents_for_file(file_id: str) -> None:
    rows = _load_store()
    filtered_rows = [
        row for row in rows if row.get("metadata", {}).get("file_id") != file_id
    ]
    _write_store(filtered_rows)


def clear_documents() -> None:
    _write_store([])


def _document_fingerprint(row: dict) -> tuple:
    metadata = row.get("metadata", {})
    return (
        metadata.get("file_id"),
        metadata.get("file_name"),
        metadata.get("page_number"),
        metadata.get("content_type", "text"),
        row.get("page_content", "").strip()[:120],
    )


def retrieve_sources(query: str, top_k: int = 6) -> RetrievalResult:
    rows = _load_store()
    if not rows:
        return RetrievalResult(documents=[], scores=[], is_not_found=True)

    rows_have_embeddings = any("embedding" in row for row in rows)
    query_embedding = _embed_single(query) if rows_have_embeddings else None
    dense_rows_available = query_embedding is not None and rows_have_embeddings

    scored_rows = []
    for row in rows:
        content = row.get("page_content", "")
        if not content.strip():
            continue

        keyword_score = _keyword_score(query, content)
        semantic_score = 0.0
        if query_embedding is not None and "embedding" in row:
            semantic_score = _cosine_similarity(query_embedding, row["embedding"])

        if query_embedding is not None and "embedding" in row:
            score = SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_score
        else:
            score = keyword_score

        scored_rows.append((row, float(score)))

    scored_rows.sort(key=lambda item: item[1], reverse=True)
    top_rows = []
    seen = set()
    for row, score in scored_rows:
        fingerprint = _document_fingerprint(row)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        top_rows.append((row, score))
        if len(top_rows) >= top_k:
            break

    documents = [
        Document(page_content=row["page_content"], metadata=row["metadata"])
        for row, _score in top_rows
    ]
    scores = [score for _row, score in top_rows]

    threshold = DEFAULT_MIN_SIMILARITY_SCORE if dense_rows_available else KEYWORD_FALLBACK_MIN_SCORE
    is_not_found = should_treat_as_not_found(documents, scores, threshold=threshold)
    if is_not_found:
        logger.info(
            "Query '%s' treated as NOT_FOUND (best score: %.4f, threshold: %.4f, dense=%s).",
            query[:60],
            max(scores) if scores else 0.0,
            threshold,
            dense_rows_available,
        )

    return RetrievalResult(
        documents=documents,
        scores=scores,
        is_not_found=is_not_found,
    )
