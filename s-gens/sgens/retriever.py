from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    AutoModel = None
    AutoTokenizer = None


TORCH_RETRIEVER_AVAILABLE = torch is not None and AutoModel is not None and AutoTokenizer is not None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class RetrievalHit:
    document_id: str
    text: str
    score: float


@dataclass
class SemanticScorer:
    """A lightweight lexical-semantic scorer used when no dense retriever is available."""

    def vectorize(self, text: str) -> Counter[str]:
        return Counter(tokenize(text))

    def similarity(self, left: str, right: str) -> float:
        lv = self.vectorize(left)
        rv = self.vectorize(right)
        if not lv or not rv:
            return 0.0
        common = set(lv) & set(rv)
        dot = sum(lv[token] * rv[token] for token in common)
        ln = math.sqrt(sum(value * value for value in lv.values()))
        rn = math.sqrt(sum(value * value for value in rv.values()))
        if ln == 0.0 or rn == 0.0:
            return 0.0
        return dot / (ln * rn)


class LexicalRetriever:
    def __init__(self) -> None:
        self.scorer = SemanticScorer()

    def score(self, query: str, text: str) -> float:
        return self.scorer.similarity(query, text)

    def rank(self, query: str, documents: Iterable[object], top_k: int = 5) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for document in documents:
            doc_id = getattr(document, "id")
            text = getattr(document, "text")
            hits.append(RetrievalHit(document_id=doc_id, text=text, score=self.score(query, text)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class TransformerBiEncoderRetriever:
    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        if not TORCH_RETRIEVER_AVAILABLE:
            raise RuntimeError(
                "Transformer retriever requires torch and transformers. Install them before using this backend."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def _mean_pool(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode_texts(self, texts: list[str], batch_size: int = 8):
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 0)

    def rank(self, query: str, documents: Iterable[object], top_k: int = 5) -> list[RetrievalHit]:
        docs = list(documents)
        if not docs:
            return []
        query_embedding = self.encode_texts([query])
        doc_embeddings = self.encode_texts([getattr(doc, "text") for doc in docs])
        scores = (query_embedding @ doc_embeddings.T).squeeze(0).tolist()
        hits = [
            RetrievalHit(document_id=getattr(doc, "id"), text=getattr(doc, "text"), score=float(score))
            for doc, score in zip(docs, scores, strict=True)
        ]
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


def build_retriever(backend: str, model_name_or_path: str | None = None):
    backend = backend.lower()
    if backend == "lexical":
        return LexicalRetriever()
    if backend in {"dense", "transformer", "bi-encoder"}:
        if not model_name_or_path:
            raise ValueError("A model name or path is required for the dense retriever backend.")
        return TransformerBiEncoderRetriever(model_name_or_path)
    raise ValueError(f"Unsupported retriever backend: {backend}")
