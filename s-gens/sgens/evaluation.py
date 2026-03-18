from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .metrics import aggregate_metrics
from .pipeline import AnchorInstance, Document
from .retriever import build_retriever


@dataclass
class EvaluationExample:
    query_id: str
    query: str
    relevant_doc_ids: set[str]


def load_documents(documents_path: str | Path) -> list[Document]:
    data = json.loads(Path(documents_path).read_text(encoding="utf-8-sig"))
    return [Document(**item) for item in data]


def load_examples(original_path: str | Path) -> list[EvaluationExample]:
    data = json.loads(Path(original_path).read_text(encoding="utf-8-sig"))
    examples: list[EvaluationExample] = []
    for item in data:
        anchor = AnchorInstance(**item)
        relevant = {anchor.positive_doc_id} if anchor.positive_doc_id else set()
        examples.append(EvaluationExample(query_id=anchor.id, query=anchor.query, relevant_doc_ids=relevant))
    return examples


def evaluate_retriever(
    backend: str,
    documents_path: str | Path,
    original_path: str | Path,
    model_name_or_path: str | None = None,
    top_k: int = 20,
) -> dict[str, float]:
    retriever = build_retriever(backend, model_name_or_path=model_name_or_path)
    documents = load_documents(documents_path)
    examples = load_examples(original_path)
    results: list[tuple[list[str], set[str]]] = []
    for example in examples:
        hits = retriever.rank(example.query, documents, top_k=top_k)
        ranked_ids = [hit.document_id for hit in hits]
        results.append((ranked_ids, example.relevant_doc_ids))
    return aggregate_metrics(results).to_dict()
