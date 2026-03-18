from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ollama import OllamaClient, OllamaConfig
from .pipeline import Document
from .retriever import RetrievalHit, build_retriever


@dataclass
class RagResult:
    query: str
    answer: str
    hits: list[RetrievalHit]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "hits": [
                {"document_id": hit.document_id, "score": hit.score, "text": hit.text}
                for hit in self.hits
            ],
        }


def load_documents(path: str | Path) -> list[Document]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [Document(**item) for item in data]


def build_rag_prompt(query: str, hits: list[RetrievalHit]) -> str:
    context_blocks = []
    for index, hit in enumerate(hits, start=1):
        context_blocks.append(f"[{index}] doc_id={hit.document_id}\n{hit.text}")
    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using only the retrieved evidence. "
        "If the evidence is insufficient, say so explicitly.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved Evidence:\n{context}\n\n"
        "Return a concise answer followed by a short evidence-based explanation."
    )


def retrieve_and_answer(
    query: str,
    documents_path: str | Path,
    retriever_backend: str = "lexical",
    retriever_model: str | None = None,
    top_k: int = 5,
    ollama_model: str = "llama3.1:8b",
    ollama_url: str = "http://127.0.0.1:11434",
) -> RagResult:
    documents = load_documents(documents_path)
    retriever = build_retriever(retriever_backend, model_name_or_path=retriever_model)
    hits = retriever.rank(query, documents, top_k=top_k)
    client = OllamaClient(OllamaConfig(model=ollama_model, base_url=ollama_url))
    prompt = build_rag_prompt(query, hits)
    answer = client.generate(prompt, system="You are a careful retrieval-augmented QA assistant.")
    return RagResult(query=query, answer=answer, hits=hits)
