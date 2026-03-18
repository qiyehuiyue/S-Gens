from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import AnchorInstance, Document, Triple, write_json


SUPPORTED_DATASETS = {"webqsp", "hotpotqa", "nq", "triviaqa"}
ENTITY_TOKEN_RE = re.compile(r"[A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)*")


@dataclass
class PreparedDataset:
    name: str
    anchors: list[AnchorInstance]
    documents: list[Document]
    triples: list[Triple]


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    data = _read_json(path)
    if isinstance(data, dict):
        for key in ("data", "examples", "items", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if "Questions" in data and isinstance(data["Questions"], list):
            return data["Questions"]
        return [data]
    return data


def prepare_dataset(
    dataset_name: str,
    raw_path: str | Path,
    kg_path: str | Path | None = None,
    max_examples: int | None = None,
) -> PreparedDataset:
    name = dataset_name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    records = _load_records(raw_path)
    if max_examples is not None:
        records = records[:max_examples]
    loader = {
        "webqsp": _prepare_webqsp,
        "hotpotqa": _prepare_hotpotqa,
        "nq": _prepare_nq,
        "triviaqa": _prepare_triviaqa,
    }[name]
    anchors, documents = loader(records)
    triples = _load_triples(kg_path) if kg_path else []
    return PreparedDataset(name=name, anchors=anchors, documents=documents, triples=triples)


def save_prepared_dataset(dataset: PreparedDataset, output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    write_json(out_dir / "original.json", [_anchor_to_dict(item) for item in dataset.anchors])
    write_json(out_dir / "documents.json", [_document_to_dict(item) for item in dataset.documents])
    write_json(out_dir / "kg.json", {"triples": [[t.head, t.relation, t.tail] for t in dataset.triples]})
    write_json(
        out_dir / "metadata.json",
        {
            "dataset": dataset.name,
            "num_anchors": len(dataset.anchors),
            "num_documents": len(dataset.documents),
            "num_triples": len(dataset.triples),
        },
    )


def _anchor_to_dict(anchor: AnchorInstance) -> dict[str, Any]:
    return {
        "id": anchor.id,
        "query": anchor.query,
        "positive_doc_id": anchor.positive_doc_id,
        "core_entities": anchor.core_entities,
    }


def _document_to_dict(document: Document) -> dict[str, Any]:
    return {"id": document.id, "text": document.text}


def _load_triples(path: str | Path) -> list[Triple]:
    data = _read_json(path)
    raw_triples = data["triples"] if isinstance(data, dict) and "triples" in data else data
    triples: list[Triple] = []
    for item in raw_triples:
        if isinstance(item, dict):
            head = item.get("head") or item.get("subject") or item.get("s")
            relation = item.get("relation") or item.get("predicate") or item.get("p")
            tail = item.get("tail") or item.get("object") or item.get("o")
        else:
            head, relation, tail = item
        if head and relation and tail:
            triples.append(Triple(head=head, relation=relation, tail=tail))
    return triples


def _prepare_webqsp(records: list[dict[str, Any]]) -> tuple[list[AnchorInstance], list[Document]]:
    anchors: list[AnchorInstance] = []
    documents: list[Document] = []
    seen_docs: set[str] = set()
    for idx, record in enumerate(records):
        question = record.get("Question") or record.get("question") or record.get("RawQuestion")
        if not question:
            continue
        topic_entity = (
            record.get("TopicEntityName")
            or _nested_get(record, "Parses", 0, "TopicEntityName")
            or _nested_get(record, "Parses", 0, "TopicEntityMid")
        )
        answers = _collect_answer_strings(record)
        positive_doc_id = f"webqsp-doc-{idx}"
        positive_text = record.get("SupportingText") or _build_webqsp_positive_text(question, topic_entity, answers, record)
        if positive_doc_id not in seen_docs:
            seen_docs.add(positive_doc_id)
            documents.append(Document(id=positive_doc_id, text=positive_text))
        core_entities = _unique([item for item in [topic_entity, *answers[:3], *_extract_named_candidates(record, question)] if item])
        anchors.append(
            AnchorInstance(
                id=str(record.get("QuestionId", f"webqsp-{idx}")),
                query=question,
                positive_doc_id=positive_doc_id,
                core_entities=core_entities,
            )
        )
    return anchors, documents


def _prepare_hotpotqa(records: list[dict[str, Any]]) -> tuple[list[AnchorInstance], list[Document]]:
    anchors: list[AnchorInstance] = []
    documents: list[Document] = []
    seen_docs: set[str] = set()
    for idx, record in enumerate(records):
        question = record.get("question")
        if not question:
            continue
        answer = record.get("answer")
        supporting_titles = {title for title, _ in record.get("supporting_facts", [])}
        positive_doc_ids: list[str] = []
        core_entities = [*supporting_titles, answer, *_extract_named_candidates(record, question)] if answer else [*supporting_titles, *_extract_named_candidates(record, question)]
        for ctx_idx, context in enumerate(record.get("context", [])):
            if not isinstance(context, list) or len(context) != 2:
                continue
            title, sentences = context
            text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            doc_id = f"hotpotqa-{idx}-{ctx_idx}"
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                documents.append(Document(id=doc_id, text=f"{title}. {text}".strip()))
            if title in supporting_titles and not positive_doc_ids:
                positive_doc_ids.append(doc_id)
        anchors.append(
            AnchorInstance(
                id=str(record.get("_id", f"hotpotqa-{idx}")),
                query=question,
                positive_doc_id=positive_doc_ids[0] if positive_doc_ids else None,
                core_entities=_unique([item for item in core_entities if item]),
            )
        )
    return anchors, documents


def _prepare_nq(records: list[dict[str, Any]]) -> tuple[list[AnchorInstance], list[Document]]:
    anchors: list[AnchorInstance] = []
    documents: list[Document] = []
    seen_docs: set[str] = set()
    for idx, record in enumerate(records):
        question = record.get("question") or record.get("query")
        if not question:
            continue
        contexts = record.get("contexts") or record.get("passages") or record.get("ctxs") or []
        positive_doc_id = None
        core_entities = _extract_named_candidates(record, question)
        answers = _collect_answer_aliases(record)
        core_entities.extend(answers[:2])
        for ctx_idx, ctx in enumerate(contexts):
            if isinstance(ctx, dict):
                text = ctx.get("text") or ctx.get("passage_text") or ctx.get("context") or ctx.get("contents") or ""
                title = ctx.get("title") or ctx.get("article_title") or ""
                is_positive = bool(ctx.get("is_positive") or ctx.get("has_answer") or ctx.get("positive"))
            else:
                text = str(ctx)
                title = ""
                is_positive = ctx_idx == 0
            doc_id = f"nq-{idx}-{ctx_idx}"
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                documents.append(Document(id=doc_id, text=f"{title}. {text}".strip(" .")))
            if is_positive and positive_doc_id is None:
                positive_doc_id = doc_id
        anchors.append(
            AnchorInstance(
                id=str(record.get("id", f"nq-{idx}")),
                query=question,
                positive_doc_id=positive_doc_id,
                core_entities=_unique([item for item in core_entities if item]),
            )
        )
    return anchors, documents


def _prepare_triviaqa(records: list[dict[str, Any]]) -> tuple[list[AnchorInstance], list[Document]]:
    anchors: list[AnchorInstance] = []
    documents: list[Document] = []
    seen_docs: set[str] = set()
    for idx, record in enumerate(records):
        question = record.get("Question") or record.get("question")
        if not question:
            continue
        answer_strings = _collect_answer_aliases(record)
        search_results = record.get("SearchResults") or record.get("EntityPages") or record.get("search_results") or []
        positive_doc_id = None
        core_entities = _extract_named_candidates(record, question) + answer_strings[:3]
        for doc_idx, page in enumerate(search_results):
            if isinstance(page, dict):
                title = page.get("Title") or page.get("title") or ""
                snippets = page.get("Snippet") or page.get("snippet") or page.get("text") or page.get("contents") or ""
                if isinstance(snippets, list):
                    snippets = " ".join(snippets)
                text = f"{title}. {snippets}".strip(" .")
                is_positive = any(alias.lower() in text.lower() for alias in answer_strings if alias)
            else:
                text = str(page)
                is_positive = any(alias.lower() in text.lower() for alias in answer_strings if alias)
            doc_id = f"triviaqa-{idx}-{doc_idx}"
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                documents.append(Document(id=doc_id, text=text))
            if is_positive and positive_doc_id is None:
                positive_doc_id = doc_id
        anchors.append(
            AnchorInstance(
                id=str(record.get("QuestionId", f"triviaqa-{idx}")),
                query=question,
                positive_doc_id=positive_doc_id,
                core_entities=_unique([item for item in core_entities if item]),
            )
        )
    return anchors, documents


def _nested_get(container: dict[str, Any], *path: Any) -> Any:
    current: Any = container
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return None
            current = current[part]
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _collect_answer_strings(record: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    parses = record.get("Parses") or []
    for parse in parses:
        for answer in parse.get("Answers", []):
            if isinstance(answer, dict):
                value = answer.get("EntityName") or answer.get("AnswerArgument") or answer.get("EntityMid")
                if value:
                    answers.append(value)
            elif answer:
                answers.append(str(answer))
    return _unique(answers)


def _collect_answer_aliases(record: dict[str, Any]) -> list[str]:
    answer = record.get("Answer") or record.get("answer") or {}
    aliases: list[str] = []
    if isinstance(answer, dict):
        aliases.extend(answer.get("Aliases", []))
        if answer.get("Value"):
            aliases.append(answer["Value"])
        if answer.get("NormalizedValue"):
            aliases.append(answer["NormalizedValue"])
    elif isinstance(answer, str):
        aliases.append(answer)
    return _unique([alias for alias in aliases if alias])


def _build_webqsp_positive_text(question: str, topic_entity: str | None, answers: list[str], record: dict[str, Any]) -> str:
    inferential_chain = _flatten_inferential_chain(record)
    parts = [question]
    if topic_entity:
        parts.append(f"Topic entity: {topic_entity}.")
    if answers:
        parts.append(f"Answer candidates: {', '.join(answers[:4])}.")
    if inferential_chain:
        parts.append(f"Inferential chain: {' ; '.join(inferential_chain[:4])}.")
    return " ".join(parts)


def _flatten_inferential_chain(record: dict[str, Any]) -> list[str]:
    chain: list[str] = []
    for parse in record.get("Parses", []):
        for relation in parse.get("InferentialChain", []):
            if relation:
                chain.append(str(relation))
    return chain


def _extract_named_candidates(record: dict[str, Any], question: str) -> list[str]:
    values: list[str] = []
    for key in (
        "title",
        "entity",
        "topic_entity",
        "subject",
        "TopicEntityName",
        "AnswerArgument",
    ):
        value = record.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    values.extend(match.group(0).strip() for match in ENTITY_TOKEN_RE.finditer(question))
    return _unique(values)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
