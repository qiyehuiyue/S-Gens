from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .gnn import SiameseGNNScorer, build_document_graph_view, heuristic_consistency
from .ollama import OllamaClient, OllamaConfig
from .retriever import build_retriever, tokenize


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str


@dataclass
class Document:
    id: str
    text: str


@dataclass
class AnchorInstance:
    id: str
    query: str
    positive_doc_id: str | None = None
    core_entities: list[str] = field(default_factory=list)


@dataclass
class ReasoningPath:
    triples: list[Triple]

    @property
    def entities(self) -> list[str]:
        if not self.triples:
            return []
        items = [self.triples[0].head]
        items.extend(triple.tail for triple in self.triples)
        return items

    @property
    def relations(self) -> list[str]:
        return [triple.relation for triple in self.triples]

    def to_dict(self) -> dict:
        return {"triples": [asdict(item) for item in self.triples]}


@dataclass
class SyntheticPositive:
    anchor_id: str
    synthetic_query: str
    positive_doc_id: str
    path: ReasoningPath
    path_coverage: float
    semantic_similarity: float
    consistency_score: float
    generation_backend: str = "template"
    path_length: int = 0

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "synthetic_query": self.synthetic_query,
            "positive_doc_id": self.positive_doc_id,
            "path": self.path.to_dict(),
            "path_coverage": self.path_coverage,
            "semantic_similarity": self.semantic_similarity,
            "consistency_score": self.consistency_score,
            "generation_backend": self.generation_backend,
            "path_length": self.path_length,
        }


@dataclass
class SyntheticTriplet:
    anchor_id: str
    query: str
    positive_doc_id: str
    negative_doc_id: str
    path: ReasoningPath
    negative_type: str
    positive_consistency: float
    negative_consistency: float
    weight: float
    consistency_backend: str = "heuristic"
    semantic_similarity: float = 0.0
    structural_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "query": self.query,
            "positive_doc_id": self.positive_doc_id,
            "negative_doc_id": self.negative_doc_id,
            "path": self.path.to_dict(),
            "negative_type": self.negative_type,
            "positive_consistency": self.positive_consistency,
            "negative_consistency": self.negative_consistency,
            "weight": self.weight,
            "consistency_backend": self.consistency_backend,
            "semantic_similarity": self.semantic_similarity,
            "structural_score": self.structural_score,
        }


@dataclass
class SgensConfig:
    min_path_length: int = 2
    max_path_length: int = 4
    positive_path_coverage_threshold: float = 0.75
    positive_similarity_threshold: float = 0.1
    structural_inconsistency_threshold: float = 0.5
    semantic_similarity_threshold: float = 0.1
    consistency_positive_threshold: float = 0.55
    consistency_negative_threshold: float = 0.45
    synthetic_ratio: float = 0.3
    candidate_pool_size: int = 12
    max_negatives_per_positive: int = 3
    temperature: float = 0.05
    structural_negative_penalty: float = 1.5
    retriever_backend: str = "lexical"
    retriever_model_name: str | None = None
    query_generator_backend: str = "template"
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://127.0.0.1:11434"
    consistency_backend: str = "heuristic"
    gnn_checkpoint_path: str | None = None
    semantic_pool_size: int = 8
    entity_pool_size: int = 8
    diversity_entity_substitution_quota: int = 1
    diversity_relation_break_quota: int = 1
    diversity_partial_path_quota: int = 1
    reliability_weight_floor: float = 0.25

    @property
    def tau_p(self) -> float:
        return self.positive_path_coverage_threshold

    @property
    def tau_n(self) -> float:
        return self.structural_inconsistency_threshold

    @property
    def tau_s(self) -> float:
        return self.semantic_similarity_threshold

    @property
    def tau_q_pos(self) -> float:
        return self.consistency_positive_threshold

    @property
    def tau_q_neg(self) -> float:
        return self.consistency_negative_threshold


class KnowledgeGraph:
    def __init__(self, triples: Iterable[Triple]) -> None:
        self.triples = list(triples)
        self.outgoing: dict[str, list[Triple]] = defaultdict(list)
        self.entities: set[str] = set()
        for triple in self.triples:
            self.outgoing[triple.head].append(triple)
            self.entities.add(triple.head)
            self.entities.add(triple.tail)

    def find_paths(self, source: str, target: str, min_hops: int, max_hops: int) -> list[ReasoningPath]:
        paths: list[ReasoningPath] = []

        def dfs(node: str, current: list[Triple], visited: set[str]) -> None:
            hops = len(current)
            if hops > max_hops:
                return
            if node == target and min_hops <= hops <= max_hops:
                paths.append(ReasoningPath(triples=list(current)))
            if hops == max_hops:
                return
            for triple in self.outgoing.get(node, []):
                if triple.tail in visited:
                    continue
                current.append(triple)
                visited.add(triple.tail)
                dfs(triple.tail, current, visited)
                visited.remove(triple.tail)
                current.pop()

        dfs(source, [], {source})
        return paths


class SgensPipeline:
    def __init__(self, graph: KnowledgeGraph, documents: list[Document], config: SgensConfig | None = None) -> None:
        self.graph = graph
        self.documents = documents
        self.config = config or SgensConfig()
        self.doc_by_id = {doc.id: doc for doc in documents}
        self.retriever = build_retriever(self.config.retriever_backend, self.config.retriever_model_name)
        self.ollama_client = None
        if self.config.query_generator_backend == "ollama":
            self.ollama_client = OllamaClient(OllamaConfig(model=self.config.ollama_model, base_url=self.config.ollama_url))
        self.gnn_scorer = None
        if self.config.consistency_backend == "gnn" and self.config.gnn_checkpoint_path:
            self.gnn_scorer = SiameseGNNScorer(self.config.gnn_checkpoint_path)

    @classmethod
    def from_json(cls, kg_path: str | Path, documents_path: str | Path, config: SgensConfig | None = None) -> "SgensPipeline":
        kg_data = json.loads(Path(kg_path).read_text(encoding="utf-8-sig"))
        doc_data = json.loads(Path(documents_path).read_text(encoding="utf-8-sig"))
        raw_triples = kg_data["triples"] if isinstance(kg_data, dict) and "triples" in kg_data else kg_data
        triples = [Triple(head=item[0], relation=item[1], tail=item[2]) if not isinstance(item, dict) else Triple(**item) for item in raw_triples]
        documents = [Document(**item) for item in doc_data]
        return cls(KnowledgeGraph(triples), documents, config=config)

    def load_original(self, original_path: str | Path) -> list[AnchorInstance]:
        data = json.loads(Path(original_path).read_text(encoding="utf-8-sig"))
        return [AnchorInstance(**item) for item in data]

    def synthesize_positive_pairs(self, anchors: list[AnchorInstance]) -> list[SyntheticPositive]:
        pairs: list[SyntheticPositive] = []
        for anchor in anchors:
            core_entities = anchor.core_entities or self._infer_core_entities(anchor.query)
            if len(core_entities) < 2:
                continue
            for source, target in self._entity_pairs(core_entities):
                paths = self.graph.find_paths(source, target, self.config.min_path_length, self.config.max_path_length)
                for path in paths:
                    synthetic_query = self._generate_query(anchor.query, path)
                    ranked_docs = self.retriever.rank(synthetic_query, self.documents, top_k=max(self.config.candidate_pool_size * 2, 10))
                    candidate_docs = [self.doc_by_id[hit.document_id] for hit in ranked_docs]
                    for doc in candidate_docs:
                        coverage = self._path_coverage(doc.text, path)
                        if coverage < self.config.tau_p:
                            continue
                        similarity = self._semantic_similarity(synthetic_query, doc.text)
                        if similarity < self.config.positive_similarity_threshold:
                            continue
                        consistency = self._consistency_score(path, doc.text)
                        if consistency < self.config.tau_q_pos:
                            continue
                        pairs.append(
                            SyntheticPositive(
                                anchor_id=anchor.id,
                                synthetic_query=synthetic_query,
                                positive_doc_id=doc.id,
                                path=path,
                                path_coverage=coverage,
                                semantic_similarity=similarity,
                                consistency_score=consistency,
                                generation_backend=self.config.query_generator_backend,
                                path_length=len(path.triples),
                            )
                        )
        return self._dedupe_positive_pairs(pairs)

    def construct_triplets(self, pairs: list[SyntheticPositive]) -> list[SyntheticTriplet]:
        triplets: list[SyntheticTriplet] = []
        for pair in pairs:
            positive_doc = self.doc_by_id[pair.positive_doc_id]
            negative_pool = self._candidate_negative_pool(pair.synthetic_query, pair.path, exclude_id=positive_doc.id)
            retained_by_type: dict[str, list[tuple[Document, float, float]]] = defaultdict(list)
            for doc in negative_pool:
                structural_score = self._structural_contribution_score(doc.text, pair.path)
                semantic_score = self._semantic_similarity(pair.synthetic_query, doc.text)
                if structural_score >= self.config.tau_n:
                    continue
                if semantic_score < self.config.tau_s:
                    continue
                consistency = self._consistency_score(pair.path, doc.text)
                if consistency > self.config.tau_q_neg:
                    continue
                negative_type = self._negative_type(doc.text, pair.path)
                retained_by_type[negative_type].append((doc, consistency, semantic_score))

            sampled = self._sample_diverse_negatives(retained_by_type)
            for negative_type, doc, consistency, semantic_score in sampled:
                structural_score = self._structural_contribution_score(doc.text, pair.path)
                weight = self._synthetic_weight(pair.consistency_score, consistency)
                triplets.append(
                    SyntheticTriplet(
                        anchor_id=pair.anchor_id,
                        query=pair.synthetic_query,
                        positive_doc_id=positive_doc.id,
                        negative_doc_id=doc.id,
                        path=pair.path,
                        negative_type=negative_type,
                        positive_consistency=pair.consistency_score,
                        negative_consistency=consistency,
                        weight=weight,
                        consistency_backend=self.config.consistency_backend,
                        semantic_similarity=semantic_score,
                        structural_score=structural_score,
                    )
                )
        return triplets

    def mix_training_data(self, anchors: list[AnchorInstance], triplets: list[SyntheticTriplet]) -> list[dict]:
        mixed: list[dict] = []
        for anchor in anchors:
            mixed.append({
                "source": "original",
                "anchor_id": anchor.id,
                "query": anchor.query,
                "positive_doc_id": anchor.positive_doc_id,
                "weight": 1.0,
            })
        synthetic_budget = max(1, math.ceil(len(anchors) * self.config.synthetic_ratio)) if triplets else 0
        for triplet in triplets[:synthetic_budget]:
            mixed.append({
                "source": "synthetic",
                "anchor_id": triplet.anchor_id,
                "query": triplet.query,
                "positive_doc_id": triplet.positive_doc_id,
                "negative_doc_id": triplet.negative_doc_id,
                "negative_type": triplet.negative_type,
                "weight": triplet.weight,
                "positive_consistency": triplet.positive_consistency,
                "negative_consistency": triplet.negative_consistency,
            })
        return mixed

    def run(self, anchors: list[AnchorInstance]) -> tuple[list[SyntheticPositive], list[SyntheticTriplet], list[dict]]:
        pairs = self.synthesize_positive_pairs(anchors)
        triplets = self.construct_triplets(pairs)
        mixed = self.mix_training_data(anchors, triplets)
        return pairs, triplets, mixed

    def _semantic_similarity(self, left: str, right: str) -> float:
        hits = self.retriever.rank(left, [Document(id="tmp", text=right)], top_k=1)
        return hits[0].score if hits else 0.0

    def _infer_core_entities(self, text: str) -> list[str]:
        lowered = text.lower()
        matches = [entity for entity in self.graph.entities if entity.lower() in lowered]
        return sorted(matches, key=len, reverse=True)

    def _entity_pairs(self, entities: list[str]) -> list[tuple[str, str]]:
        return [(left, right) for left in entities for right in entities if left != right]

    def _generate_query(self, anchor_query: str, path: ReasoningPath) -> str:
        if self.ollama_client is not None:
            try:
                prompt = self._build_query_generation_prompt(anchor_query, path)
                generated = self.ollama_client.generate(prompt, system=self._query_generation_system_prompt())
                if generated:
                    return generated
            except Exception:
                pass
        entities = path.entities
        relations = path.relations
        chain = ", then ".join(f"{relations[idx]} {entities[idx + 1]}" for idx in range(len(relations)))
        return f"{anchor_query} Follow the reasoning chain starting from {entities[0]} and {chain}."

    def _query_generation_system_prompt(self) -> str:
        return (
            "You generate synthetic retrieval queries for reasoning-intensive dense retrieval. "
            "Preserve the latent multi-hop path, keep the wording natural, and prefer implicit references, "
            "temporal constraints, compositional descriptions, or spatial reasoning when appropriate. "
            "Output exactly one query."
        )

    def _build_query_generation_prompt(self, anchor_query: str, path: ReasoningPath) -> str:
        triple_lines = [f"- {triple.head} --{triple.relation}--> {triple.tail}" for triple in path.triples]
        triples_text = "\n".join(triple_lines)
        return (
            f"Original query:\n{anchor_query}\n\n"
            f"Reasoning path:\n{triples_text}\n\n"
            "Rewrite the query into a single natural retrieval query that is logically grounded in the path. "
            "The query should not simply copy the path. It should remain answerable only if the reasoning chain is recoverable."
        )

    def _path_coverage(self, document: str, path: ReasoningPath) -> float:
        entities = path.entities
        hits = sum(1 for entity in entities if entity.lower() in document.lower())
        return hits / len(entities) if entities else 0.0

    def _candidate_negative_pool(self, query: str, path: ReasoningPath, exclude_id: str) -> list[Document]:
        semantic_hits = self.retriever.rank(query, self.documents, top_k=max(self.config.semantic_pool_size * 2, 10))
        semantic_pool = [self.doc_by_id[hit.document_id] for hit in semantic_hits if hit.document_id != exclude_id]
        entity_pool: list[Document] = []
        path_entities = {entity.lower() for entity in path.entities}
        for doc in self.documents:
            if doc.id == exclude_id:
                continue
            if any(entity in doc.text.lower() for entity in path_entities):
                entity_pool.append(doc)
        combined: dict[str, Document] = {}
        for doc in semantic_pool[: self.config.semantic_pool_size]:
            combined[doc.id] = doc
        for doc in entity_pool[: self.config.entity_pool_size]:
            combined.setdefault(doc.id, doc)
        return list(combined.values())[: self.config.candidate_pool_size]

    def _sample_diverse_negatives(self, retained_by_type: dict[str, list[tuple[Document, float, float]]]) -> list[tuple[str, Document, float, float]]:
        quotas = {
            "entity-substitution conflict": self.config.diversity_entity_substitution_quota,
            "relation-break conflict": self.config.diversity_relation_break_quota,
            "partial-path conflict": self.config.diversity_partial_path_quota,
        }
        sampled: list[tuple[str, Document, float, float]] = []
        for negative_type, quota in quotas.items():
            candidates = sorted(retained_by_type.get(negative_type, []), key=lambda item: (item[2], -item[1]), reverse=True)
            for doc, consistency, semantic_score in candidates[:quota]:
                sampled.append((negative_type, doc, consistency, semantic_score))
        if len(sampled) < self.config.max_negatives_per_positive:
            leftovers: list[tuple[str, Document, float, float]] = []
            for negative_type, candidates in retained_by_type.items():
                for doc, consistency, semantic_score in candidates:
                    item = (negative_type, doc, consistency, semantic_score)
                    if item not in sampled:
                        leftovers.append(item)
            leftovers.sort(key=lambda item: (item[3], -item[2]), reverse=True)
            for item in leftovers:
                if len(sampled) >= self.config.max_negatives_per_positive:
                    break
                sampled.append(item)
        return sampled[: self.config.max_negatives_per_positive]

    def _structural_contribution_score(self, document: str, path: ReasoningPath) -> float:
        if not path.triples:
            return 0.0
        supported = sum(1 for triple in path.triples if self._supports_triple(document, triple))
        return supported / len(path.triples)

    def _supports_triple(self, document: str, triple: Triple) -> bool:
        lowered = document.lower()
        relation_tokens = tokenize(triple.relation)
        return triple.head.lower() in lowered and triple.tail.lower() in lowered and all(token in lowered for token in relation_tokens)

    def _negative_type(self, document: str, path: ReasoningPath) -> str:
        lowered = document.lower()
        entities = path.entities
        relations = path.relations
        entity_hits = sum(1 for entity in entities if entity.lower() in lowered)
        relation_hits = sum(1 for relation in relations if all(token in lowered for token in tokenize(relation)))
        if entity_hits >= max(1, len(entities) - 1) and relation_hits == 0:
            return "relation-break conflict"
        if 0 < entity_hits < len(entities):
            return "partial-path conflict"
        return "entity-substitution conflict"

    def _consistency_score(self, path: ReasoningPath, document: str) -> float:
        if self.gnn_scorer is not None:
            document_triples = build_document_graph_view(path, document, self.graph.triples)
            return self.gnn_scorer.score(path, document_triples)
        return heuristic_consistency(path, document)

    def _synthetic_weight(self, positive_consistency: float, negative_consistency: float) -> float:
        margin = max(0.0, positive_consistency - negative_consistency)
        return max(self.config.reliability_weight_floor, margin) * self.config.structural_negative_penalty + 1.0

    def _dedupe_positive_pairs(self, pairs: list[SyntheticPositive]) -> list[SyntheticPositive]:
        best: dict[tuple[str, str, str], SyntheticPositive] = {}
        for pair in pairs:
            key = (pair.anchor_id, pair.synthetic_query, pair.positive_doc_id)
            current = best.get(key)
            if current is None or pair.consistency_score > current.consistency_score:
                best[key] = pair
        return list(best.values())


def write_json(path: str | Path, data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
