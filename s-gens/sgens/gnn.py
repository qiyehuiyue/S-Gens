from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


TORCH_GNN_AVAILABLE = torch is not None


@dataclass
class GraphPairExample:
    path_triples: list[Any]
    document_triples: list[Any]
    label: float

    def to_dict(self) -> dict:
        return {
            "path_triples": [_triple_to_dict(item) for item in self.path_triples],
            "document_triples": [_triple_to_dict(item) for item in self.document_triples],
            "label": self.label,
        }


@dataclass
class GraphTrainingConfig:
    embedding_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    epochs: int = 5
    learning_rate: float = 1e-3
    node_vocab_size: int = 8192
    relation_vocab_size: int = 2048


class HashingGraphIndexer:
    def __init__(self, node_vocab_size: int, relation_vocab_size: int) -> None:
        self.node_vocab_size = node_vocab_size
        self.relation_vocab_size = relation_vocab_size

    def node_id(self, text: str) -> int:
        return abs(hash(("node", text.lower()))) % self.node_vocab_size

    def relation_id(self, text: str) -> int:
        return abs(hash(("relation", text.lower()))) % self.relation_vocab_size


if TORCH_GNN_AVAILABLE:
    class SiameseGNN(nn.Module):
        def __init__(self, config: GraphTrainingConfig) -> None:
            super().__init__()
            self.config = config
            self.indexer = HashingGraphIndexer(config.node_vocab_size, config.relation_vocab_size)
            self.node_embeddings = nn.Embedding(config.node_vocab_size, config.embedding_dim)
            self.relation_embeddings = nn.Embedding(config.relation_vocab_size, config.embedding_dim)
            self.message_layers = nn.ModuleList(
                nn.Linear(config.embedding_dim * 2, config.hidden_dim) for _ in range(config.num_layers)
            )
            self.update_layers = nn.ModuleList(
                nn.Linear(config.embedding_dim + config.hidden_dim, config.embedding_dim) for _ in range(config.num_layers)
            )
            self.output = nn.Linear(config.embedding_dim, config.hidden_dim)

        def _build_graph(self, triples: list[Any]):
            nodes: dict[str, int] = {}
            edge_tuples: list[tuple[int, int, int]] = []
            for triple in triples:
                if triple.head not in nodes:
                    nodes[triple.head] = len(nodes)
                if triple.tail not in nodes:
                    nodes[triple.tail] = len(nodes)
                edge_tuples.append((nodes[triple.head], nodes[triple.tail], self.indexer.relation_id(triple.relation)))
            node_ids = [self.indexer.node_id(node) for node in nodes]
            return node_ids, edge_tuples

        def encode_graph(self, triples: list[Any]):
            node_ids, edges = self._build_graph(triples)
            if not node_ids:
                return torch.zeros(self.config.hidden_dim)
            device = self.node_embeddings.weight.device
            node_tensor = torch.tensor(node_ids, dtype=torch.long, device=device)
            node_repr = self.node_embeddings(node_tensor)
            for message_layer, update_layer in zip(self.message_layers, self.update_layers, strict=True):
                aggregated = torch.zeros(node_repr.size(0), self.config.hidden_dim, device=device)
                degree = torch.zeros(node_repr.size(0), 1, device=device)
                for src, dst, rel_id in edges:
                    relation_vec = self.relation_embeddings(torch.tensor(rel_id, dtype=torch.long, device=device))
                    src_message = message_layer(torch.cat([node_repr[src], relation_vec], dim=0))
                    dst_message = message_layer(torch.cat([node_repr[dst], relation_vec], dim=0))
                    aggregated[dst] += src_message
                    aggregated[src] += dst_message
                    degree[dst] += 1.0
                    degree[src] += 1.0
                aggregated = aggregated / torch.clamp(degree, min=1.0)
                updated = torch.cat([node_repr, aggregated], dim=1)
                node_repr = F.relu(update_layer(updated))
            graph_repr = node_repr.mean(dim=0)
            return F.normalize(self.output(graph_repr), dim=0)

        def score_pair(self, path_triples: list[Any], document_triples: list[Any]):
            path_repr = self.encode_graph(path_triples)
            document_repr = self.encode_graph(document_triples)
            return F.cosine_similarity(path_repr.unsqueeze(0), document_repr.unsqueeze(0)).squeeze(0)


class SiameseGNNTrainer:
    def __init__(self, config: GraphTrainingConfig | None = None) -> None:
        if not TORCH_GNN_AVAILABLE:
            raise RuntimeError("PyTorch is required to train the Siamese GNN filter.")
        self.config = config or GraphTrainingConfig()
        self.model = SiameseGNN(self.config)

    def train(self, examples: list[GraphPairExample], output_dir: str | Path) -> str:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        for _ in range(self.config.epochs):
            for example in examples:
                optimizer.zero_grad()
                score = self.model.score_pair(example.path_triples, example.document_triples)
                target = torch.tensor(example.label, dtype=torch.float32)
                loss = F.mse_loss(score, target)
                loss.backward()
                optimizer.step()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = out_dir / "siamese_gnn.pt"
        torch.save({"config": asdict(self.config), "state_dict": self.model.state_dict()}, checkpoint)
        return str(checkpoint)


class SiameseGNNScorer:
    def __init__(self, checkpoint_path: str | Path) -> None:
        if not TORCH_GNN_AVAILABLE:
            raise RuntimeError("PyTorch is required to load the Siamese GNN scorer.")
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.config = GraphTrainingConfig(**payload["config"])
        self.model = SiameseGNN(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def score(self, path: Any, document_triples: list[Any]) -> float:
        with torch.no_grad():
            return float(self.model.score_pair(path.triples, document_triples).item())


def build_graph_training_examples_from_triplets(triplets_path: str | Path, kg_triples: list[Any]) -> list[GraphPairExample]:
    raw = json.loads(Path(triplets_path).read_text(encoding="utf-8-sig"))
    examples: list[GraphPairExample] = []
    for item in raw:
        path_triples = [_triple_from_dict(triple) for triple in item["path"]["triples"]]
        document_triples = path_triples
        examples.append(GraphPairExample(path_triples=path_triples, document_triples=document_triples, label=1.0))
        negative_document_triples = path_triples[:-1] if len(path_triples) > 1 else []
        examples.append(GraphPairExample(path_triples=path_triples, document_triples=negative_document_triples, label=0.0))
    return examples


def build_document_graph_view(path: Any, document_text: str, kg_triples: list[Any], max_triples: int = 12) -> list[Any]:
    lowered = document_text.lower()
    path_entities = {entity.lower() for entity in path.entities}
    graph_view: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in kg_triples:
        key = (triple.head, triple.relation, triple.tail)
        if key in seen:
            continue
        head_in_doc = triple.head.lower() in lowered
        tail_in_doc = triple.tail.lower() in lowered
        relation_tokens = triple.relation.lower().split()
        relation_in_doc = all(token in lowered for token in relation_tokens)
        anchored_to_path = triple.head.lower() in path_entities or triple.tail.lower() in path_entities
        if anchored_to_path and ((head_in_doc and tail_in_doc) or (head_in_doc and relation_in_doc) or (tail_in_doc and relation_in_doc)):
            graph_view.append(triple)
            seen.add(key)
        if len(graph_view) >= max_triples:
            break
    return graph_view


def _triple_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "__dataclass_fields__"):
        return asdict(item)
    return {"head": item.head, "relation": item.relation, "tail": item.tail}


def _triple_from_dict(item: dict[str, Any]):
    return type("TripleLike", (), item)()


def heuristic_supported_triples(path: Any, document_text: str) -> list[Any]:
    lowered = document_text.lower()
    supported: list[Any] = []
    for triple in path.triples:
        relation_tokens = triple.relation.lower().split()
        if triple.head.lower() in lowered and triple.tail.lower() in lowered and all(token in lowered for token in relation_tokens):
            supported.append(triple)
    return supported


def heuristic_consistency(path: Any, document_text: str) -> float:
    supported = heuristic_supported_triples(path, document_text)
    if not path.triples:
        return 0.0
    return len(supported) / len(path.triples)
