from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .pipeline import Document
from .retriever import TORCH_RETRIEVER_AVAILABLE, build_retriever


try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    AutoModel = None
    AutoTokenizer = None


@dataclass
class RetrieverTrainingConfig:
    model_name: str = "distilbert-base-uncased"
    learning_rate: float = 2e-5
    epochs: int = 1
    batch_size: int = 4
    max_length: int = 256
    margin: float = 0.2
    temperature: float = 0.05
    structural_negative_penalty: float = 1.5


class BiEncoderTrainer:
    def __init__(self, config: RetrieverTrainingConfig | None = None) -> None:
        if not TORCH_RETRIEVER_AVAILABLE:
            raise RuntimeError("Retriever training requires torch and transformers.")
        self.config = config or RetrieverTrainingConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name).to(self.device)

    def _mean_pool(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _encode(self, texts: list[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def _contrastive_loss(self, q_vec, p_vec, n_vec, weights):
        positive_scores = (q_vec * p_vec).sum(dim=1) / self.config.temperature
        negative_scores = (q_vec * n_vec).sum(dim=1) / self.config.temperature
        amplified_negative_scores = negative_scores * self.config.structural_negative_penalty
        stacked = torch.stack([positive_scores, amplified_negative_scores], dim=1)
        log_probs = F.log_softmax(stacked, dim=1)
        losses = -log_probs[:, 0] * weights
        return losses.mean()

    def train(self, triplets_path: str | Path, documents_path: str | Path, output_dir: str | Path) -> str:
        raw_triplets = json.loads(Path(triplets_path).read_text(encoding="utf-8-sig"))
        raw_documents = json.loads(Path(documents_path).read_text(encoding="utf-8-sig"))
        documents = {item["id"]: Document(**item) for item in raw_documents}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()
        for _ in range(self.config.epochs):
            for start in range(0, len(raw_triplets), self.config.batch_size):
                batch = raw_triplets[start : start + self.config.batch_size]
                queries = [item["query"] for item in batch]
                positives = [documents[item["positive_doc_id"]].text for item in batch]
                negatives = [documents[item["negative_doc_id"]].text for item in batch]
                weights = torch.tensor([float(item.get("weight", 1.0)) for item in batch], device=self.device)
                q_vec = self._encode(queries)
                p_vec = self._encode(positives)
                n_vec = self._encode(negatives)
                loss = self._contrastive_loss(q_vec, p_vec, n_vec, weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        (out_dir / "training_config.json").write_text(
            json.dumps(self.config.__dict__, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(out_dir)


def train_retriever(triplets_path: str | Path, documents_path: str | Path, output_dir: str | Path, model_name: str, epochs: int = 1) -> str:
    trainer = BiEncoderTrainer(RetrieverTrainingConfig(model_name=model_name, epochs=epochs))
    return trainer.train(triplets_path, documents_path, output_dir)


def load_retriever_for_inference(backend: str, model_name_or_path: str | None = None):
    return build_retriever(backend, model_name_or_path=model_name_or_path)
