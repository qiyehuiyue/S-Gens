"""Reference implementation of the S-Gens pipeline."""

from .datasets import PreparedDataset, prepare_dataset, save_prepared_dataset
from .evaluation import evaluate_retriever
from .metrics import RankingMetrics
from .pipeline import SgensConfig, SgensPipeline
from .rag import RagResult, retrieve_and_answer

__all__ = [
    "PreparedDataset",
    "RankingMetrics",
    "RagResult",
    "SgensConfig",
    "SgensPipeline",
    "evaluate_retriever",
    "prepare_dataset",
    "retrieve_and_answer",
    "save_prepared_dataset",
]
