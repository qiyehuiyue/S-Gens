from __future__ import annotations

from .pipeline import SgensConfig


def apply_paper_preset(config: SgensConfig, preset_name: str, dataset_name: str | None = None) -> SgensConfig:
    preset = preset_name.lower()
    dataset = (dataset_name or "").lower()

    if preset == "paper":
        config.min_path_length = 2
        config.max_path_length = 4
        config.synthetic_ratio = 0.3
        config.temperature = 0.05
        config.structural_negative_penalty = 1.5
        config.positive_path_coverage_threshold = 0.75
        config.structural_inconsistency_threshold = 0.5
        config.semantic_similarity_threshold = 0.1
        config.consistency_positive_threshold = 0.55
        config.consistency_negative_threshold = 0.45
        config.semantic_pool_size = 8
        config.entity_pool_size = 8
        config.candidate_pool_size = 12
        config.max_negatives_per_positive = 3
        config.diversity_entity_substitution_quota = 1
        config.diversity_relation_break_quota = 1
        config.diversity_partial_path_quota = 1

        if dataset == "webqsp":
            config.synthetic_ratio = 0.3
            config.semantic_similarity_threshold = 0.08
        elif dataset == "hotpotqa":
            config.synthetic_ratio = 0.3
            config.consistency_positive_threshold = 0.55
        elif dataset == "nq":
            config.synthetic_ratio = 0.3
            config.semantic_similarity_threshold = 0.12
        elif dataset == "triviaqa":
            config.synthetic_ratio = 0.3
            config.semantic_similarity_threshold = 0.12
        return config

    if preset == "faithful":
        config.query_generator_backend = "ollama" if config.query_generator_backend == "template" else config.query_generator_backend
        config.consistency_backend = "gnn" if config.consistency_backend == "heuristic" else config.consistency_backend
        config.semantic_pool_size = 10
        config.entity_pool_size = 10
        config.candidate_pool_size = 16
        config.max_negatives_per_positive = 3
        config.diversity_entity_substitution_quota = 1
        config.diversity_relation_break_quota = 1
        config.diversity_partial_path_quota = 1
        return apply_paper_preset(config, "paper", dataset_name=dataset_name)

    if preset == "lightweight":
        config.query_generator_backend = "template"
        config.consistency_backend = "heuristic"
        config.semantic_pool_size = 6
        config.entity_pool_size = 4
        config.candidate_pool_size = 8
        config.max_negatives_per_positive = 2
        return config

    raise ValueError(f"Unsupported preset: {preset_name}")
