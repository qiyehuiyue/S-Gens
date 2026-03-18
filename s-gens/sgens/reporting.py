from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    dataset: str
    preset: str
    num_pairs: int
    num_triplets: int
    metrics: dict[str, float]
    run_dir: str

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "preset": self.preset,
            "num_pairs": self.num_pairs,
            "num_triplets": self.num_triplets,
            "metrics": self.metrics,
            "run_dir": self.run_dir,
        }


@dataclass
class ResultsTable:
    rows: list[ExperimentResult]

    def to_markdown(self) -> str:
        lines = [
            "# Results Table",
            "",
            "| Dataset | Preset | Positives | Triplets | MRR@10 | Recall@20 | Hit@1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in self.rows:
            metrics = row.metrics or {}
            lines.append(
                "| {dataset} | {preset} | {pairs} | {triplets} | {mrr:.4f} | {recall:.4f} | {hit1:.4f} |".format(
                    dataset=row.dataset,
                    preset=row.preset,
                    pairs=row.num_pairs,
                    triplets=row.num_triplets,
                    mrr=float(metrics.get("mrr@10", 0.0)),
                    recall=float(metrics.get("recall@20", 0.0)),
                    hit1=float(metrics.get("hit@1", 0.0)),
                )
            )
        return "\n".join(lines) + "\n"


def collect_experiment_result(run_root: str | Path, metrics: dict[str, float] | None = None) -> ExperimentResult:
    run_root = Path(run_root)
    artifacts_dir, manifest_path = _resolve_run_structure(run_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {"metadata": {}}
    pairs = json.loads((artifacts_dir / "pairs.json").read_text(encoding="utf-8-sig"))
    triplets = json.loads((artifacts_dir / "triplets.json").read_text(encoding="utf-8-sig"))
    metadata = manifest.get("metadata", {})
    dataset = metadata.get("dataset") or run_root.name.replace("_paper", "").replace("_faithful", "")
    preset = metadata.get("preset") or _infer_preset_from_name(run_root.name)
    return ExperimentResult(
        dataset=dataset,
        preset=preset,
        num_pairs=len(pairs),
        num_triplets=len(triplets),
        metrics=metrics or _load_metrics_if_present(run_root),
        run_dir=str(run_root.resolve()),
    )


def collect_results_table(run_roots: list[str | Path]) -> ResultsTable:
    rows = [collect_experiment_result(run_root) for run_root in run_roots]
    return ResultsTable(rows=rows)


def write_results_json(path: str | Path, result: ExperimentResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def write_results_markdown(path: str | Path, result: ExperimentResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Results",
        "",
        f"- Dataset: `{result.dataset}`",
        f"- Preset: `{result.preset}`",
        f"- Synthetic positives: `{result.num_pairs}`",
        f"- Synthetic triplets: `{result.num_triplets}`",
        f"- Run directory: `{result.run_dir}`",
        "",
        "## Metrics",
        "",
    ]
    if result.metrics:
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key, value in result.metrics.items():
            lines.append(f"| {key} | {value:.4f} |")
    else:
        lines.append("No retriever metrics were provided.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_table_markdown(path: str | Path, table: ResultsTable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table.to_markdown(), encoding="utf-8")


def _load_metrics_if_present(run_root: Path) -> dict[str, float]:
    metrics_path = run_root / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    return {}


def _resolve_run_structure(run_root: Path) -> tuple[Path, Path]:
    new_artifacts = run_root / "sgens_run" / "artifacts"
    new_manifest = run_root / "sgens_run" / "run_manifest.json"
    if new_artifacts.exists():
        return new_artifacts, new_manifest
    legacy_artifacts = run_root / "sgens"
    legacy_manifest = run_root / "run_manifest.json"
    if legacy_artifacts.exists():
        return legacy_artifacts, legacy_manifest
    raise FileNotFoundError(f"Could not locate artifacts for run root: {run_root}")


def _infer_preset_from_name(name: str) -> str:
    if "faithful" in name:
        return "faithful"
    if "paper" in name:
        return "paper"
    if "lightweight" in name:
        return "lightweight"
    return "unknown"
