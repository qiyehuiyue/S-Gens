from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import SUPPORTED_DATASETS, prepare_dataset, save_prepared_dataset
from .evaluation import evaluate_retriever
from .gnn import GraphTrainingConfig, SiameseGNNTrainer, build_graph_training_examples_from_triplets
from .pipeline import SgensConfig, SgensPipeline, Triple, write_json
from .presets import apply_paper_preset
from .rag import retrieve_and_answer
from .reporting import (
    collect_experiment_result,
    collect_results_table,
    write_results_json,
    write_results_markdown,
    write_results_table_markdown,
)
from .training import train_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the S-Gens pipeline, training, and Ollama-backed RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the bundled demo dataset.")
    demo.add_argument("--output-dir", default="outputs", help="Directory for generated outputs.")

    run = subparsers.add_parser("run", help="Run on user-provided JSON inputs.")
    run.add_argument("--kg", required=True, help="Path to KG JSON.")
    run.add_argument("--documents", required=True, help="Path to document JSON.")
    run.add_argument("--original", required=True, help="Path to original anchor JSON.")
    run.add_argument("--output-dir", default="outputs", help="Directory for generated outputs.")
    _add_pipeline_args(run)

    prepare = subparsers.add_parser("prepare-dataset", help="Normalize raw benchmark data into S-Gens inputs.")
    prepare.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS), help="Dataset name.")
    prepare.add_argument("--raw", required=True, help="Path to raw dataset JSON or JSONL.")
    prepare.add_argument("--kg", help="Optional path to KG JSON.")
    prepare.add_argument("--output-dir", required=True, help="Directory for normalized outputs.")
    prepare.add_argument("--max-examples", type=int, help="Optional max number of raw examples to ingest.")

    run_dataset = subparsers.add_parser("run-dataset", help="Prepare a raw dataset, then run S-Gens on it.")
    run_dataset.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS), help="Dataset name.")
    run_dataset.add_argument("--raw", required=True, help="Path to raw dataset JSON or JSONL.")
    run_dataset.add_argument("--kg", required=True, help="Path to KG JSON.")
    run_dataset.add_argument("--output-dir", default="outputs", help="Directory for generated outputs.")
    run_dataset.add_argument("--max-examples", type=int, help="Optional max number of raw examples to ingest.")
    _add_pipeline_args(run_dataset)

    train_gnn = subparsers.add_parser("train-gnn", help="Train the Siamese GNN consistency filter from synthetic triplets.")
    train_gnn.add_argument("--triplets", required=True, help="Path to synthetic triplets JSON.")
    train_gnn.add_argument("--kg", required=True, help="Path to KG JSON.")
    train_gnn.add_argument("--output-dir", required=True, help="Directory for the trained checkpoint.")
    train_gnn.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")

    train_dense = subparsers.add_parser("train-retriever", help="Train a dense bi-encoder retriever from synthetic triplets.")
    train_dense.add_argument("--triplets", required=True, help="Path to synthetic triplets JSON.")
    train_dense.add_argument("--documents", required=True, help="Path to documents JSON.")
    train_dense.add_argument("--output-dir", required=True, help="Directory for the trained retriever.")
    train_dense.add_argument("--model-name", default="distilbert-base-uncased", help="Hugging Face model name or path.")
    train_dense.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")

    evaluate = subparsers.add_parser("evaluate-retriever", help="Evaluate a lexical or dense retriever.")
    evaluate.add_argument("--backend", choices=["lexical", "dense"], default="lexical", help="Retriever backend.")
    evaluate.add_argument("--documents", required=True, help="Path to documents JSON.")
    evaluate.add_argument("--original", required=True, help="Path to original anchor JSON.")
    evaluate.add_argument("--model", help="Dense retriever model path.")
    evaluate.add_argument("--top-k", type=int, default=20, help="Maximum ranking depth.")
    evaluate.add_argument("--save-to-run-root", help="Optional run root. If provided, metrics.json/results.* will be updated.")

    answer = subparsers.add_parser("answer", help="Retrieve documents and answer with Ollama.")
    answer.add_argument("--query", required=True, help="User query.")
    answer.add_argument("--documents", required=True, help="Path to documents JSON.")
    answer.add_argument("--retriever-backend", choices=["lexical", "dense"], default="lexical", help="Retriever backend.")
    answer.add_argument("--retriever-model", help="Dense retriever model name or path.")
    answer.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve.")
    answer.add_argument("--ollama-model", default="llama3.1:8b", help="Ollama model name.")
    answer.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    answer.add_argument("--output", help="Optional path to save the full RAG result JSON.")

    summarize = subparsers.add_parser("summarize-run", help="Write results.json and results.md for a completed run.")
    summarize.add_argument("--run-root", required=True, help="Experiment root directory produced by run-dataset.")
    summarize.add_argument("--metrics-file", help="Optional JSON file containing retriever metrics.")

    summarize_many = subparsers.add_parser("summarize-many", help="Write a markdown table across multiple experiment roots.")
    summarize_many.add_argument("--run-roots", nargs='+', required=True, help="Experiment roots to summarize.")
    summarize_many.add_argument("--output", required=True, help="Output markdown path.")
    return parser


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=["lightweight", "paper", "faithful"], default="paper", help="Pipeline preset.")
    parser.add_argument("--retriever-backend", choices=["lexical", "dense"], default="lexical", help="Retriever backend.")
    parser.add_argument("--retriever-model", help="Dense retriever model name or path.")
    parser.add_argument("--query-generator", choices=["template", "ollama"], default="template", help="Synthetic query generation backend.")
    parser.add_argument("--ollama-model", default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--consistency-backend", choices=["heuristic", "gnn"], default="heuristic", help="Consistency filtering backend.")
    parser.add_argument("--gnn-checkpoint", help="Path to a trained Siamese GNN checkpoint.")
    parser.add_argument("--experiment-name", help="Optional experiment directory name.")


def _build_config(args: argparse.Namespace, dataset_name: str | None = None) -> SgensConfig:
    config = SgensConfig(
        retriever_backend=args.retriever_backend,
        retriever_model_name=args.retriever_model,
        query_generator_backend=args.query_generator,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        consistency_backend=args.consistency_backend,
        gnn_checkpoint_path=args.gnn_checkpoint,
    )
    return apply_paper_preset(config, args.preset, dataset_name=dataset_name)


def _resolve_output_dir(output_dir: str, experiment_name: str | None) -> Path:
    base = Path(output_dir)
    return base / experiment_name if experiment_name else base


def _write_run_manifest(out_dir: Path, config: SgensConfig, metadata: dict) -> None:
    manifest = {"config": config.__dict__, "metadata": metadata}
    write_json(out_dir / "run_manifest.json", manifest)


def run_pipeline(kg: str, documents: str, original: str, output_dir: str, config: SgensConfig | None = None, metadata: dict | None = None) -> None:
    pipeline = SgensPipeline.from_json(kg, documents, config=config or SgensConfig())
    anchors = pipeline.load_original(original)
    pairs, triplets, mixed = pipeline.run(anchors)
    out_dir = Path(output_dir)
    artifacts_dir = out_dir / "artifacts"
    write_json(artifacts_dir / "pairs.json", [item.to_dict() for item in pairs])
    write_json(artifacts_dir / "triplets.json", [item.to_dict() for item in triplets])
    write_json(artifacts_dir / "mixed_training.json", mixed)
    _write_run_manifest(out_dir, pipeline.config, metadata or {})
    print(f"Generated {len(pairs)} synthetic positive pairs.")
    print(f"Generated {len(triplets)} filtered synthetic triplets.")
    print(f"Wrote outputs to {out_dir.resolve()}.")


def run_demo(output_dir: str) -> None:
    base = Path(__file__).resolve().parent.parent / "data"
    config = apply_paper_preset(SgensConfig(), "lightweight")
    pipeline = SgensPipeline.from_json(base / "demo_kg.json", base / "demo_documents.json", config=config)
    anchors = pipeline.load_original(base / "demo_original.json")
    pairs, triplets, mixed = pipeline.run(anchors)
    out_dir = Path(output_dir)
    write_json(out_dir / "demo_pairs.json", [item.to_dict() for item in pairs])
    write_json(out_dir / "demo_triplets.json", [item.to_dict() for item in triplets])
    write_json(out_dir / "demo_mixed_training.json", mixed)
    print(f"Demo positives: {len(pairs)}")
    print(f"Demo triplets: {len(triplets)}")
    print(f"Output directory: {out_dir.resolve()}")


def prepare_raw_dataset(dataset: str, raw: str, kg: str | None, output_dir: str, max_examples: int | None) -> None:
    prepared = prepare_dataset(dataset, raw, kg_path=kg, max_examples=max_examples)
    save_prepared_dataset(prepared, output_dir)
    print(f"Prepared dataset: {prepared.name}")
    print(f"Anchors: {len(prepared.anchors)}")
    print(f"Documents: {len(prepared.documents)}")
    print(f"Triples: {len(prepared.triples)}")
    print(f"Normalized data written to {Path(output_dir).resolve()}")


def run_raw_dataset(args: argparse.Namespace) -> None:
    experiment_dir = _resolve_output_dir(args.output_dir, args.experiment_name or f"{args.dataset}_{args.preset}")
    normalized_dir = experiment_dir / "normalized"
    prepared = prepare_dataset(args.dataset, args.raw, kg_path=args.kg, max_examples=args.max_examples)
    save_prepared_dataset(prepared, normalized_dir)
    run_pipeline(
        str(normalized_dir / "kg.json"),
        str(normalized_dir / "documents.json"),
        str(normalized_dir / "original.json"),
        str(experiment_dir / "sgens_run"),
        config=_build_config(args, dataset_name=args.dataset),
        metadata={"dataset": args.dataset, "raw": args.raw, "preset": args.preset},
    )


def train_gnn_filter(triplets_path: str, kg_path: str, output_dir: str, epochs: int) -> None:
    kg_data = json.loads(Path(kg_path).read_text(encoding="utf-8-sig"))
    raw_triples = kg_data["triples"] if isinstance(kg_data, dict) and "triples" in kg_data else kg_data
    triples = [Triple(head=item[0], relation=item[1], tail=item[2]) if not isinstance(item, dict) else Triple(**item) for item in raw_triples]
    examples = build_graph_training_examples_from_triplets(triplets_path, triples)
    trainer = SiameseGNNTrainer(GraphTrainingConfig(epochs=epochs))
    checkpoint = trainer.train(examples, output_dir)
    print(f"Saved Siamese GNN checkpoint to {checkpoint}")


def run_answer(args: argparse.Namespace) -> None:
    result = retrieve_and_answer(
        query=args.query,
        documents_path=args.documents,
        retriever_backend=args.retriever_backend,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )
    print(result.answer)
    if args.output:
        write_json(args.output, result.to_dict())
        print(f"Saved RAG result to {Path(args.output).resolve()}")


def summarize_run(run_root: str, metrics_file: str | None = None) -> None:
    metrics = None
    if metrics_file:
        metrics = json.loads(Path(metrics_file).read_text(encoding="utf-8-sig"))
    result = collect_experiment_result(run_root, metrics=metrics)
    run_root_path = Path(run_root)
    write_results_json(run_root_path / "results.json", result)
    write_results_markdown(run_root_path / "results.md", result)
    print(f"Wrote {run_root_path / 'results.json'}")
    print(f"Wrote {run_root_path / 'results.md'}")


def summarize_many(run_roots: list[str], output: str) -> None:
    table = collect_results_table(run_roots)
    write_results_table_markdown(output, table)
    print(f"Wrote {Path(output).resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.output_dir)
        return
    if args.command == "prepare-dataset":
        prepare_raw_dataset(args.dataset, args.raw, args.kg, args.output_dir, args.max_examples)
        return
    if args.command == "run-dataset":
        run_raw_dataset(args)
        return
    if args.command == "train-gnn":
        train_gnn_filter(args.triplets, args.kg, args.output_dir, args.epochs)
        return
    if args.command == "train-retriever":
        output_dir = train_retriever(args.triplets, args.documents, args.output_dir, args.model_name, args.epochs)
        print(f"Saved retriever to {output_dir}")
        return
    if args.command == "evaluate-retriever":
        metrics = evaluate_retriever(args.backend, args.documents, args.original, model_name_or_path=args.model, top_k=args.top_k)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        if args.save_to_run_root:
            run_root = Path(args.save_to_run_root)
            write_json(run_root / 'metrics.json', metrics)
            result = collect_experiment_result(run_root, metrics=metrics)
            write_results_json(run_root / 'results.json', result)
            write_results_markdown(run_root / 'results.md', result)
            print(f"Updated {run_root / 'metrics.json'}")
            print(f"Updated {run_root / 'results.json'}")
            print(f"Updated {run_root / 'results.md'}")
        return
    if args.command == "answer":
        run_answer(args)
        return
    if args.command == "summarize-run":
        summarize_run(args.run_root, args.metrics_file)
        return
    if args.command == "summarize-many":
        summarize_many(args.run_roots, args.output)
        return
    experiment_dir = _resolve_output_dir(args.output_dir, args.experiment_name)
    run_pipeline(args.kg, args.documents, args.original, str(experiment_dir), config=_build_config(args), metadata={"preset": args.preset})


if __name__ == "__main__":
    main()
