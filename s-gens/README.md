# S-Gens

This project contains a reproduction-oriented implementation of `S-Gens` with paper-aligned presets, offline synthetic data generation, optional Ollama-backed query generation, Siamese GNN filtering, retriever training/evaluation, retrieval-augmented answering, and experiment reporting.

## Main Modules

- `sgens/pipeline.py`: offline S-Gens generation pipeline
- `sgens/presets.py`: paper-style configuration presets
- `sgens/datasets.py`: dataset normalization for WebQSP, HotpotQA, NQ, TriviaQA
- `sgens/training.py`: dense bi-encoder training entrypoint
- `sgens/evaluation.py`: lexical / dense retrieval evaluation
- `sgens/gnn.py`: Siamese GNN filter training and scoring
- `sgens/ollama.py`: Ollama client
- `sgens/rag.py`: retrieve-then-answer flow using Ollama
- `sgens/reporting.py`: run summaries and multi-run result tables

## Presets

The CLI exposes three presets:

- `lightweight`
- `paper`
- `faithful`

## CLI

```bash
python -m sgens.cli --help
```

Available commands:

- `demo`
- `run`
- `prepare-dataset`
- `run-dataset`
- `train-gnn`
- `train-retriever`
- `evaluate-retriever`
- `answer`
- `summarize-run`
- `summarize-many`

## Experiment Layout

Example:

```text
outputs/
  hotpotqa_paper/
    normalized/
      original.json
      documents.json
      kg.json
      metadata.json
    sgens_run/
      run_manifest.json
      artifacts/
        pairs.json
        triplets.json
        mixed_training.json
    checkpoints/
      gnn/
      retriever/
    metrics.json
    results.json
    results.md
```

## PowerShell Scripts

- `scripts/run_paper_pipeline.ps1`: run dataset preparation + S-Gens generation
- `scripts/run_full_pipeline.ps1`: run generation + optional GNN training + optional retriever training + evaluation + optional Ollama RAG + auto-write results
- `scripts/summarize_run.ps1`: write `results.json` and `results.md` for one run

## Typical Flows

Run paper-style generation only:

```powershell
.\scripts\run_paper_pipeline.ps1 `
  -Dataset hotpotqa `
  -Raw data\raw_samples\hotpotqa_sample.json `
  -Kg data\demo_kg.json `
  -OutputDir outputs
```

Run generation and evaluation in one shot:

```powershell
.\scripts\run_full_pipeline.ps1 `
  -Dataset hotpotqa `
  -Raw data\raw_samples\hotpotqa_sample.json `
  -Kg data\demo_kg.json `
  -OutputDir outputs `
  -Preset paper `
  -EvalBackend lexical
```

Run the fuller experiment chain with GNN and dense retriever training:

```powershell
.\scripts\run_full_pipeline.ps1 `
  -Dataset hotpotqa `
  -Raw data\raw_samples\hotpotqa_sample.json `
  -Kg data\demo_kg.json `
  -OutputDir outputs `
  -Preset paper `
  -TrainGnn `
  -TrainRetriever `
  -RetrieverModelName distilbert-base-uncased `
  -UseTrainedRetrieverForEval
```

Run the faithful chain with Ollama query generation and a final RAG answer:

```powershell
.\scripts\run_full_pipeline.ps1 `
  -Dataset hotpotqa `
  -Raw data\raw_samples\hotpotqa_sample.json `
  -Kg data\demo_kg.json `
  -OutputDir outputs `
  -Preset faithful `
  -UseOllamaQueryGeneration `
  -OllamaModel llama3.1:8b `
  -RunRagAnswer `
  -RagQuery "Which actor starred in the Christopher Nolan film about dreams?"
```

Summarize multiple runs into one markdown table:

```bash
python -m sgens.cli summarize-many \
  --run-roots outputs/hotpotqa_paper outputs/webqsp_run \
  --output outputs/results_table.md
```

## Current Environment Gap

The code supports PyTorch/Transformers training and Ollama integration, but this machine still does not have:

- `torch`
- `transformers`
- `ollama`

So the lexical pipeline, dataset prep, experiment layout, reporting, evaluation, and non-Ollama parts are verified locally.

The dense training, GNN training, and Ollama-backed generation paths are implemented but not executed in this environment yet.

## Data Notes

Detailed dataset instructions are in:

- `data/README.md`
- `data/README.zh-CN.md`
