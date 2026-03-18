# Data Guide

This document explains how to prepare data for the current S-Gens reproduction.

It is intentionally detailed. The goal is to make the `data/` folder self-contained, so you can understand:

1. where each benchmark dataset comes from;
2. what raw file format this project expects;
3. how the current code transforms raw benchmark data into S-Gens inputs;
4. what the external knowledge graph file must look like;
5. what limitations to expect for each dataset.

## Directory Layout

The current `data/` folder contains:

- `demo_original.json`: a tiny hand-written anchor set for the demo pipeline.
- `demo_documents.json`: a tiny hand-written passage collection for the demo pipeline.
- `demo_kg.json`: a tiny hand-written knowledge graph for the demo pipeline.
- `raw_samples/`: minimal raw-format examples for each supported benchmark.

Recommended structure for real experiments:

```text
data/
  README.md
  demo_original.json
  demo_documents.json
  demo_kg.json
  raw_samples/
    webqsp_sample.json
    hotpotqa_sample.json
    nq_sample.json
    triviaqa_sample.json
  raw/
    webqsp/
      train.json
      test.json
    hotpotqa/
      hotpot_train_v1.1.json
      hotpot_dev_distractor_v1.json
    nq/
      train.jsonl
      dev.jsonl
    triviaqa/
      wikipedia-train.json
      wikipedia-dev.json
  kg/
    freebase_subset.json
    wikidata_subset.json
  prepared/
    hotpotqa/
      original.json
      documents.json
      kg.json
      metadata.json
```

## The Three Standardized Files

All datasets are converted into the same three files before entering the S-Gens pipeline:

### `original.json`

This is the anchor training set. Each item contains:

- `id`: unique example id
- `query`: the user question or retrieval query
- `positive_doc_id`: the document id of a positive passage if one can be identified
- `core_entities`: a list of entities used to search KG reasoning paths

Example:

```json
[
  {
    "id": "ex1",
    "query": "Which actor starred in the movie directed by Christopher Nolan about dreams?",
    "positive_doc_id": "d1",
    "core_entities": ["Christopher Nolan", "Leonardo DiCaprio"]
  }
]
```

### `documents.json`

This is the passage collection used by the current S-Gens pipeline.

Each item contains:

- `id`: unique document id
- `text`: passage text

Example:

```json
[
  {
    "id": "d1",
    "text": "Christopher Nolan directed Inception. Leonardo DiCaprio starred in Inception."
  }
]
```

### `kg.json`

This is an external knowledge graph, not something automatically provided by the benchmark datasets.

The current implementation accepts either of these forms:

```json
{
  "triples": [
    ["Christopher Nolan", "directed", "Inception"],
    ["Inception", "stars", "Leonardo DiCaprio"]
  ]
}
```

or

```json
[
  {"head": "Christopher Nolan", "relation": "directed", "tail": "Inception"},
  {"head": "Inception", "relation": "stars", "tail": "Leonardo DiCaprio"}
]
```

## Important Constraint

These benchmark datasets are not knowledge graphs. They provide questions, answers, and sometimes contexts or retrieved pages.

The KG must be prepared separately.

In the current codebase:

- benchmark data becomes `original.json` and `documents.json`
- external KG becomes `kg.json`

If you skip the KG, the path-based part of S-Gens cannot function properly.

## How Data Preparation Works in Code

The normalization entrypoint is `prepare_dataset()` in:

- `sgens/datasets.py`

The saved output format is implemented in:

- `sgens/datasets.py`

The CLI commands are:

```bash
python -m sgens.cli prepare-dataset ...
python -m sgens.cli run-dataset ...
```

`prepare-dataset` only standardizes data.

`run-dataset` standardizes raw data first, then runs the S-Gens pipeline on the prepared files.

## Supported Datasets

The current project supports four dataset adapters:

- `webqsp`
- `hotpotqa`
- `nq`
- `triviaqa`

## 1. WebQSP

### What It Is

WebQSP is a knowledge-base question answering dataset derived from WebQuestions and aligned to Freebase-style semantic parses.

It is useful for reasoning-oriented retrieval because the question often implies a relation chain, even when that chain is not explicitly verbalized.

### Recommended Download Source

Use an official or widely used mirror of WebQSP in JSON format.

You should expect fields such as:

- `Question`
- `Parses`
- `TopicEntityName`
- `InferentialChain`
- `Answers`

This project does not fetch the dataset automatically. You download it manually and place it under something like:

```text
data/raw/webqsp/train.json
```

### Raw Fields Used by This Project

The adapter reads:

- `Question`, `question`, or `RawQuestion` as the query text
- `TopicEntityName` or `Parses[0].TopicEntityName` as the topic entity
- `Parses[].Answers[]` as answer entities
- `SupportingText` if available

### How It Is Converted

For each raw sample:

1. the question becomes `query`
2. a synthetic document id is created as `webqsp-doc-{idx}`
3. core entities are built from:
   - topic entity
   - the first one or two answer entities
4. the positive document text is chosen as:
   - `SupportingText` if your raw data includes it
   - otherwise a fallback text built from the question, topic entity, answer list, and inferential chain

### Why This Matters

WebQSP usually does not come with a real passage corpus in the same way HotpotQA does.

That means there are two possible workflows:

1. lightweight workflow:
   use the current fallback document construction to keep the pipeline runnable
2. proper retrieval workflow:
   attach an external text corpus aligned with the entities and answers

The second is much closer to the paper's intent.

### Limitation

If you only use raw WebQSP question files and do not provide supporting passages or an external passage corpus, the generated `documents.json` will be weak supervision rather than real retrieval evidence.

In that case:

- positive pairs may still be generated
- hard negatives may be sparse or poor quality

### Example Command

```bash
python -m sgens.cli prepare-dataset \
  --dataset webqsp \
  --raw data/raw/webqsp/train.json \
  --kg data/kg/freebase_subset.json \
  --output-dir data/prepared/webqsp
```

## 2. HotpotQA

### What It Is

HotpotQA is a multi-hop QA dataset with supporting facts and document contexts.

Among the currently supported datasets, this is the most natural fit for the current S-Gens implementation because it already provides multi-document evidence.

### Recommended Download Source

Use the official HotpotQA release, typically files such as:

- `hotpot_train_v1.1.json`
- `hotpot_dev_distractor_v1.json`

Place them under something like:

```text
data/raw/hotpotqa/hotpot_train_v1.1.json
```

### Raw Fields Used by This Project

The adapter reads:

- `question`
- `answer`
- `supporting_facts`
- `context`

### How It Is Converted

For each raw sample:

1. `question` becomes `query`
2. every item in `context` becomes a `Document`
   - each context item is expected to look like `[title, sentences]`
   - the document text becomes `"{title}. " + " ".join(sentences)`
3. the first context whose title appears in `supporting_facts` becomes `positive_doc_id`
4. `core_entities` are built from:
   - titles that appear in `supporting_facts`
   - the answer string, if present

### Why HotpotQA Works Well Here

HotpotQA already contains:

- explicit multi-hop supervision
- supporting titles
- context passages

So it usually gives the best early signal when testing whether the current S-Gens path extraction and hard negative logic are behaving reasonably.

### Limitation

The current implementation picks a single positive document id, even though HotpotQA is genuinely multi-document.

That means:

- it is suitable for getting the pipeline running
- it is not yet a full faithful multi-positive retrieval training setup

### Example Command

```bash
python -m sgens.cli prepare-dataset \
  --dataset hotpotqa \
  --raw data/raw/hotpotqa/hotpot_train_v1.1.json \
  --kg data/kg/wikidata_subset.json \
  --output-dir data/prepared/hotpotqa
```

## 3. Natural Questions (NQ)

### What It Is

Natural Questions is an open-domain QA dataset based on real user queries.

Many retrieval pipelines use a preprocessed open-domain version rather than the original raw Google annotation format.

### Recommended Download Source

Use an open-domain NQ release that already includes passages or contexts.

This project is designed for files where each example already includes fields like:

- `question` or `query`
- `contexts` or `passages`

Place them under something like:

```text
data/raw/nq/train.jsonl
```

### Raw Fields Used by This Project

The adapter reads:

- `question` or `query`
- `contexts` or `passages`

For each context or passage, it also looks for:

- `text`
- `passage_text`
- `context`
- `title`
- `is_positive`
- `has_answer`

### How It Is Converted

For each raw sample:

1. the question becomes `query`
2. every context becomes one document in `documents.json`
3. the first context marked with `is_positive` or `has_answer` becomes `positive_doc_id`
4. if no explicit positive marker exists, the first context is used as a fallback positive
5. `core_entities` are built with a simple heuristic:
   - named-looking tokens from the question
   - a few optional metadata fields if present

### Limitation

The current NQ adapter uses only heuristic entity extraction. It does not yet run entity linking.

So for strong KG-based path extraction, NQ will benefit from a better entity linker later.

### Example Command

```bash
python -m sgens.cli prepare-dataset \
  --dataset nq \
  --raw data/raw/nq/train.jsonl \
  --kg data/kg/wikidata_subset.json \
  --output-dir data/prepared/nq
```

## 4. TriviaQA

### What It Is

TriviaQA is a large-scale factoid QA dataset with answers and retrieved evidence pages.

It is more retrieval-friendly than WebQSP because it often comes with search result pages or entity pages.

### Recommended Download Source

Use the TriviaQA release that includes evidence pages or search results, for example Wikipedia-backed or web-backed versions.

The current adapter expects fields such as:

- `Question`
- `Answer`
- `SearchResults`

or alternatively:

- `question`
- `EntityPages`

Place them under something like:

```text
data/raw/triviaqa/wikipedia-train.json
```

### Raw Fields Used by This Project

The adapter reads:

- `Question` or `question`
- `Answer.Value`
- `Answer.Aliases`
- `SearchResults` or `EntityPages`

For each retrieved page it uses:

- `Title`
- `Snippet`
- `text`

### How It Is Converted

For each raw sample:

1. the question becomes `query`
2. every search result page becomes one document
3. the first page containing an answer alias becomes `positive_doc_id`
4. `core_entities` are built from:
   - named-looking tokens from the question
   - the first answer aliases

### Limitation

Answer-string matching is a simple heuristic. It is useful for bootstrapping but not as reliable as curated relevance labels.

### Example Command

```bash
python -m sgens.cli prepare-dataset \
  --dataset triviaqa \
  --raw data/raw/triviaqa/wikipedia-train.json \
  --kg data/kg/wikidata_subset.json \
  --output-dir data/prepared/triviaqa
```

## Knowledge Graph Preparation

### Why It Is Separate

The paper's method depends on an external KG for bounded path search.

None of the four benchmark adapters automatically create a proper KG from the benchmark itself.

You must supply one yourself.

### What the Current Code Expects

Accepted triple formats:

```json
{
  "triples": [
    ["head", "relation", "tail"]
  ]
}
```

or

```json
[
  {"head": "head", "relation": "relation", "tail": "tail"}
]
```

It also tolerates common alternative keys:

- `subject` or `s` for head
- `predicate` or `p` for relation
- `object` or `o` for tail

### Practical Advice

Recommended KG source by task type:

- `WebQSP`: Freebase-style subgraph is the most natural fit
- `HotpotQA`: Wikidata-style or Wikipedia-derived entity graph is often more practical
- `NQ`: Wikidata or a Wikipedia-derived entity relation graph
- `TriviaQA`: Wikidata or Wikipedia-derived graph

### Minimum Quality Requirement

Your KG should contain enough triples to connect the entities that appear in `core_entities`.

If your KG is too sparse:

- path extraction will fail
- no synthetic positives will be created
- no structural hard negatives will be created

## Minimal End-to-End Workflow

### Step 1: Put raw dataset files under `data/raw/`

Example:

```text
data/raw/hotpotqa/hotpot_train_v1.1.json
data/kg/wikidata_subset.json
```

### Step 2: Standardize the dataset

```bash
python -m sgens.cli prepare-dataset \
  --dataset hotpotqa \
  --raw data/raw/hotpotqa/hotpot_train_v1.1.json \
  --kg data/kg/wikidata_subset.json \
  --output-dir data/prepared/hotpotqa
```

### Step 3: Run the S-Gens pipeline

```bash
python -m sgens.cli run \
  --kg data/prepared/hotpotqa/kg.json \
  --documents data/prepared/hotpotqa/documents.json \
  --original data/prepared/hotpotqa/original.json \
  --output-dir outputs/hotpotqa
```

Or do both preparation and generation in one command:

```bash
python -m sgens.cli run-dataset \
  --dataset hotpotqa \
  --raw data/raw/hotpotqa/hotpot_train_v1.1.json \
  --kg data/kg/wikidata_subset.json \
  --output-dir outputs/hotpotqa
```

## Common Problems

### No synthetic positives are generated

Common causes:

- `core_entities` do not match your KG entity names
- the KG does not contain a path between source and target entities
- your document text does not mention enough path entities to pass path coverage

### WebQSP gives weak results

This is expected if you only use the raw QA file and do not attach a real text corpus.

### NQ or TriviaQA positives look noisy

This is also expected in the current implementation because positives are chosen with lightweight heuristics:

- `has_answer` or `is_positive` flags when available
- otherwise answer-string matching

### JSON files fail to load because of encoding

The current loader reads with `utf-8-sig`, so BOM-marked JSON files are supported.

## Files in This Folder

Use these files as references:

- `demo_original.json`
- `demo_documents.json`
- `demo_kg.json`
- `raw_samples/webqsp_sample.json`
- `raw_samples/hotpotqa_sample.json`
- `raw_samples/nq_sample.json`
- `raw_samples/triviaqa_sample.json`

## Recommended Starting Point

If you want to test the whole project with the least friction, start with HotpotQA.

Reason:

- it already contains contexts
- it already contains supporting facts
- it fits the current implementation better than WebQSP

If your goal is to be closer to the paper's knowledge-graph-heavy setting, then WebQSP plus a Freebase-style subgraph is the stronger choice, but it needs more data engineering.
