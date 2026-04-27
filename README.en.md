# QAnchor

> A weak-supervision retrieval and reranking pipeline for financial-domain RAG

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Best MRR@10](https://img.shields.io/badge/MRR%4010-0.7758-brightgreen.svg)
![Status](https://img.shields.io/badge/status-Phase1%20complete-green.svg)

QAnchor is a retrieval and reranking optimization pipeline for Chinese financial annual-report question answering. It is not designed to train a QA system from scratch. Instead, it addresses a narrower and practical problem: **when chunk-level relevance labels are unavailable, existing question-answer pairs can still be used as weak supervision signals. QAnchor uses Reverse Mining to build training data automatically, fine-tunes a Cross-Encoder Reranker, and measures ranking improvements with Gold Eval.**

Pipeline overview: **annual-report PDF chunking -> multi-way candidate retrieval -> positive/negative mining from QA answers -> reranker fine-tuning -> ranking evaluation**.

---

## Key Results

> Fine-tuning Qwen3-Reranker-0.6B improves MRR@10 from **0.6115** to **0.7758** (+26.9%). Paired bootstrap and sign-flip tests for `Base -> Finetuned` show that the improvement is significantly positive.

| Model Configuration | Stage | max_length | MRR@10 | NDCG@10 | P@10 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Reranker-0.6B-seq-cls | Base | 768 | 0.6115 | 0.7572 | 0.192 |
| **Qwen3-Reranker-0.6B-seq-cls** | **Finetuned** | **768** | **0.7758 (+26.9%)** | **0.8761 (+15.7%)** | **0.228 (+18.8%)** |
| BGE-v2-m3 | Base | 768 | 0.6443 | 0.7986 | 0.190 |
| BGE-v2-m3 | Finetuned | 768 | 0.7096 (+10.1%) | 0.8522 (+6.7%) | 0.226 (+18.9%) |
| BGE-v2-m3 | Base | 512 | 0.6553 | 0.7840 | 0.182 |
| BGE-v2-m3 | Finetuned | 512 | 0.7103 (+8.4%) | 0.8282 (+5.6%) | 0.224 (+23.1%) |

The best configuration in this phase is **Qwen3-Reranker-0.6B-seq-cls Finetuned**. `max_length` is the maximum reranker input length after query and document text are combined; longer sequences are truncated.

Full report: `data/output/eval/stage1_reranker_comparison_report_20260118_v2.md`
Statistical significance report: `data/output/eval/qwen3-reranker-0.6b/significance_report_qwen3_template_20260117_base_vs_finetuned_20260423.md`

### Model Release

Published Qwen3 Reranker models:
- Merged: [`souflex56/qanchor-reranker-qwen3-0.6b-merged`](https://huggingface.co/souflex56/qanchor-reranker-qwen3-0.6b-merged)
- LoRA: [`souflex56/qanchor-reranker-qwen3-0.6b-lora`](https://huggingface.co/souflex56/qanchor-reranker-qwen3-0.6b-lora)

After installing dependencies with `pip install -r requirements.txt`, the reranker loading path in this repository can load both HuggingFace repo ids. The `merged` model is the recommended direct-use entry point. The `lora` repo is distributed as a LoRA/adapter version and should be loaded with PEFT.

**Qwen3 Reranker (recommended: merged version)**: Qwen3 needs a chat-template style pair format. For the underlying format, see the base model's ["Updated Sentence Transformers Usage"](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls).

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("souflex56/qanchor-reranker-qwen3-0.6b-merged")
queries = model.format_queries(["什么是营业收入？"])
docs = model.format_document(["公司2024年营业收入为..."])
scores = model.predict(list(zip(queries, docs)))
```

---

## Background and Data Source

### Where the Data Comes From

This project uses **FinGLM** annual-report QA data as the upstream QA signal. The data covers annual reports from Chinese A-share listed companies. Each sample contains a financial-domain question (`query`) and its corresponding **manually annotated standard answer** (`answer`).

The data cleaning, deduplication, QA type analysis, and company/year-level EDA are documented in [`FinGLM-data-eda`](https://github.com/souflex56/FinGLM-data-eda). QAnchor uses the standard answer mapping built from that data as the answer source for Reverse Mining.

The repository stores this signal through:
- `finglm_data_store/finglm_master.jsonl` - FinGLM manually annotated standard answer store, mapping queries to answers
- `finglm_data_store/` - data indexes and statistics

### Why This Is Still Weak Supervision

The missing supervision is **chunk-level relevance annotation**, not all QA signal. The available data is at the **question-answer level**: each record tells us what the answer text is, but it does **not** label which exact PDF chunk contains the evidence.

Reverse Mining maps this answer-level signal back to candidate chunks and automatically constructs the positive examples and hard negatives required for reranker training. This makes the project a weakly supervised ranking optimization pipeline, not a strongly supervised model trained on manually labeled chunk relevance.

---

## Method Overview

### Problem

Long financial documents, such as A-share annual reports, are hard to retrieve from reliably when no chunk-level labels are available.
- Annual-report PDFs contain many tables, so structured chunking quality strongly affects downstream retrieval.
- Without chunk-level relevance labels, a reranker cannot be fine-tuned directly.
- Cross-document retrieval introduces extra noise, so Phase1 restricts the retrieval scope.

This problem framing comes from the earlier FinGLM data exploration: the data provides standard answers and QA type/dimension statistics, but it does not provide evidence labels that point directly to PDF chunks.

### Solution

QAnchor combines Reverse Mining with listwise reranker fine-tuning:
- Mine training triplets automatically from existing QA answers.
- Fine-tune a Cross-Encoder Reranker with Listwise Softmax-CE Loss.
- Align training more directly with ranking metrics such as MRR and NDCG.

### Difference from General RAG Frameworks

| Dimension | General Frameworks (LangChain/LlamaIndex) | QAnchor |
|------|----------------------------------|---------|
| Chunking | General text splitters | PDF table-aware chunking with parent/child structure |
| Training data | Usually depends on manual labels | Automatically mined by Reverse Mining |
| Ranking optimization | Off-the-shelf or zero-shot | LoRA + listwise fine-tuning |
| Evaluation | No built-in project-specific protocol | Gold Eval + statistical significance checks |

**Best fit**: domain-document QA, financial-report retrieval, and RAG ranking optimization when labeled chunk relevance is unavailable.

---

## Pipeline

```text
QAnchor Pipeline

Step 1-2: PDF chunking + quality check
  ZenParse chunking -> parent/child hierarchy

Step 3-5: Three-way retrieval baseline
  Qwen3-Embedding-0.6B vector retrieval
  BM25 + jieba keyword retrieval
  RRF fusion (k=60) -> Top-50 candidates

Step 6: Reverse Mining
  Answer Matching -> automatic positive/negative labeling

Step 7-8: Gold Eval + Train/Dev split
  Multi-annotator adjudication (Gemini/Qwen/Codex)
  Blacklist isolation for the evaluation set

Step 9: Reranker LoRA training
  Qwen3-Reranker-0.6B-seq-cls / BGE-v2-m3
  Listwise Softmax-CE Loss
  PEFT + Accelerate + wandb

Step 10: Evaluation with 4 comparison groups
  MRR@10 / NDCG@10 / P@10
```

Notes:
- The "three-way retrieval baseline" in Step 3-5 is the candidate retrieval and initial ranking stage before reranker fine-tuning.
- Answer Matching in Step 6 uses the answer signal from `finglm_master.jsonl` to decide positives and hard negatives.
- Multi-annotator adjudication in Step 7 means several models judge Gold Eval candidate relevance separately, and their judgments are then aggregated.

| Module | Purpose | Key Code |
|------|------|----------|
| Data chunking | PDF -> parent/child chunks | `src/chunk_manager.py` |
| Multi-way retrieval | Embedding + BM25 + RRF fusion | `src/embedding_retriever.py` / `src/bm25_retriever.py` |
| Reverse Mining | QA standard answers -> training triplets | `scripts/06_reverse_mining.py` / `src/answer_matcher.py` |
| Reranker fine-tuning | LoRA + Listwise Loss | `scripts/09_train_reranker.py` |
| Gold Eval | Multi-model adjudication + evaluation | `scripts/10_evaluate.py` / `scripts/07b_adjudicate_gold_eval.py` |

For detailed module design, see [`docs/architecture.md`](docs/architecture.md).

---

## Project Scope and Data Scale

**Boundary conditions (Phase1)**
- Only Type1 questions are covered: single-company, single-year financial-report QA.
- The project only performs **intra-document evidence ranking**, not cross-PDF retrieval.
- Retrieval is limited to chunks from the query's corresponding `pdf_stem`.

**Data scale**
- Train/dev: 1,274 training triplets and 247 validation triplets
- Gold Eval: 50 gold evaluation queries
- Reverse Mining: 1,871 triplets and 3,069 hard negatives, averaging 12.58 hard negatives per valid query

---

## Quick Start

> If you only want to reproduce weak-supervision training and evaluation, this repository already includes the intermediate artifacts needed for retrieval, mining, training, and evaluation. After `git clone`, you can run Step A and later scripts directly. If you want to start from raw PDFs, you need to download the original PDFs and chunk outputs separately.

### Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Pipeline Overview

```text
PDF -> 1) Chunking -> 2) Retrieval -> 3) Reverse Mining -> 4) Fine-tuning -> 5) Evaluation
       needs files    needs files     included             included          included
```

- `included`: the repository already contains the input/output artifacts for this step, so it can be run after cloning.
- `needs files`: raw PDFs and chunk outputs are large and are not included in the repository. You can run the chunking scripts yourself, which is slower; see Step 3 in [`docs/reproduction.md`](docs/reproduction.md).

### Step A: Reverse Mining - Build Training Data from QA Answers

> **What this does**: reads the FinGLM standard answer store (`finglm_master.jsonl`) and uses key-value matching rules against the Top-50 results from three-way retrieval. Matched chunks become positives; high-ranked unmatched chunks become hard negatives. The output is a set of `(query, positive, negatives)` triplets.
>
> The `answer` field in `finglm_master.jsonl` is not a chunk label. It is the matching basis used by Reverse Mining.

```bash
python scripts/06_reverse_mining.py --stage stage1
```

- **Input**: `data/output/retrieval/hybrid_rrf_top50_stage1.jsonl` (included) + `finglm_data_store/finglm_master.jsonl` (included)
- **Output**: `data/output/mining/mined_triplets_stage1.jsonl`
- **Check**: `head -1 data/output/mining/mined_triplets_stage1.jsonl | python -m json.tool`

### Step B: Prepare Training Data - Filter Noise and Split Train/Dev

> **What this does**: filters the mined triplets again by removing queries in the evaluation blacklist, dropping noisy samples with `confidence < 0.7`, and splitting the remaining data into train/dev sets with a 9:1 ratio.

```bash
python scripts/08_prepare_train_data.py --stage stage1
```

- **Input**: `data/output/mining/mined_triplets_stage1.jsonl` + `config/eval_blacklist.json` (both included)
- **Output**: `data/output/train/train_triplets_stage1.jsonl` (~1,274 records), `dev_triplets_stage1.jsonl` (~247 records)
- **Check**: `wc -l data/output/train/*.jsonl`

### Later Steps

- **Fine-tuning (Step 4)**: train a Reranker LoRA adapter from the triplets generated in Step B. This requires a GPU. The repository already includes trained weights in `data/output/artifacts/reranker/`.
- **Evaluation (Step 5)**: compare Base vs Finetuned on Gold Eval and report MRR/NDCG/P metrics. Precomputed results are listed in the SSOT section of [`docs/architecture.md`](docs/architecture.md).
- For the full Step 0-10 reproduction flow, see [`docs/reproduction.md`](docs/reproduction.md).

---

## Project Structure

```text
QAnchor/
|-- scripts/                  # Full pipeline scripts (01-10)
|-- src/                      # Core modules (chunk_manager / retriever / answer_matcher / ...)
|-- config/
|   |-- weak_supervision_config.yaml   # Main configuration
|   |-- zenparse_config.yaml           # Chunking configuration
|   `-- eval_blacklist.json            # Evaluation isolation list
|-- data/
|   |-- input/                # Raw PDF data
|   `-- output/               # Outputs: chunks / retrieval / mining / train / eval / annotations
|-- docs/                     # Project documentation
`-- finglm_data_store/        # FinGLM QA mappings, standard answer indexes, and statistics
```

Key configuration file: `config/weak_supervision_config.yaml` manages chunking, retrieval, and fusion parameters.

---

## Documentation Index

| Document | Contents |
|------|------|
| [`docs/architecture.md`](docs/architecture.md) | Detailed module design, SSOT evidence paths, and statistical robustness |
| [`docs/reproduction.md`](docs/reproduction.md) | Full Step 0-10 reproduction guide with script commands and parameters |
| [`docs/faq.md`](docs/faq.md) | Technical FAQ for design decisions and core questions |
| [`docs/future_work.md`](docs/future_work.md) | Roadmap: data quality, negative mining, and evaluation robustness |
| [`docs/retrieval_config_contract.md`](docs/retrieval_config_contract.md) | Retrieval configuration contract and parameter behavior |

---

## Design Decisions

### Why Type1 / Intra-document Retrieval Only

Phase1 only covers Type1 questions: single-company, single-year financial-report QA. Retrieval is limited to the `pdf_stem` associated with each query. This reduces cross-document noise and keeps the ranking target focused: reorder chunks within one PDF, rather than retrieve across many PDFs.

### Why a 0.6B Small Model

LoRA fine-tuning was completed on a single RTX 4090 (24GB), and evaluation was run on Apple M2 Max (MPS). After weak-supervision fine-tuning, the 0.6B model improves MRR@10 from 0.6115 to 0.7758 (+26.9%), giving a practical balance between ranking quality and engineering cost.

### Why Listwise Instead of Pairwise

Listwise Softmax-CE Loss is directly aligned with ranking objectives such as MRR and NDCG. Compared with Pairwise Loss, it is better suited to Top-k ranking scenarios. For the rationale, see [`docs/faq.md`](docs/faq.md).

---

## Acknowledgements

This project relies on the following open-source projects for retrieval and evaluation:
- [ZenParse](https://github.com/AIDC-WX/ZenParse) - PDF structural parsing
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - base embedding model for vector retrieval
- [Qwen3-Reranker-0.6B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls) - base Cross-Encoder Reranker model used by the best-performing configuration
- [BGE-v2-m3](https://huggingface.co/BAAI/bge-m3) - comparison Cross-Encoder Reranker model

**Model replacement**: embedding and reranker model paths are configured through `config/weak_supervision_config.yaml`. When switching rerankers, pay attention to `--pair-format`: different models use different input formats, such as `qwen3_template` for Qwen3 and `hf_pair` for general pair models. The scripts support `auto` detection, and manual specification is also available. See Step 7 in [`docs/reproduction.md`](docs/reproduction.md).
