#!/usr/bin/env python3
"""Step 10 evaluation: compute metrics for four comparison groups."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunk_manager import ChunkIndex, load_chunks  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{_now_stamp()}] {msg}", flush=True)


def _now_iso() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).isoformat()


def _now_stamp() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y%m%d-%H%M%S")


def _resolve_output_prefix(
    stage: str, requested: Optional[str], run_stamp: str
) -> str:
    if not requested or requested == "auto":
        return f"{stage}_{run_stamp}"
    if requested == "final":
        return stage
    return requested


def _resolve_checkpoint_suffix(
    requested: Optional[str], run_stamp: str
) -> Optional[str]:
    if not requested or requested == "auto":
        return run_stamp
    if requested in ("final", "none"):
        return None
    return requested


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_candidates(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    if rec.get("hits") is not None:
        return rec["hits"]
    if rec.get("candidates") is not None:
        return rec["candidates"]
    if rec.get("chunks") is not None:
        return rec["chunks"]
    return []


def _sorted_candidates(cands: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(hit: Dict[str, Any]) -> Tuple[int, float]:
        rank = hit.get("rerank_rank") or hit.get("rank")
        if rank is None:
            rank = 10**9
        score = hit.get("rerank_score")
        if score is None:
            score = hit.get("score", 0.0)
        return (int(rank), -float(score))

    return sorted(list(cands), key=_key)


def _label_gain(label: str) -> int:
    if label == "evidence":
        return 2
    if label == "related":
        return 1
    return 0


def _is_positive(label: str) -> bool:
    return label == "evidence"


def _dcg(gains: Sequence[int], k: int) -> float:
    total = 0.0
    for idx, gain in enumerate(gains[:k], start=1):
        total += gain / math.log2(idx + 1)
    return total


def _ndcg(gains: Sequence[int], k: int) -> float:
    dcg = _dcg(gains, k)
    if dcg == 0.0:
        return 0.0
    ideal = sorted(gains, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _mrr_at_k(labels: Sequence[str], k: int) -> float:
    for idx, label in enumerate(labels[:k], start=1):
        if _is_positive(label):
            return 1.0 / idx
    return 0.0


def _precision_at_k(labels: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    positives = sum(1 for label in labels[:k] if _is_positive(label))
    return positives / k


def _metrics_from_labels(labels: Sequence[str], k: int) -> Dict[str, float]:
    gains = [_label_gain(label) for label in labels]
    return {
        "mrr@10": _mrr_at_k(labels, k),
        "ndcg@10": _ndcg(gains, k),
        "p@10": _precision_at_k(labels, k),
    }


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(fmean(values))


def _scores_from_logits(logits: torch.Tensor, config: Any) -> torch.Tensor:
    if logits.dim() == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits.view(-1)
    if logits.shape[-1] == 2:
        id2label = getattr(config, "id2label", {}) or {}
        label_map = {str(v).lower(): int(k) for k, v in id2label.items()}
        if "yes" in label_map and "no" in label_map:
            return logits[:, label_map["yes"]] - logits[:, label_map["no"]]
        if "pos" in label_map and "neg" in label_map:
            return logits[:, label_map["pos"]] - logits[:, label_map["neg"]]
        return logits[:, 1]
    return logits[:, 0]


_PAIR_FORMAT_CHOICES = (
    "hf_pair",
    "qwen3_template",
    "qwen3_marked",  # alias to template for backward compatibility
    "qwen3_marked_legacy",
    "qwen3_chat",
    "auto",
)

_QWEN3_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _resolve_pair_format(pair_format: str, model_config: Any, model_name: str) -> str:
    if pair_format != "auto":
        return pair_format
    model_type = getattr(model_config, "model_type", None)
    if model_type == "qwen3":
        return "qwen3_template"
    for arch in getattr(model_config, "architectures", None) or []:
        if "Qwen3" in str(arch):
            return "qwen3_template"
    if "qwen3" in model_name.lower():
        return "qwen3_template"
    return "hf_pair"


def _format_pair(
    query: str, passage: str, pair_format: str
) -> Union[Tuple[str, str], str]:
    if pair_format == "hf_pair":
        return (query, passage)
    if pair_format == "qwen3_marked_legacy":
        return (f"<Query>: {query}", f"<Document>: {passage}")
    if pair_format in ("qwen3_template", "qwen3_marked", "qwen3_chat"):
        return (
            f"{_QWEN3_PREFIX}"
            f"<Instruct>: {_QWEN3_INSTRUCTION}\n<Query>: {query}\n<Document>: {passage}{_QWEN3_SUFFIX}"
        )
    raise ValueError(f"Unknown pair_format: {pair_format}")


def _format_qwen3_template(
    tokenizer: Any, max_length: int, query: str, passage: str
) -> str:
    prefix = (
        f"{_QWEN3_PREFIX}"
        f"<Instruct>: {_QWEN3_INSTRUCTION}\n<Query>: {query}\n<Document>: "
    )
    suffix = _QWEN3_SUFFIX
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    doc_ids = tokenizer(passage, add_special_tokens=False).input_ids
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    available = max_length - len(prefix_ids) - len(suffix_ids) - special_tokens
    if available < 0:
        available = 0
    if len(doc_ids) > available:
        doc_ids = doc_ids[:available]
    doc_text = tokenizer.decode(doc_ids, skip_special_tokens=True)
    return f"{prefix}{doc_text}{suffix}"


class Reranker:
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        max_length: int,
        pair_format: str = "hf_pair",
    ) -> None:
        self.device = torch.device(device)
        self.max_length = max_length
        self.pair_format_requested = pair_format

        adapter_path = Path(model_name_or_path)
        pair_format_from_adapter: Optional[str] = None
        if adapter_path.is_dir() and (adapter_path / "adapter_config.json").exists():
            adapter_cfg = json.loads(
                (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
            )
            base_name = adapter_cfg.get("base_model_name_or_path")
            if not base_name:
                raise ValueError(
                    f"adapter_config.json missing base_model_name_or_path: {adapter_path}"
                )
            ta_path = adapter_path.parent / "training_args.json"
            if ta_path.exists():
                try:
                    ta_payload = json.loads(ta_path.read_text(encoding="utf-8"))
                    pf = ta_payload.get("pair_format")
                    if isinstance(pf, str):
                        pair_format_from_adapter = pf
                except Exception:
                    pair_format_from_adapter = None
            tokenizer_path = adapter_path.parent / "tokenizer"
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path if tokenizer_path.exists() else base_name)
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(base_name)
            self.model = PeftModel.from_pretrained(self.model, model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        resolved = _resolve_pair_format(pair_format, self.model.config, model_name_or_path)
        if pair_format == "auto" and pair_format_from_adapter:
            self.pair_format = pair_format_from_adapter
        else:
            self.pair_format = resolved
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
        batch_size: int,
    ) -> List[float]:
        scores: List[float] = []
        with torch.inference_mode():
            for start in range(0, len(queries), batch_size):
                end = start + batch_size
                if self.pair_format in ("hf_pair", "qwen3_marked_legacy"):
                    formatted_queries: List[str] = []
                    formatted_passages: List[str] = []
                    for query, passage in zip(queries[start:end], passages[start:end]):
                        fmt_query, fmt_passage = _format_pair(
                            query, passage, self.pair_format
                        )
                        formatted_queries.append(fmt_query)
                        formatted_passages.append(fmt_passage)
                    truncation = "only_second" if self.pair_format == "qwen3_marked_legacy" else True
                    batch = self.tokenizer(
                        formatted_queries,
                        formatted_passages,
                        padding=True,
                        truncation=truncation,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                else:
                    formatted_texts: List[str] = []
                    for query, passage in zip(queries[start:end], passages[start:end]):
                        formatted_texts.append(
                            _format_qwen3_template(
                                self.tokenizer, self.max_length, query, passage
                            )
                        )
                    batch = self.tokenizer(
                        formatted_texts,
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                batch_scores = _scores_from_logits(outputs.logits, self.model.config)
                scores.extend(batch_scores.detach().cpu().tolist())
        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        queries = [query] * len(candidates)
        passages = [c.get("text") or "" for c in candidates]
        scores = self.score_pairs(queries, passages, batch_size)
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        for idx, cand in enumerate(reranked, start=1):
            cand["rerank_rank"] = idx
        return reranked


def _build_gold_maps(
    gold_records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, str]], Dict[int, Dict[str, str]], Dict[int, str]]:
    label_map: Dict[int, Dict[str, str]] = {}
    text_map: Dict[int, Dict[str, str]] = {}
    query_map: Dict[int, str] = {}
    for rec in gold_records:
        qid = int(rec.get("query_id"))
        query_map[qid] = str(rec.get("query") or "").strip()
        label_map.setdefault(qid, {})
        text_map.setdefault(qid, {})
        for cand in _get_candidates(rec):
            cid = str(cand.get("chunk_id"))
            label = cand.get("label") or "irrelevant"
            label_map[qid][cid] = label
            text_map[qid][cid] = cand.get("text") or ""
    return label_map, text_map, query_map


def _index_results(
    records: Sequence[Dict[str, Any]],
    gold_qids: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    qid_set = set(gold_qids)
    indexed: Dict[int, Dict[str, Any]] = {}
    for rec in records:
        try:
            qid = int(rec.get("query_id"))
        except (TypeError, ValueError):
            continue
        if qid not in qid_set:
            continue
        indexed[qid] = rec
    return indexed


def _candidate_labels(
    hits: Sequence[Dict[str, Any]],
    label_lookup: Dict[str, str],
) -> List[str]:
    labels: List[str] = []
    for hit in hits:
        cid = str(hit.get("chunk_id"))
        labels.append(label_lookup.get(cid, "irrelevant"))
    return labels


def _evaluate_group(
    results_by_qid: Dict[int, Dict[str, Any]],
    label_map: Dict[int, Dict[str, str]],
    k: int,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, Any]]]:
    per_query: Dict[int, Dict[str, Any]] = {}
    mrrs: List[float] = []
    ndcgs: List[float] = []
    precisions: List[float] = []
    unjudged_count = 0
    total_count = 0

    for qid, rec in results_by_qid.items():
        hits = _sorted_candidates(_get_candidates(rec))
        label_lookup = label_map.get(qid, {})
        top_hits = hits[:k]
        labels = _candidate_labels(top_hits, label_lookup)
        total_count += len(labels)
        for hit in top_hits:
            cid = str(hit.get("chunk_id"))
            if cid not in label_lookup:
                unjudged_count += 1
        metrics = _metrics_from_labels(labels, k)
        per_query[qid] = {
            "mrr@10": metrics["mrr@10"],
            "ndcg@10": metrics["ndcg@10"],
            "p@10": metrics["p@10"],
            "top_k_chunk_ids": [h.get("chunk_id") for h in hits[:k]],
            "top_k_labels": labels,
        }
        mrrs.append(metrics["mrr@10"])
        ndcgs.append(metrics["ndcg@10"])
        precisions.append(metrics["p@10"])

    summary = {
        "mrr@10": _safe_mean(mrrs),
        "ndcg@10": _safe_mean(ndcgs),
        "p@10": _safe_mean(precisions),
        "unjudged_rate": (unjudged_count / total_count) if total_count else 0.0,
    }
    return summary, per_query


def _get_chunk_text(index: Optional[ChunkIndex], chunk_id: str) -> str:
    if not index:
        return ""
    chunk = index.get(chunk_id)
    if not chunk:
        return ""
    return chunk.get("content") or chunk.get("text") or ""


def _rerank_results(
    results_by_qid: Dict[int, Dict[str, Any]],
    query_map: Dict[int, str],
    text_map: Dict[int, Dict[str, str]],
    chunk_index: Optional[ChunkIndex],
    reranker: Reranker,
    batch_size: int,
    progress_label: Optional[str] = None,
    log_every: int = 50,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, float]]:
    reranked: Dict[int, Dict[str, Any]] = {}
    missing_text_count = 0
    total_candidates = 0
    total_qids = len(results_by_qid)
    for idx, (qid, rec) in enumerate(results_by_qid.items(), start=1):
        query = query_map.get(qid) or str(rec.get("query") or "").strip()
        hits = _sorted_candidates(_get_candidates(rec))
        candidates: List[Dict[str, Any]] = []
        for hit in hits:
            cid = str(hit.get("chunk_id"))
            text = text_map.get(qid, {}).get(cid) or _get_chunk_text(chunk_index, cid)
            if not text:
                missing_text_count += 1
            total_candidates += 1
            candidates.append(
                {
                    "chunk_id": cid,
                    "rank": hit.get("rank"),
                    "score": hit.get("score"),
                    "text": text or "",
                }
            )
        reranked_hits = reranker.rerank(query, candidates, batch_size)
        reranked[qid] = {
            "query_id": qid,
            "query": query,
            "hits": reranked_hits,
        }
        if progress_label and (idx == 1 or idx == total_qids or idx % log_every == 0):
            _log(f"{progress_label}: processed {idx}/{total_qids} queries")
    missing_rate = (missing_text_count / total_candidates) if total_candidates else 0.0
    stats = {
        "missing_text_count": missing_text_count,
        "missing_text_rate": missing_rate,
        "total_candidates": total_candidates,
    }
    return reranked, stats


def _build_report(
    stage: str,
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    def _fmt(value: float) -> str:
        return f"{value:.4f}"

    embed = metrics["embedding_only"]
    hybrid = metrics["hybrid_rrf"]
    base = metrics["hybrid_base_reranker"]
    ft = metrics["hybrid_finetuned_reranker"]

    lines = [
        f"# {stage} Reranker Evaluation Report",
        "",
        "## 1. Metrics",
        "",
        "| Group | MRR@10 | NDCG@10 | P@10 |",
        "| --- | --- | --- | --- |",
        f"| 1. Embedding-only | {_fmt(embed['mrr@10'])} | {_fmt(embed['ndcg@10'])} | {_fmt(embed['p@10'])} |",
        f"| 2. Hybrid (RRF) | {_fmt(hybrid['mrr@10'])} | {_fmt(hybrid['ndcg@10'])} | {_fmt(hybrid['p@10'])} |",
        f"| 3. Hybrid + Base Reranker | {_fmt(base['mrr@10'])} | {_fmt(base['ndcg@10'])} | {_fmt(base['p@10'])} |",
        f"| 4. Hybrid + Fine-tuned Reranker | {_fmt(ft['mrr@10'])} | {_fmt(ft['ndcg@10'])} | {_fmt(ft['p@10'])} |",
        "",
        "## 2. Improvements",
        "",
        f"- Hybrid vs Embedding-only: MRR {_fmt(hybrid['mrr@10'] - embed['mrr@10'])}",
        f"- Fine-tuned vs Base: MRR {_fmt(ft['mrr@10'] - base['mrr@10'])}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 10 evaluation")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--config", default="config/weak_supervision_config.yaml")
    parser.add_argument("--gold-eval", default=None)
    parser.add_argument("--embedding-results", default=None)
    parser.add_argument("--hybrid-results", default=None)
    parser.add_argument("--chunks-dir", default=None)
    parser.add_argument("--base-reranker", required=True)
    parser.add_argument("--finetuned-reranker", required=True)
    parser.add_argument("--output-dir", default="data/output/eval")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output prefix for eval artifacts. Default is 'auto' (stage + timestamp). "
            "Use 'final' to reuse the legacy stage name."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--pair-format",
        type=str,
        default="auto",
        choices=_PAIR_FORMAT_CHOICES,
        help=(
            "Input format for query-passage pairs. "
            "hf_pair: tokenizer(query, passage). "
            "qwen3_template/qwen3_chat: full chat template for Qwen3 reranker (system/user/assistant + <think>). "
            "qwen3_marked: alias to qwen3_template. "
            "qwen3_marked_legacy: legacy <Query>/<Document> markers (for A/B only). "
            "auto (default): detect based on model config or adapter metadata."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--checkpoint-suffix",
        default=None,
        help=(
            "Suffix for eval checkpoint filename. Default is 'auto' (timestamp). "
            "Use 'final' to keep the legacy name, or provide a custom suffix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_yaml(Path(args.config)) if args.config else {}
    data_cfg = cfg.get("data", {})

    stage = args.stage
    run_stamp = _now_stamp()
    output_prefix_requested = (args.output_prefix or "").strip() or "auto"
    output_prefix = _resolve_output_prefix(stage, output_prefix_requested, run_stamp)
    checkpoint_suffix_requested = (args.checkpoint_suffix or "").strip() or "auto"
    checkpoint_suffix_used: Optional[str] = None
    if args.save_checkpoint:
        checkpoint_suffix_used = _resolve_checkpoint_suffix(
            checkpoint_suffix_requested, run_stamp
        )
    _log(f"Stage: {stage}, output prefix: {output_prefix}")

    retrieval_dir = Path(data_cfg.get("retrieval_output", "data/output/retrieval"))
    annotations_dir = Path(data_cfg.get("annotations_output", "data/output/annotations"))
    chunks_dir = Path(args.chunks_dir or data_cfg.get("chunk_output", "data/output/chunks"))

    gold_eval = Path(args.gold_eval or annotations_dir / "gold_eval_50_extended_final.jsonl")
    embedding_results = Path(
        args.embedding_results or retrieval_dir / f"embedding_top20_{stage}.jsonl"
    )
    hybrid_results = Path(
        args.hybrid_results or retrieval_dir / f"hybrid_rrf_top20_{stage}.jsonl"
    )

    _log(f"Loading gold eval from {gold_eval}")
    gold_records = _load_jsonl(gold_eval)
    if not gold_records:
        raise RuntimeError(f"gold eval is empty: {gold_eval}")

    label_map, text_map, query_map = _build_gold_maps(gold_records)
    gold_qids = sorted(label_map.keys())
    _log(f"Loaded {len(gold_qids)} gold queries")

    _log("Loading retrieval results (embedding/hybrid)")
    embed_records = _load_jsonl(embedding_results)
    hybrid_records = _load_jsonl(hybrid_results)
    embed_by_qid = _index_results(embed_records, gold_qids)
    hybrid_by_qid = _index_results(hybrid_records, gold_qids)

    _log("Computing metrics for embedding-only and hybrid baselines")
    embed_metrics, embed_per_query = _evaluate_group(embed_by_qid, label_map, k=10)
    hybrid_metrics, hybrid_per_query = _evaluate_group(hybrid_by_qid, label_map, k=10)

    device = _resolve_device(args.device)
    _log(f"Using device: {device}")
    if chunks_dir.exists():
        _log(f"Loading chunk index from {chunks_dir}")
    chunk_index = load_chunks(chunks_dir) if chunks_dir.exists() else None

    _log(f"Loading base reranker: {args.base_reranker}")
    try:
        base_reranker = Reranker(
            args.base_reranker,
            device=device,
            max_length=args.max_length,
            pair_format=args.pair_format,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load base reranker '{args.base_reranker}'. "
            "Make sure the model is available locally or cached."
        ) from exc
    _log("Reranking with base reranker")
    base_reranked, base_text_stats = _rerank_results(
        hybrid_by_qid,
        query_map,
        text_map,
        chunk_index,
        base_reranker,
        args.batch_size,
        progress_label="Base reranker",
    )
    base_metrics, base_per_query = _evaluate_group(base_reranked, label_map, k=10)

    _log(f"Loading finetuned reranker: {args.finetuned_reranker}")
    try:
        ft_reranker = Reranker(
            args.finetuned_reranker,
            device=device,
            max_length=args.max_length,
            pair_format=args.pair_format,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load finetuned reranker '{args.finetuned_reranker}'. "
            "Make sure the adapter path is correct and locally available."
        ) from exc
    _log("Reranking with finetuned reranker")
    ft_reranked, ft_text_stats = _rerank_results(
        hybrid_by_qid,
        query_map,
        text_map,
        chunk_index,
        ft_reranker,
        args.batch_size,
        progress_label="Finetuned reranker",
    )
    ft_metrics, ft_per_query = _evaluate_group(ft_reranked, label_map, k=10)

    metrics = {
        "embedding_only": embed_metrics,
        "hybrid_rrf": hybrid_metrics,
        "hybrid_base_reranker": base_metrics,
        "hybrid_finetuned_reranker": ft_metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_text_stats["missing_text_count"] > 0:
        print(
            (
                "Warning: base reranker missing text for "
                f"{int(base_text_stats['missing_text_count'])} candidates "
                f"({base_text_stats['missing_text_rate']:.2%})."
            ),
            file=sys.stderr,
        )
    if ft_text_stats["missing_text_count"] > 0:
        print(
            (
                "Warning: finetuned reranker missing text for "
                f"{int(ft_text_stats['missing_text_count'])} candidates "
                f"({ft_text_stats['missing_text_rate']:.2%})."
            ),
            file=sys.stderr,
        )

    _build_report(
        stage=stage,
        metrics=metrics,
        output_path=output_dir / f"eval_report_{output_prefix}.md",
    )
    (output_dir / f"metrics_comparison_{output_prefix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"Metrics and reports written to {output_dir}")

    per_query_records: List[Dict[str, Any]] = []
    for qid in gold_qids:
        per_query_records.append(
            {
                "query_id": qid,
                "query": query_map.get(qid, ""),
                "embedding_only": embed_per_query.get(qid, {}),
                "hybrid_rrf": hybrid_per_query.get(qid, {}),
                "hybrid_base_reranker": base_per_query.get(qid, {}),
                "hybrid_finetuned_reranker": ft_per_query.get(qid, {}),
            }
        )
    _write_jsonl(output_dir / f"per_query_scores_{output_prefix}.jsonl", per_query_records)

    degraded: List[Dict[str, Any]] = []
    for qid in gold_qids:
        base_ndcg = base_per_query.get(qid, {}).get("ndcg@10", 0.0)
        ft_ndcg = ft_per_query.get(qid, {}).get("ndcg@10", 0.0)
        degraded.append(
            {
                "query_id": qid,
                "query": query_map.get(qid, ""),
                "base_ndcg@10": base_ndcg,
                "finetuned_ndcg@10": ft_ndcg,
                "delta_ndcg@10": ft_ndcg - base_ndcg,
                "base_top1_chunk_id": (
                    base_per_query.get(qid, {}).get("top_k_chunk_ids", [])[:1]
                ),
                "finetuned_top1_chunk_id": (
                    ft_per_query.get(qid, {}).get("top_k_chunk_ids", [])[:1]
                ),
            }
        )
    degraded.sort(key=lambda x: x["delta_ndcg@10"])
    _write_jsonl(
        output_dir / f"degraded_cases_{output_prefix}_top10.jsonl",
        degraded[:10],
    )

    eval_config = {
        "stage": stage,
        "gold_eval_file": str(gold_eval),
        "gold_eval_count": len(gold_qids),
        "embedding_results": str(embedding_results),
        "hybrid_results": str(hybrid_results),
        "base_reranker": args.base_reranker,
        "finetuned_reranker": args.finetuned_reranker,
        "device": device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pair_format_requested": args.pair_format,
        "pair_format": base_reranker.pair_format,
        "base_pair_format": base_reranker.pair_format,
        "finetuned_pair_format": ft_reranker.pair_format,
        "checkpoint_suffix_requested": checkpoint_suffix_requested,
        "checkpoint_suffix_used": checkpoint_suffix_used,
        "base_text_stats": base_text_stats,
        "finetuned_text_stats": ft_text_stats,
        "created_at": _now_iso(),
    }
    (output_dir / f"eval_config_{output_prefix}.json").write_text(
        json.dumps(eval_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_checkpoint:
        if checkpoint_suffix_used:
            ckpt_name = f"{stage}_step_10_eval_{checkpoint_suffix_used}.json"
        else:
            ckpt_name = f"{stage}_step_10_eval.json"
        ckpt = {
            "stage": stage,
            "step": 10,
            "completed": True,
            "completed_at": _now_iso(),
            "gold_eval_file": str(gold_eval),
            "gold_eval_count": len(gold_qids),
            "metrics_comparison": metrics,
            "degraded_cases_count": min(10, len(degraded)),
            "base_reranker": args.base_reranker,
            "finetuned_reranker": args.finetuned_reranker,
            "evaluation_config": eval_config,
        }
        ckpt_path = Path("data/output/checkpoints") / ckpt_name
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")

    _log("Evaluation complete.")


if __name__ == "__main__":
    main()
