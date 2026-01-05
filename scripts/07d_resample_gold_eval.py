#!/usr/bin/env python3
"""
Resample Gold Eval queries by dropping no-positive queries and sampling replacements.

Outputs:
- New gold_eval template JSONL (with chunk text)
- New eval blacklist JSON
- Resample report JSON
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunk_manager import load_chunks  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def build_candidate(hit: Dict[str, Any], chunk_index: Dict[str, Dict[str, Any]], missing: List[str]) -> Dict[str, Any]:
    chunk = chunk_index.get(hit.get("chunk_id"))
    if chunk is None:
        missing.append(hit.get("chunk_id"))
    text = None
    if chunk is not None:
        text = chunk.get("text") or chunk.get("content")
    return {
        "chunk_id": hit.get("chunk_id"),
        "parent_id": hit.get("parent_id"),
        "pdf": hit.get("pdf"),
        "pdf_stem": hit.get("pdf_stem"),
        "page_numbers": hit.get("page_numbers"),
        "section_path": hit.get("section_path"),
        "score": hit.get("score"),
        "rank": hit.get("rank"),
        "text": text,
        "label": None,
        "notes": "",
    }


def compute_no_positive_qids(extended_path: Path) -> Set[int]:
    extended = load_jsonl(extended_path)
    no_positive: Set[int] = set()
    for rec in extended:
        qid = rec.get("query_id")
        if qid is None:
            continue
        cands = rec.get("candidates") or []
        pos = sum(1 for c in cands if c.get("label") in ("evidence", "related"))
        if pos == 0:
            no_positive.add(qid)
    return no_positive


def load_dropped_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("query_ids", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample Gold Eval queries by dropping no-positive queries.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/output/retrieval/hybrid_rrf_top20_stage1.jsonl"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/output/annotations/gold_eval_50_template.jsonl"),
    )
    parser.add_argument(
        "--extended",
        type=Path,
        default=Path("data/output/annotations/gold_eval_50_extended.jsonl"),
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/output/chunks"),
    )
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/annotations/gold_eval_50_template_resampled.jsonl"),
    )
    parser.add_argument(
        "--blacklist-out",
        type=Path,
        default=Path("config/eval_blacklist_resampled.json"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/output/annotations/gold_eval_50_resample_report.json"),
    )
    parser.add_argument(
        "--exclude-dropped",
        type=Path,
        default=Path("config/gold_eval_dropped.json"),
        help="Dropped/no-positive query list JSON to exclude if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retrieval_records = load_jsonl(args.input)
    template_records = load_jsonl(args.template)

    current_qids = {rec.get("query_id") for rec in template_records if rec.get("query_id") is not None}
    no_positive_qids = compute_no_positive_qids(args.extended)
    dropped_qids = load_dropped_ids(args.exclude_dropped)

    drop_qids = sorted(current_qids & no_positive_qids)
    keep_qids = sorted(current_qids - set(drop_qids))

    all_qids = [rec.get("query_id") for rec in retrieval_records if rec.get("query_id") is not None]
    available_qids = [qid for qid in all_qids if qid not in current_qids and qid not in dropped_qids]

    if len(drop_qids) == 0:
        raise RuntimeError("No no-positive queries found to drop.")
    if len(available_qids) < len(drop_qids):
        raise RuntimeError("Not enough available queries to resample replacements.")

    random.seed(args.seed)
    sampled_qids = sorted(random.sample(available_qids, len(drop_qids)))
    new_qids = sorted(set(keep_qids) | set(sampled_qids))

    if len(new_qids) != args.size:
        raise RuntimeError(f"Expected size {args.size} but got {len(new_qids)}.")

    chunk_index = load_chunks(args.chunks_dir)
    missing_chunks: List[str] = []

    output_records: List[Dict[str, Any]] = []
    selected = {qid for qid in new_qids}
    for item in retrieval_records:
        qid = item.get("query_id")
        if qid not in selected:
            continue
        hits = item.get("hits", []) or []
        candidates = [build_candidate(hit, chunk_index, missing_chunks) for hit in hits[:20]]
        pdf_stem = None
        if hits:
            pdf_stem = hits[0].get("pdf_stem") or hits[0].get("pdf")
        pdf_stem = pdf_stem or item.get("pdf")
        output_records.append(
            {
                "stage": item.get("stage"),
                "query_id": qid,
                "query": item.get("query"),
                "pdf": item.get("pdf"),
                "pdf_stem": pdf_stem,
                "company": item.get("company"),
                "year": item.get("year"),
                "answers": item.get("answers", []),
                "candidates": candidates,
            }
        )

    if len(output_records) != args.size:
        raise RuntimeError(f"Expected {args.size} output records but got {len(output_records)}.")

    write_jsonl(args.output, output_records)

    blacklist = {
        "query_ids": sorted(new_qids),
        "count": len(new_qids),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"gold_eval_{args.size}_resampled",
    }
    args.blacklist_out.parent.mkdir(parents=True, exist_ok=True)
    args.blacklist_out.write_text(json.dumps(blacklist, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "input_retrieval": str(args.input),
        "input_template": str(args.template),
        "input_extended": str(args.extended),
        "output_template": str(args.output),
        "blacklist_out": str(args.blacklist_out),
        "size": args.size,
        "seed": args.seed,
        "dropped_qids": drop_qids,
        "exclude_dropped": str(args.exclude_dropped),
        "excluded_dropped_qids": sorted(dropped_qids),
        "sampled_qids": sampled_qids,
        "kept_qids": keep_qids,
        "missing_chunks": sorted(set(missing_chunks)),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Resample complete.")
    print(f"Dropped: {drop_qids}")
    print(f"Sampled: {sampled_qids}")
    print(f"Output: {args.output}")
    print(f"Blacklist: {args.blacklist_out}")
    print(f"Report: {args.report_out}")


if __name__ == "__main__":
    main()
