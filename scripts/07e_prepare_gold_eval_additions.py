#!/usr/bin/env python3
"""
Prepare Gold Eval addition templates by sampling new queries not in existing sets.

Outputs:
- Additions template JSONL (with chunk text)
- Optional merged blacklist JSON
- Optional report JSON
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


def load_excluded_from_template(path: Path) -> Set[int]:
    excluded: Set[int] = set()
    if not path.exists():
        return excluded
    for rec in load_jsonl(path):
        qid = rec.get("query_id")
        if qid is not None:
            excluded.add(qid)
    return excluded


def load_excluded_from_blacklist(path: Path) -> Set[int]:
    excluded: Set[int] = set()
    if not path.exists():
        return excluded
    payload = json.loads(path.read_text(encoding="utf-8"))
    for qid in payload.get("query_ids", []):
        excluded.add(qid)
    return excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Gold Eval additions (new queries only).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/output/retrieval/hybrid_rrf_top20_stage1.jsonl"),
        help="Hybrid Top20 retrieval JSONL.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/output/chunks"),
    )
    parser.add_argument("--add-size", type=int, default=8, help="Number of new queries to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--exclude-template",
        type=Path,
        default=None,
        help="Existing gold eval template JSONL to exclude.",
    )
    parser.add_argument(
        "--exclude-blacklist",
        type=Path,
        default=None,
        help="Existing eval blacklist JSON to exclude.",
    )
    parser.add_argument(
        "--exclude-dropped",
        type=Path,
        default=Path("config/gold_eval_dropped.json"),
        help="Dropped/no-positive query list JSON to exclude if present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Additions template output JSONL.",
    )
    parser.add_argument(
        "--blacklist-out",
        type=Path,
        default=None,
        help="Optional merged blacklist output JSON.",
    )
    parser.add_argument(
        "--blacklist-mode",
        choices=["merge", "additions_only"],
        default="merge",
        help="merge: union with excluded set; additions_only: write only new qids.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional report JSON path.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any chunk text is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or Path(f"data/output/annotations/gold_eval_50_template_additions_{args.add_size}.jsonl")
    report_path = args.report_out or Path(
        f"data/output/annotations/gold_eval_50_additions_{args.add_size}_report.json"
    )

    excluded: Set[int] = set()
    if args.exclude_template is not None:
        excluded |= load_excluded_from_template(args.exclude_template)
    if args.exclude_blacklist is not None:
        excluded |= load_excluded_from_blacklist(args.exclude_blacklist)
    if args.exclude_dropped is not None and args.exclude_dropped.exists():
        excluded |= load_excluded_from_blacklist(args.exclude_dropped)

    retrieval_records = load_jsonl(args.input)
    available = [rec for rec in retrieval_records if rec.get("query_id") not in excluded]

    if len(available) < args.add_size:
        raise RuntimeError(f"Not enough available queries ({len(available)}) for add-size {args.add_size}.")

    random.seed(args.seed)
    selected = random.sample(available, args.add_size)
    selected_qids = [rec.get("query_id") for rec in selected if rec.get("query_id") is not None]

    chunk_index = load_chunks(args.chunks_dir)
    missing_chunks: List[str] = []
    output_records: List[Dict[str, Any]] = []

    for item in selected:
        hits = item.get("hits", []) or []
        candidates = [build_candidate(hit, chunk_index, missing_chunks) for hit in hits[:20]]
        pdf_stem = None
        if hits:
            pdf_stem = hits[0].get("pdf_stem") or hits[0].get("pdf")
        pdf_stem = pdf_stem or item.get("pdf")
        output_records.append(
            {
                "stage": item.get("stage"),
                "query_id": item.get("query_id"),
                "query": item.get("query"),
                "pdf": item.get("pdf"),
                "pdf_stem": pdf_stem,
                "company": item.get("company"),
                "year": item.get("year"),
                "answers": item.get("answers", []),
                "candidates": candidates,
            }
        )

    write_jsonl(output_path, output_records)

    if missing_chunks:
        warn_msg = f"[WARNING] missing {len(set(missing_chunks))} chunks: {list(set(missing_chunks))[:10]}"
        print(warn_msg)
        if args.strict_missing:
            raise RuntimeError(warn_msg)

    if args.blacklist_out is not None:
        base_ids = excluded if args.blacklist_mode == "merge" else set()
        merged = set(base_ids) | set(selected_qids) if args.blacklist_mode == "merge" else set(selected_qids)
        blacklist = {
            "query_ids": sorted(merged),
            "count": len(merged),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": f"gold_eval_additions_{args.add_size}",
        }
        args.blacklist_out.parent.mkdir(parents=True, exist_ok=True)
        args.blacklist_out.write_text(json.dumps(blacklist, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "input_retrieval": str(args.input),
        "exclude_template": str(args.exclude_template) if args.exclude_template else None,
        "exclude_blacklist": str(args.exclude_blacklist) if args.exclude_blacklist else None,
        "exclude_dropped": str(args.exclude_dropped) if args.exclude_dropped else None,
        "output_template": str(output_path),
        "blacklist_out": str(args.blacklist_out) if args.blacklist_out else None,
        "add_size": args.add_size,
        "seed": args.seed,
        "excluded_qids": sorted(excluded),
        "sampled_qids": sorted(selected_qids),
        "missing_chunks": sorted(set(missing_chunks)),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Additions sample complete.")
    print(f"Sampled: {sorted(selected_qids)}")
    print(f"Output: {output_path}")
    if args.blacklist_out is not None:
        print(f"Blacklist: {args.blacklist_out}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
