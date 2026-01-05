#!/usr/bin/env python3
"""
对 Gemini/Qwen/Codex 多模型的 Gold Eval 标注结果进行裁决，遵循最小变更规则。

输出文件：
- Extended（保留所有候选，给出最终标签）
- Core（仅包含全体一致通过的候选）
- Adjudication sidecar（附加：各候选的原始投票及所用裁决规则）
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LABELS = {"evidence", "related", "irrelevant"}


def normalize_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, str):
        value = label.strip().lower()
        if value.startswith("evid"):
            return "evidence"
        if value.startswith("rel"):
            return "related"
        if value.startswith("irr"):
            return "irrelevant"
        if value in LABELS:
            return value
    return None


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


def build_label_map(records: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Optional[str]]:
    label_map: Dict[Tuple[int, str], Optional[str]] = {}
    for rec in records:
        qid = rec.get("query_id")
        if qid is None:
            continue
        for cand in rec.get("candidates") or []:
            key = (qid, cand.get("chunk_id"))
            label_map[key] = normalize_label(cand.get("label"))
    return label_map


def load_gemini_csv(path: Path, template_keys: set[Tuple[int, str]]) -> Dict[Tuple[int, str], Dict[str, str]]:
    rows: Dict[Tuple[int, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["query_id"]) if row.get("query_id") else None
            chunk = row.get("chunk_id")
            if qid is None or not chunk:
                continue
            if chunk.isdigit() and len(chunk) < 8:
                padded = chunk.zfill(8)
                if (qid, padded) in template_keys:
                    chunk = padded
            rows[(qid, chunk)] = {
                "label": normalize_label(row.get("label")),
                "notes": (row.get("notes") or "").strip(),
                "query": (row.get("query") or "").strip(),
            }
    return rows


def decide_label(
    votes: Dict[str, Optional[str]],
    *,
    all_diff_policy: str,
) -> Tuple[str, str]:
    counts = Counter(label for label in votes.values() if label in LABELS)
    if counts.get("evidence", 0) >= 2:
        return ("evidence", "majority_evidence")
    if counts.get("irrelevant", 0) >= 2:
        return ("irrelevant", "majority_irrelevant")
    if counts.get("related", 0) >= 2:
        return ("related", "majority_related")

    # All-diff or incomplete votes
    if all_diff_policy == "related":
        return ("related", "all_diff_related")
    gemini_label = votes.get("gemini")
    if gemini_label in LABELS:
        return (gemini_label, "all_diff_gemini")
    return ("related", "all_diff_fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate gold-eval labels with minimal-change rules.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--gemini-csv", type=Path, required=True)
    parser.add_argument("--qwen-jsonl", type=Path, required=True)
    parser.add_argument("--codex-jsonl", type=Path, required=True)
    parser.add_argument("--extended-out", type=Path, required=True)
    parser.add_argument("--core-out", type=Path, required=True)
    parser.add_argument("--adjudication-out", type=Path, required=True)
    parser.add_argument(
        "--all-diff-policy",
        choices=["gemini", "related"],
        default="gemini",
        help="Policy when votes are all different. Default: gemini.",
    )
    parser.add_argument(
        "--notes-policy",
        choices=["gemini_if_match", "gemini_always", "empty"],
        default="gemini_if_match",
        help="How to fill notes in output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    template_records = load_jsonl(args.template)
    template_keys = {
        (rec.get("query_id"), cand.get("chunk_id"))
        for rec in template_records
        for cand in (rec.get("candidates") or [])
    }

    gemini_rows = load_gemini_csv(args.gemini_csv, template_keys)
    qwen_labels = build_label_map(load_jsonl(args.qwen_jsonl))
    codex_labels = build_label_map(load_jsonl(args.codex_jsonl))

    stats = Counter()
    adjudication_rows: List[Dict[str, Any]] = []
    extended_records: List[Dict[str, Any]] = []
    core_records: List[Dict[str, Any]] = []

    for rec in template_records:
        qid = rec.get("query_id")
        candidates = rec.get("candidates") or []

        extended_candidates: List[Dict[str, Any]] = []
        core_candidates: List[Dict[str, Any]] = []

        for cand in candidates:
            key = (qid, cand.get("chunk_id"))
            gemini = gemini_rows.get(key, {})
            votes = {
                "gemini": gemini.get("label"),
                "qwen": qwen_labels.get(key),
                "codex": codex_labels.get(key),
            }

            final_label, rule = decide_label(votes, all_diff_policy=args.all_diff_policy)
            stats[f"final_{final_label}"] += 1
            stats[f"rule_{rule}"] += 1

            notes = ""
            gemini_notes = gemini.get("notes", "")
            if args.notes_policy == "gemini_always":
                notes = gemini_notes
            elif args.notes_policy == "gemini_if_match" and gemini.get("label") == final_label:
                notes = gemini_notes

            updated = dict(cand)
            updated["label"] = final_label
            updated["notes"] = notes
            extended_candidates.append(updated)

            # Core = unanimous only
            vote_values = [v for v in votes.values() if v in LABELS]
            if len(vote_values) == 3 and len(set(vote_values)) == 1:
                core_candidates.append(updated)
                stats["core_candidates"] += 1

            adjudication_rows.append(
                {
                    "query_id": qid,
                    "chunk_id": cand.get("chunk_id"),
                    "rank": cand.get("rank"),
                    "final_label": final_label,
                    "adjudication_rule": rule,
                    "raw_votes": votes,
                }
            )

        extended_records.append({**rec, "candidates": extended_candidates})
        core_records.append({**rec, "candidates": core_candidates})

    write_jsonl(args.extended_out, extended_records)
    write_jsonl(args.core_out, core_records)
    write_jsonl(args.adjudication_out, adjudication_rows)

    print("Adjudication complete.")
    print(f"Extended: {args.extended_out}")
    print(f"Core: {args.core_out}")
    print(f"Adjudication: {args.adjudication_out}")
    print(dict(stats))


if __name__ == "__main__":
    main()
