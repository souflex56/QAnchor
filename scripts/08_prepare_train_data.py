#!/usr/bin/env python3
"""Step 8: Clean mined triplets and split train/dev by query_id.

Rules:
- filter eval blacklist (config/eval_blacklist.json)
- filter confidence < threshold
- split by query_id (default 90/10) with fixed seed
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _beijing_now_iso() -> str:
    tz_bj = timezone(timedelta(hours=8))
    return datetime.now(tz_bj).isoformat()


def _normalize_qid(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        return str(value)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def load_blacklist(path: Path) -> Tuple[Set[Any], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_ids = payload.get("query_ids", [])
    ids: Set[Any] = set()
    for qid in raw_ids:
        norm = _normalize_qid(qid)
        if norm is not None:
            ids.add(norm)
    meta = {
        "path": str(path),
        "count": payload.get("count", len(ids)),
        "source": payload.get("source", "unknown"),
        "updated_at": payload.get("updated_at"),
    }
    return ids, meta


def split_qids(qids: List[Any], train_ratio: float, seed: int) -> Tuple[Set[Any], Set[Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    rng = random.Random(seed)
    qids_copy = list(qids)
    rng.shuffle(qids_copy)
    split_idx = int(len(qids_copy) * train_ratio)
    train_qids = set(qids_copy[:split_idx])
    dev_qids = set(qids_copy[split_idx:])
    return train_qids, dev_qids


def build_checkpoint(
    *,
    stage: str,
    input_path: Path,
    blacklist_path: Path,
    output_train: Path,
    output_dev: Path,
    params: Dict[str, Any],
    stats: Dict[str, Any],
    blacklist_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "step": 8,
        "completed": True,
        "completed_at": _beijing_now_iso(),
        "input_files": {
            "triplets": str(input_path),
            "blacklist": str(blacklist_path),
        },
        "output_files": {
            "train_triplets": str(output_train),
            "dev_triplets": str(output_dev),
        },
        "params": params,
        "stats": stats,
        "blacklist_meta": blacklist_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 8: prepare train/dev triplets.")
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Triplets JSONL. Default: data/output/mining/mined_triplets_{stage}.jsonl",
    )
    parser.add_argument(
        "--blacklist",
        type=Path,
        default=Path("config/eval_blacklist.json"),
        help="Eval blacklist JSON. Default: config/eval_blacklist.json",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Drop triplets with confidence < threshold.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train ratio by query_id.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-train",
        type=Path,
        default=None,
        help="Train output JSONL. Default: data/output/train/train_triplets_{stage}.jsonl",
    )
    parser.add_argument(
        "--output-dev",
        type=Path,
        default=None,
        help="Dev output JSONL. Default: data/output/train/dev_triplets_{stage}.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint JSON. Default: data/output/checkpoints/{stage}_step_8_train_data.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute stats only; no file writes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input or Path(f"data/output/mining/mined_triplets_{args.stage}.jsonl")
    output_train = args.output_train or Path(f"data/output/train/train_triplets_{args.stage}.jsonl")
    output_dev = args.output_dev or Path(f"data/output/train/dev_triplets_{args.stage}.jsonl")
    checkpoint_path = args.checkpoint or Path(f"data/output/checkpoints/{args.stage}_step_8_train_data.json")

    triplets = load_jsonl(input_path)
    blacklist_ids, blacklist_meta = load_blacklist(args.blacklist)

    filtered: List[Dict[str, Any]] = []
    filtered_blacklist = 0
    filtered_confidence = 0
    filtered_missing_qid = 0

    for rec in triplets:
        qid = _normalize_qid(rec.get("query_id"))
        if qid is None:
            filtered_missing_qid += 1
            continue
        if qid in blacklist_ids:
            filtered_blacklist += 1
            continue
        try:
            confidence = float(rec.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if confidence < args.confidence_threshold:
            filtered_confidence += 1
            continue
        filtered.append(rec)

    unique_qids = sorted({_normalize_qid(rec.get("query_id")) for rec in filtered if _normalize_qid(rec.get("query_id")) is not None})
    train_qids, dev_qids = split_qids(unique_qids, args.train_ratio, args.seed)

    train_records: List[Dict[str, Any]] = []
    dev_records: List[Dict[str, Any]] = []

    for rec in filtered:
        qid = _normalize_qid(rec.get("query_id"))
        if qid in dev_qids:
            dev_records.append(rec)
        else:
            train_records.append(rec)

    stats = {
        "input_triplets": len(triplets),
        "filtered_blacklist": filtered_blacklist,
        "filtered_confidence": filtered_confidence,
        "filtered_missing_qid": filtered_missing_qid,
        "kept_triplets": len(filtered),
        "unique_qids": len(unique_qids),
        "train_qids": len(train_qids),
        "dev_qids": len(dev_qids),
        "train_triplets": len(train_records),
        "dev_triplets": len(dev_records),
    }

    params = {
        "confidence_threshold": args.confidence_threshold,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "split_by": "query_id",
    }

    if not args.dry_run:
        write_jsonl(output_train, train_records)
        write_jsonl(output_dev, dev_records)
        checkpoint = build_checkpoint(
            stage=args.stage,
            input_path=input_path,
            blacklist_path=args.blacklist,
            output_train=output_train,
            output_dev=output_dev,
            params=params,
            stats=stats,
            blacklist_meta=blacklist_meta,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Step8] triplets:", stats["input_triplets"])
    print("[Step8] filtered_blacklist:", stats["filtered_blacklist"])
    print("[Step8] filtered_confidence:", stats["filtered_confidence"])
    print("[Step8] filtered_missing_qid:", stats["filtered_missing_qid"])
    print("[Step8] kept_triplets:", stats["kept_triplets"])
    print("[Step8] train_triplets:", stats["train_triplets"])
    print("[Step8] dev_triplets:", stats["dev_triplets"])
    if args.dry_run:
        print("[Step8] dry-run enabled; no files written.")


if __name__ == "__main__":
    main()
