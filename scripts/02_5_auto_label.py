#!/usr/bin/env python3
"""半自动标注脚本（Step 4a，Type1-only）。

输入：embedding 检索模板（nested JSONL）+ 标准答案（config 指向的 finglm_master.jsonl）
输出：在 hits 中追加 label_auto/match_type/match_evidence/matched_keys/confidence，写 nested JSONL。
标签规则：
- positive_auto: 匹配命中
- candidate: 未命中且 score>=min_candidate_score 或 rank<=max_candidate_rank
- negative_auto: 其他
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from tqdm import tqdm

# repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunk_manager import ChunkIndex, load_chunks  # noqa: E402
from src.answer_matcher import (  # noqa: E402
    MatchResult,
    extract_key_values,
    is_type1,
    match_chunk_to_answer,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_answers(path: Path) -> Dict[int, Dict[str, Any]]:
    idx: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mid = rec.get("master_id")
            if mid is None:
                continue
            try:
                idx[int(mid)] = rec
            except Exception:
                continue
    return idx


def _load_chunk_text(index: ChunkIndex, chunk_id: str) -> str:
    item = index.get(chunk_id)
    if not item:
        return ""
    text = item.get("content") or item.get("text") or ""
    return str(text)


def _label_hit(hit: Dict[str, Any], match: MatchResult, score: float, min_candidate_score: float, max_candidate_rank: int) -> Tuple[str, float]:
    if match.is_match:
        return "positive_auto", match.confidence

    rank = hit.get("rank")
    try:
        rank_val = int(rank)
    except Exception:
        rank_val = None

    is_candidate = score >= min_candidate_score or (rank_val is not None and rank_val <= max_candidate_rank)
    if is_candidate:
        return "candidate", score
    return "negative_auto", score


def _ensure_rank(hit: Dict[str, Any], idx: int) -> int:
    try:
        return int(hit.get("rank"))
    except Exception:
        pass
    hit["rank"] = idx + 1
    return hit["rank"]


def _build_checkpoint(
    *,
    stage: str,
    input_path: Path,
    output_path: Path,
    answers_path: Path,
    chunk_dir: Path,
    stats: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    beijing_tz = timezone(timedelta(hours=8))
    return {
        "stage": stage,
        "step": "4a",
        "completed": True,
        "completed_at": datetime.now(beijing_tz).isoformat(),
        "input_files": {
            "retrieval_template": str(input_path),
            "answers": str(answers_path),
            "chunks": str(chunk_dir),
        },
        "output_files": {
            "auto_label": str(output_path),
        },
        "params": params,
        "stats": stats,
    }


def run(
    stage: str,
    config_path: Path,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    min_candidate_score: float,
    max_candidate_rank: int,
    dry_run: bool = False,
    save_checkpoint: bool = False,
) -> None:
    config = yaml.safe_load(config_path.read_text())
    answers_path = Path(config["data"]["answers"])
    chunk_dir = Path(config["data"]["chunk_output"])

    answers_idx = _load_answers(answers_path)
    retrieval_records = load_jsonl(input_path)
    chunk_index = load_chunks(chunk_dir)

    total_hits = 0
    label_counts = {"positive_auto": 0, "candidate": 0, "negative_auto": 0}
    missing_answer = 0
    missing_chunk = 0
    match_queries = 0
    total_queries = 0
    positives_per_q: List[int] = []

    output_records: List[Dict[str, Any]] = []

    for rec in tqdm(retrieval_records, desc="auto-label", unit="query"):
        total_queries += 1
        qid = rec.get("query_id")
        try:
            qid_int = int(qid)
        except Exception:
            qid_int = None

        ans_rec = answers_idx.get(qid_int) if qid_int is not None else None
        if not ans_rec or not is_type1(ans_rec):
            missing_answer += 1
            output_records.append(rec)
            continue

        key_values = extract_key_values(ans_rec)
        hits = rec.get("hits") or []
        pos_cnt = 0

        for idx, hit in enumerate(hits):
            _ensure_rank(hit, idx)
            score = float(hit.get("score", 0.0))
            chunk_id = hit.get("chunk_id")
            chunk_text = _load_chunk_text(chunk_index, chunk_id) if chunk_id else ""
            if not chunk_text:
                missing_chunk += 1
                label, conf = _label_hit(hit, MatchResult(False, "none", [], "", 0.0), score, min_candidate_score, max_candidate_rank)
                hit.update({
                    "label_auto": label,
                    "match_type": "none",
                    "match_evidence": "",
                    "matched_keys": [],
                    "confidence": conf,
                })
                label_counts[label] += 1
                total_hits += 1
                continue

            match_res = match_chunk_to_answer(chunk_text, key_values)
            label, conf = _label_hit(hit, match_res, score, min_candidate_score, max_candidate_rank)
            if label == "positive_auto":
                pos_cnt += 1

            hit.update({
                "label_auto": label,
                "match_type": match_res.match_type,
                "match_evidence": match_res.match_evidence,
                "matched_keys": match_res.matched_keys,
                "confidence": conf,
            })

            label_counts[label] += 1
            total_hits += 1

        if pos_cnt > 0:
            match_queries += 1
        positives_per_q.append(pos_cnt)
        output_records.append(rec)

    stats = {
        "total_queries": total_queries,
        "total_hits": total_hits,
        "label_counts": label_counts,
        "missing_answer": missing_answer,
        "missing_chunk": missing_chunk,
        "match_rate": match_queries / total_queries if total_queries else 0.0,
        "avg_positive_per_query": (sum(positives_per_q) / len(positives_per_q)) if positives_per_q else 0.0,
    }

    params = {
        "min_candidate_score": min_candidate_score,
        "max_candidate_rank": max_candidate_rank,
        "stage": stage,
    }

    if dry_run:
        print(json.dumps({"stats": stats, "params": params}, ensure_ascii=False, indent=2))
        return

    save_jsonl(output_records, output_path)

    if save_checkpoint:
        if checkpoint_path is None:
            ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
            checkpoint_path = Path(f"data/output/checkpoints/{stage}_step_4a_{ts}.json")
        checkpoint = _build_checkpoint(
            stage=stage,
            input_path=input_path,
            output_path=output_path,
            answers_path=answers_path,
            chunk_dir=chunk_dir,
            stats=stats,
            params=params,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 checkpoint: {checkpoint_path}")

    print(json.dumps({"stats": stats, "params": params}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer-based auto labeling (Type1-only)")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="embedding retrieval template (nested jsonl)")
    parser.add_argument("--output", required=True, type=Path, help="auto label output (nested jsonl)")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--save-checkpoint", action="store_true", help="写入 checkpoint；未提供路径则生成带时间戳的默认路径")
    parser.add_argument("--min-candidate-score", type=float, default=0.4)
    parser.add_argument("--max-candidate-rank", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="only print stats, do not write output")
    args = parser.parse_args()

    run(
        stage=args.stage,
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint_path,
        min_candidate_score=args.min_candidate_score,
        max_candidate_rank=args.max_candidate_rank,
        dry_run=args.dry_run,
        save_checkpoint=args.save_checkpoint,
    )


if __name__ == "__main__":
    main()
