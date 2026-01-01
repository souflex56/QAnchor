"""生成 Gold Eval 标注模板及评测 blacklist。

使用示例（Stage1，覆盖模式）：
python scripts/07_prepare_gold_eval.py \
    --stage stage1 \
    --input data/output/retrieval/hybrid_rrf_top20_stage1.jsonl \
    --chunks-dir data/output/chunks \
    --size 50 \
    --seed 42 \
    --output data/output/annotations/gold_eval_50_template.jsonl \
    --blacklist config/eval_blacklist.json \
    --checkpoint data/output/checkpoints/stage1_step_7_gold_eval.json \
    --blacklist-mode replace

参数提示：
- blacklist-mode：
  - replace（默认）：blacklist 始终与本次采样一致。例：首次 size=50 → 50 条，改跑 size=100 → 覆盖为 100 条。
  - merge：与已有 blacklist 取并集，适合多批次累加采样。
- strict-missing：缺失 chunk 时是否报错退出。Stage2 规模大建议开启，Stage1 可选关闭。
- “缺失 chunk” 指候选引用的 `chunk_id` 在 `data/output/chunks/*.json` 中找不到对应文本（如分块未生成/被删或 ID 变更）；开启 strict-missing 可在发现空文本前直接中止，避免产出不可标注的候选。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunk_manager import ChunkIndex, load_chunks  # noqa: E402


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


def build_candidate(hit: Dict[str, Any], index: ChunkIndex, missing_chunks: List[str]) -> Dict[str, Any]:
    chunk = index.get(hit.get("chunk_id"))
    if chunk is None:
        missing_chunks.append(hit.get("chunk_id"))
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


def prepare_gold_eval(
    stage: str,
    input_path: Path,
    chunks_dir: Path,
    sample_size: int,
    seed: int,
    output_path: Path,
    blacklist_path: Path,
    checkpoint_path: Path,
    blacklist_mode: str = "replace",  # replace | merge
    strict_missing: bool = False,
) -> None:
    retrieval_records = load_jsonl(input_path)
    total_queries = len(retrieval_records)

    random.seed(seed)
    if sample_size >= total_queries:
        selected = retrieval_records
    else:
        selected = random.sample(retrieval_records, sample_size)

    chunk_index = load_chunks(chunks_dir)
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
                "stage": stage,
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
        if strict_missing:
            raise RuntimeError(warn_msg)

    # Update blacklist
    existing: Dict[str, Any] = {}
    if blacklist_path.exists():
        existing = json.loads(blacklist_path.read_text(encoding="utf-8"))
    new_ids = {rec.get("query_id") for rec in output_records if rec.get("query_id") is not None}
    if blacklist_mode == "merge":
        query_ids = set(existing.get("query_ids", [])) | new_ids
    else:
        query_ids = new_ids
    new_blacklist = {
        "query_ids": sorted(query_ids),
        "count": len(query_ids),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"gold_eval_{sample_size}",
    }
    blacklist_path.parent.mkdir(parents=True, exist_ok=True)
    blacklist_path.write_text(json.dumps(new_blacklist, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write checkpoint
    checkpoint = {
        "stage": stage,
        "step": 7,
        "completed_at": datetime.now().isoformat(),
        "input_file": str(input_path),
        "chunks_dir": str(chunks_dir),
        "output_file": str(output_path),
        "blacklist_file": str(blacklist_path),
        "sample_size_requested": sample_size,
        "selected_queries": len(output_records),
        "total_queries": total_queries,
        "missing_chunks": sorted(set(missing_chunks)),
        "blacklist_mode": blacklist_mode,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Gold Eval templates and blacklist.")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Hybrid Top20 retrieval JSONL. Default: data/output/retrieval/hybrid_rrf_top20_{stage}.jsonl",
    )
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/output/chunks"))
    parser.add_argument("--size", type=int, default=50, help="Number of queries to sample for Gold Eval.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Gold Eval template output JSONL. Default: data/output/annotations/gold_eval_{size}_template.jsonl",
    )
    parser.add_argument(
        "--blacklist",
        type=Path,
        default=Path("config/eval_blacklist.json"),
        help="Eval blacklist JSON path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint JSON path. Default: data/output/checkpoints/{stage}_step_7_gold_eval.json",
    )
    parser.add_argument(
        "--blacklist-mode",
        choices=["replace", "merge"],
        default="replace",
        help=(
            "replace(默认): blacklist 与本次采样保持一致（如 size=100 则覆盖旧 50，不保留历史）；"
            "merge: 与现有 blacklist 并集，适合多批次采样。"
        ),
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="缺失 chunk 时抛错退出（默认仅告警；Stage2 推荐开启）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or Path(f"data/output/retrieval/hybrid_rrf_top20_{args.stage}.jsonl")
    output_path = args.output or Path(f"data/output/annotations/gold_eval_{args.size}_template.jsonl")
    checkpoint_path = args.checkpoint or Path(f"data/output/checkpoints/{args.stage}_step_7_gold_eval.json")

    prepare_gold_eval(
        stage=args.stage,
        input_path=input_path,
        chunks_dir=args.chunks_dir,
        sample_size=args.size,
        seed=args.seed,
        output_path=output_path,
        blacklist_path=args.blacklist,
        checkpoint_path=checkpoint_path,
        blacklist_mode=args.blacklist_mode,
        strict_missing=args.strict_missing,
    )


if __name__ == "__main__":
    main()
