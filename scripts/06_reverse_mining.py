#!/usr/bin/env python3
"""Step 6: Reverse Mining - 从 Hybrid Top-50 召回中挖掘正例与 hard negatives.

流程：
1) 读取 hybrid_rrf_top50_stage{X}.jsonl（候选池）
2) 读取 finglm_master.jsonl 获取答案（query_id == master_id join）
3) 读取 chunks 获取 chunk content
4) 使用 answer_matcher 规则匹配候选 chunk
5) 匹配成功的作为正例，未匹配的高 rank 作为 hard negatives
6) 输出 mined_triplets + stats

产物：
- data/output/mining/mined_triplets_stage{X}.jsonl
- data/output/mining/mining_stats_stage{X}.json
- data/output/checkpoints/stage{X}_step_6_mining.json

验收（Stage1）：
- pos_found_rate ≥ 50%（Stretch ≥ 60%）
- total_triplets ≥ 180
- avg_best_pos_rank ≤ 10（每 query 最佳正例的平均 rank，--strict 模式下校验）

Triplet 格式说明：
- 每条 triplet 为 1 query + 1 positive + N negatives（ListWise 格式）
- 下游训练可直接用于 Contrastive Loss 或展开为 pair-wise
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from answer_matcher import extract_key_values, match_chunk_to_answer, is_type1
from chunk_manager import load_chunks


def _beijing_now() -> str:
    """返回北京时区时间字符串。"""
    tz_bj = timezone(timedelta(hours=8))
    return datetime.now(tz_bj).strftime("%Y-%m-%d %H:%M:%S %Z")


def load_master_index(master_path: Path) -> Dict[int, Dict[str, Any]]:
    """加载 finglm_master.jsonl，返回 master_id -> record 映射。"""
    index: Dict[int, Dict[str, Any]] = {}
    with master_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mid = rec.get("master_id")
            if mid is not None:
                index[int(mid)] = rec
    return index


def load_retrieval_results(retrieval_path: Path) -> List[Dict[str, Any]]:
    """加载检索结果 JSONL。"""
    results: List[Dict[str, Any]] = []
    with retrieval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def mine_triplets(
    retrieval_results: List[Dict[str, Any]],
    master_index: Dict[int, Dict[str, Any]],
    chunk_index,
    neg_ratio: int = 3,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """执行 Reverse Mining，返回 triplets 和统计信息。

    Args:
        retrieval_results: 检索结果列表（每个 query 包含 hits）
        master_index: master_id -> 答案记录映射
        chunk_index: ChunkIndex 实例
        neg_ratio: 每个正例对应的 hard negative 数量
        confidence_threshold: 正例置信度阈值

    Returns:
        {
            "triplets": [...],
            "stats": {...}
        }
    """
    triplets: List[Dict[str, Any]] = []

    # 统计
    total_queries = 0
    queries_with_pos = 0
    total_positives = 0
    total_negatives = 0
    pos_ranks: List[int] = []  # 所有正例的 rank
    best_pos_ranks: List[int] = []  # 每 query 最佳正例的 rank
    skipped_no_master = 0
    skipped_not_type1 = 0
    skipped_no_key_values = 0
    skipped_no_hits = 0
    skipped_no_chunk = 0
    skipped_chunk_ids: List[str] = []  # 记录找不到内容的 chunk_id（最多 10 个示例）

    for query_rec in retrieval_results:
        query_id = query_rec.get("query_id")
        query_text = query_rec.get("query", "")
        hits = query_rec.get("hits", [])
        pdf_stem = query_rec.get("pdf", "")

        total_queries += 1

        # 获取 master 记录
        master_rec = master_index.get(query_id)
        if not master_rec:
            skipped_no_master += 1
            continue

        # 检查是否 Type1
        if not is_type1(master_rec):
            skipped_not_type1 += 1
            continue

        # 提取关键值
        key_values = extract_key_values(master_rec)
        if not key_values:
            skipped_no_key_values += 1
            continue

        if not hits:
            skipped_no_hits += 1
            continue

        # 获取答案文本（用于记录）
        answers = master_rec.get("answers", [])
        answer_text = answers[0] if answers else ""
        prompt = master_rec.get("prompt", {})
        prom_answer = prompt.get("prom_answer", "")

        # 遍历 hits 进行匹配
        pos_chunks: List[Dict[str, Any]] = []
        neg_candidates: List[Dict[str, Any]] = []

        for hit in hits:
            chunk_id = hit.get("chunk_id")
            rank = hit.get("rank", 999)
            score = hit.get("score", 0.0)

            # 获取 chunk content
            chunk_data = chunk_index.get(chunk_id)
            if not chunk_data:
                # 尝试从 hits 中获取 text（如果已内嵌）
                chunk_text = hit.get("text", hit.get("content", ""))
            else:
                chunk_text = chunk_data.get("content", chunk_data.get("text", ""))

            if not chunk_text:
                skipped_no_chunk += 1
                if len(skipped_chunk_ids) < 10:
                    skipped_chunk_ids.append(chunk_id)
                continue

            # 规则匹配
            match_result = match_chunk_to_answer(chunk_text, key_values)

            if match_result.is_match and match_result.confidence >= confidence_threshold:
                pos_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text[:500],  # 截断避免过长
                    "rank": rank,
                    "score": score,
                    "confidence": match_result.confidence,
                    "match_type": match_result.match_type,
                    "match_evidence": match_result.match_evidence[:300],
                })
            else:
                neg_candidates.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text[:500],
                    "rank": rank,
                    "score": score,
                })

        if not pos_chunks:
            continue

        queries_with_pos += 1
        total_positives += len(pos_chunks)

        # 记录正例 rank
        query_pos_ranks = []
        for pos in pos_chunks:
            pos_ranks.append(pos["rank"])
            query_pos_ranks.append(pos["rank"])
        
        # 记录该 query 的最佳正例 rank
        best_pos_ranks.append(min(query_pos_ranks))

        # 选取 hard negatives：未匹配中 rank 最高的
        neg_selected = neg_candidates[:neg_ratio * len(pos_chunks)]
        total_negatives += len(neg_selected)

        # 构建 triplet 记录（每个正例一条）
        for pos in pos_chunks:
            triplet = {
                "query_id": query_id,
                "query": query_text,
                "pdf_stem": pdf_stem,
                "answer": prom_answer or answer_text[:200],
                "pos_chunk_id": pos["chunk_id"],
                "pos_text": pos["text"],
                "pos_rank": pos["rank"],
                "pos_score": pos["score"],
                "confidence": pos["confidence"],
                "match_type": pos["match_type"],
                "match_evidence": pos["match_evidence"],
                "neg_chunk_ids": [n["chunk_id"] for n in neg_selected],
                "neg_texts": [n["text"] for n in neg_selected],
            }
            triplets.append(triplet)

    # 统计汇总
    pos_found_rate = queries_with_pos / total_queries if total_queries > 0 else 0.0
    avg_pos_rank = sum(pos_ranks) / len(pos_ranks) if pos_ranks else 0.0
    avg_best_pos_rank = sum(best_pos_ranks) / len(best_pos_ranks) if best_pos_ranks else 0.0

    stats = {
        "total_queries": total_queries,
        "queries_with_pos": queries_with_pos,
        "pos_found_rate": round(pos_found_rate, 4),
        "total_triplets": len(triplets),
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "avg_pos_rank": round(avg_pos_rank, 2),
        "avg_best_pos_rank": round(avg_best_pos_rank, 2),  # 每 query 最佳正例的平均 rank
        "min_pos_rank": min(pos_ranks) if pos_ranks else None,
        "max_pos_rank": max(pos_ranks) if pos_ranks else None,
        "skipped_no_master": skipped_no_master,
        "skipped_not_type1": skipped_not_type1,
        "skipped_no_key_values": skipped_no_key_values,
        "skipped_no_hits": skipped_no_hits,
        "skipped_no_chunk": skipped_no_chunk,
        "skipped_chunk_ids_sample": skipped_chunk_ids,  # 最多 10 个示例
        "confidence_threshold": confidence_threshold,
        "neg_ratio": neg_ratio,
        "triplet_format": "query-pos-negatives",  # ListWise 格式说明
    }

    return {"triplets": triplets, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Step 6: Reverse Mining")
    parser.add_argument("--stage", type=str, default="stage1", help="Stage 名称")
    parser.add_argument(
        "--retrieval-input",
        type=str,
        default=None,
        help="检索结果文件路径（默认 data/output/retrieval/hybrid_rrf_top50_{stage}.jsonl）",
    )
    parser.add_argument(
        "--master-path",
        type=str,
        default="finglm_data_store/finglm_master.jsonl",
        help="finglm_master.jsonl 路径",
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="data/output/chunks",
        help="Chunk 文件目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/mining",
        help="输出目录",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/output/checkpoints",
        help="Checkpoint 目录",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=3,
        help="每个正例对应的 hard negative 数量",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="正例置信度阈值",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="保存 checkpoint",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：avg_best_pos_rank 未达标时返回非零退出码",
    )

    args = parser.parse_args()

    # 路径处理
    project_root = Path(__file__).resolve().parent.parent
    stage = args.stage

    if args.retrieval_input:
        retrieval_path = Path(args.retrieval_input)
    else:
        retrieval_path = project_root / f"data/output/retrieval/hybrid_rrf_top50_{stage}.jsonl"

    master_path = project_root / args.master_path
    chunk_dir = project_root / args.chunk_dir
    output_dir = project_root / args.output_dir
    checkpoint_dir = project_root / args.checkpoint_dir

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 6] Reverse Mining for {stage}")
    print(f"  Retrieval input: {retrieval_path}")
    print(f"  Master path: {master_path}")
    print(f"  Chunk dir: {chunk_dir}")
    print(f"  Neg ratio: {args.neg_ratio}")
    print(f"  Confidence threshold: {args.confidence_threshold}")

    # 加载数据
    print("\n[1/4] Loading master index...")
    master_index = load_master_index(master_path)
    print(f"  Loaded {len(master_index)} master records")

    print("\n[2/4] Loading retrieval results...")
    retrieval_results = load_retrieval_results(retrieval_path)
    print(f"  Loaded {len(retrieval_results)} query results")

    print("\n[3/4] Loading chunk index...")
    chunk_index = load_chunks(chunk_dir)
    print(f"  Loaded chunks from {chunk_dir}")

    print("\n[4/4] Mining triplets...")
    result = mine_triplets(
        retrieval_results=retrieval_results,
        master_index=master_index,
        chunk_index=chunk_index,
        neg_ratio=args.neg_ratio,
        confidence_threshold=args.confidence_threshold,
    )

    triplets = result["triplets"]
    stats = result["stats"]

    # 输出统计
    print("\n=== Mining Statistics ===")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Queries with positive: {stats['queries_with_pos']}")
    print(f"  Pos found rate: {stats['pos_found_rate']*100:.2f}%")
    print(f"  Total triplets: {stats['total_triplets']}")
    print(f"  Avg pos rank (all): {stats['avg_pos_rank']}")
    print(f"  Avg best pos rank (per query): {stats['avg_best_pos_rank']}")
    print(f"  Min/Max pos rank: {stats['min_pos_rank']} / {stats['max_pos_rank']}")
    print(f"  Skipped (no master): {stats['skipped_no_master']}")
    print(f"  Skipped (not Type1): {stats['skipped_not_type1']}")
    print(f"  Skipped (no key values): {stats['skipped_no_key_values']}")
    print(f"  Skipped (no hits): {stats['skipped_no_hits']}")
    print(f"  Skipped (no chunk content): {stats['skipped_no_chunk']}")
    if stats['skipped_chunk_ids_sample']:
        print(f"    Sample missing chunk_ids: {stats['skipped_chunk_ids_sample'][:5]}")

    # 验收检查
    print("\n=== Acceptance Criteria ===")
    pos_rate_ok = stats["pos_found_rate"] >= 0.50
    triplets_ok = stats["total_triplets"] >= 180
    avg_best_rank_ok = stats["avg_best_pos_rank"] <= 10 if stats["avg_best_pos_rank"] else False

    print(f"  pos_found_rate >= 50%: {'✓' if pos_rate_ok else '✗'} ({stats['pos_found_rate']*100:.2f}%)")
    print(f"  total_triplets >= 180: {'✓' if triplets_ok else '✗'} ({stats['total_triplets']})")
    print(f"  avg_best_pos_rank <= 10: {'✓' if avg_best_rank_ok else '⚠'} ({stats['avg_best_pos_rank']})")
    if not avg_best_rank_ok:
        print(f"    ⚠ Note: avg_best_pos_rank reflects retriever quality (Step 5), not mining quality.")
        print(f"    ⚠ This metric should improve after Reranker training (Step 9).")

    # 保存 triplets
    triplets_path = output_dir / f"mined_triplets_{stage}.jsonl"
    with triplets_path.open("w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n[Output] Triplets saved to: {triplets_path}")

    # 保存 stats
    stats_path = output_dir / f"mining_stats_{stage}.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[Output] Stats saved to: {stats_path}")

    # 保存 checkpoint
    if args.save_checkpoint:
        checkpoint = {
            "stage": stage,
            "step": 6,
            "completed_at": _beijing_now(),
            "input_files": {
                "retrieval": str(retrieval_path),
                "master": str(master_path),
                "chunk_dir": str(chunk_dir),
            },
            "output_files": {
                "triplets": str(triplets_path),
                "stats": str(stats_path),
            },
            "params": {
                "neg_ratio": args.neg_ratio,
                "confidence_threshold": args.confidence_threshold,
            },
            "stats": stats,
        }
        checkpoint_path = checkpoint_dir / f"{stage}_step_6_mining.json"
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        print(f"[Output] Checkpoint saved to: {checkpoint_path}")

    # 返回状态码
    # 核心验收：pos_found_rate 和 total_triplets（mining 产出质量）
    # avg_best_pos_rank 反映上游检索器能力，--strict 模式下才作为失败条件
    core_ok = pos_rate_ok and triplets_ok
    strict_ok = core_ok and avg_best_rank_ok

    if args.strict:
        if strict_ok:
            print("\n✓ Mining completed successfully (strict mode)!")
            return 0
        else:
            print("\n✗ Mining failed strict validation.")
            return 1
    else:
        if core_ok:
            print("\n✓ Mining completed successfully!")
            if not avg_best_rank_ok:
                print("  (avg_best_pos_rank warning ignored; use --strict to enforce)")
            return 0
        else:
            print("\n✗ Mining failed core validation (pos_found_rate or total_triplets).")
            return 1


if __name__ == "__main__":
    sys.exit(main())
