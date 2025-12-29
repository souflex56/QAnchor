#!/usr/bin/env python3
"""Stage1 三路检索：Embedding + BM25 + Hybrid (RRF)。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# 确保仓库根目录可导入
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_answers, load_qa_mapping, select_pdf_subset  # noqa: E402
from src.chunk_manager import ChunkIndex, load_chunks  # noqa: E402
from src.embedding_retriever import EmbeddingRetriever  # noqa: E402
from src.bm25_retriever import BM25Retriever  # noqa: E402
from src.hybrid_fusion import rrf_fuse  # noqa: E402


def _pdf_stem(path_str: str) -> str:
    return Path(path_str).stem


def _load_problematic(path: Path | None) -> List[str]:
    if not path or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pdf_list: List[str] = []
        for item in data.get("problematic", []):
            if isinstance(item, dict) and "pdf_stem" in item:
                pdf_list.append(str(item["pdf_stem"]))
            else:
                pdf_list.append(str(item))
        return pdf_list
    except Exception:
        return []


def _filter_qa_by_pdfs(qa_df: pd.DataFrame, pdf_stems: Sequence[str]) -> pd.DataFrame:
    stems_set = set(pdf_stems)

    def _stem_from_report(val: Any) -> str:
        try:
            return Path(str(val)).stem
        except Exception:
            return str(val)

    qa_df = qa_df.copy()
    qa_df["report_stem"] = qa_df["report_paths"].apply(_stem_from_report)
    return qa_df[qa_df["report_stem"].isin(stems_set)]


def _prepare_queries(
    qa_df: pd.DataFrame,
    answers_index: Dict[int, List[str]],
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    subset_df = qa_df if limit is None else qa_df.head(limit)
    for _, row in subset_df.iterrows():
        try:
            master_id = int(row["master_id"])
        except (TypeError, ValueError):
            continue
        query_text = str(row["question"]).strip()
        if len(query_text) < 3:
            continue
        answers = [
            {"answer_id": f"{master_id}_{i}", "text": ans}
            for i, ans in enumerate(answers_index.get(master_id, []))
        ]
        records.append(
            {
                "query_id": master_id,
                "query": query_text,
                "pdf": row["report_stem"],
                "company": row.get("company"),
                "year": row.get("year"),
                "answers": answers,
            }
        )
    return records


def _build_model_tag(config: Dict[str, Any]) -> str:
    model_name = str(config.get("embedding_model"))
    normalize = config.get("normalize_embeddings", True)
    max_seq_length = config.get("max_seq_length")
    precision_mode = (config.get("precision_mode") or "auto").lower()
    section_blacklist_enabled = config.get("section_blacklist_enabled", False)

    safe_name = (
        model_name.lower()
        .replace("/", "-")
        .replace(" ", "-")
        .replace("@", "-")
    )
    norm_tag = "normT" if normalize else "normF"
    msl_tag = f"msl{max_seq_length}" if max_seq_length else "mslNone"
    pm_tag = f"pm{precision_mode}"
    blk_tag = "blkT" if section_blacklist_enabled else "blkF"
    return f"{safe_name}_{norm_tag}_{msl_tag}_{pm_tag}_{blk_tag}"


def _compute_chunk_hash(chunk_ids: Sequence[str], texts: Sequence[str]) -> str:
    import hashlib

    pairs = list(zip(chunk_ids, texts))
    pairs.sort(key=lambda x: str(x[0]))
    h = hashlib.sha1()
    for cid, text in pairs:
        h.update(str(cid or "").encode("utf-8"))
        h.update(b"::")
        h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _load_cache(cache_dir: Path, pdf_stem: str) -> Tuple[np.ndarray | None, Dict[str, Any] | None]:
    npz_path = cache_dir / f"{pdf_stem}.npz"
    meta_path = cache_dir / f"{pdf_stem}.json"
    if not npz_path.exists():
        return None, None
    try:
        emb = np.load(npz_path)["embeddings"]
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return emb, meta
    except Exception:
        return None, None


def _save_cache(cache_dir: Path, pdf_stem: str, embeddings: np.ndarray, meta: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{pdf_stem}.npz", embeddings=embeddings)
    (cache_dir / f"{pdf_stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_chunk_payload(chunks: Sequence[Dict], pdf_stem: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []
    for child in chunks:
        cid = child.get("chunk_id")
        text = child.get("content") or child.get("text") or ""
        meta = child.get("metadata") or {}
        metas.append(
            {
                "chunk_id": cid,
                "parent_id": child.get("parent_id"),
                "pdf": meta.get("pdf") or child.get("pdf") or f"{pdf_stem}.pdf",
                "pdf_stem": meta.get("pdf_stem") or child.get("pdf_stem") or pdf_stem,
                "page_numbers": meta.get("page_numbers") or child.get("page_numbers") or [],
                "section_path": meta.get("section_path") or child.get("section_path") or [],
            }
        )
        ids.append(cid)
        texts.append(text)
    return ids, texts, metas


def _format_hits(raw_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for hit in raw_hits:
        results.append(
            {
                "chunk_id": hit.get("chunk_id"),
                "parent_id": hit.get("parent_id"),
                "pdf": hit.get("pdf"),
                "pdf_stem": hit.get("pdf_stem"),
                "page_numbers": hit.get("page_numbers") or [],
                "section_path": hit.get("section_path") or [],
                "score": float(hit.get("score", 0.0)),
                "rank": int(hit.get("rank", 0)),
            }
        )
    return results


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_three_way_retrieval(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    retrieval_cfg = cfg["retrieval"]
    stage_cfg = cfg.get("stages", {})

    qa_df = load_qa_mapping(data_cfg["qa_mapping"])
    raw_query_count = len(qa_df)
    answers = load_answers(data_cfg["answers"])

    subset_info = select_pdf_subset(args.stage, data_cfg["summary"], data_cfg["pdf_dir"], stage_cfg)
    pdf_records = subset_info["records"]
    pdf_stems = [_pdf_stem(rec["pdf_path"]) for rec in pdf_records]

    excluded = set(_load_problematic(Path(args.exclude_pdfs) if args.exclude_pdfs else None))
    pdf_stems = [p for p in pdf_stems if p not in excluded]

    qa_df = _filter_qa_by_pdfs(qa_df, pdf_stems)
    filtered_query_count = len(qa_df)
    print(f"过滤后 query 数={filtered_query_count}, 原始={raw_query_count}")
    answers_index: Dict[int, List[str]] = {}
    for rec in answers:
        mid = rec.get("master_id")
        if mid is None:
            continue
        answers_index[int(mid)] = list(rec.get("answers") or [])

    queries = _prepare_queries(qa_df, answers_index, limit=stage_cfg.get(args.stage, {}).get("qa_count"))
    if not queries:
        raise RuntimeError("未找到可检索的 query")

    chunk_index: ChunkIndex = load_chunks(data_cfg["chunk_output"])
    bm25 = BM25Retriever()
    pdf_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
    for stem in pdf_stems:
        chunks = chunk_index.get_chunks_by_pdf(stem)
        pdf_to_chunks[stem] = chunks
        bm25.add_pdf(stem, chunks)

    model_tag = _build_model_tag(retrieval_cfg)
    cache_dir = Path(data_cfg["output_root"] if "output_root" in data_cfg else data_cfg["retrieval_output"]).parent / "embeddings" / model_tag
    retriever = EmbeddingRetriever(
        model_name=retrieval_cfg["embedding_model"],
        device=retrieval_cfg.get("device"),
        batch_size=retrieval_cfg.get("batch_size", 16),
        normalize=retrieval_cfg.get("normalize_embeddings", True),
        max_seq_length=retrieval_cfg.get("max_seq_length"),
        empty_cache_interval=retrieval_cfg.get("empty_cache_interval", 5),
        precision_mode=retrieval_cfg.get("precision_mode", "auto"),
        enable_empty_cache=retrieval_cfg.get("enable_empty_cache", True),
        use_inference_mode=retrieval_cfg.get("use_inference_mode", True),
        mem_log_interval=retrieval_cfg.get("mem_log_interval"),
    )

    top_k_train = args.top_k_train
    top_k_eval = args.top_k_eval
    emb_records_50: List[Dict[str, Any]] = []
    emb_records_20: List[Dict[str, Any]] = []
    bm25_records_50: List[Dict[str, Any]] = []
    bm25_records_20: List[Dict[str, Any]] = []
    hybrid_records_50: List[Dict[str, Any]] = []
    hybrid_records_20: List[Dict[str, Any]] = []

    queries_by_pdf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in queries:
        queries_by_pdf[q["pdf"]].append(q)

    total_cache_hits = 0
    total_cache_miss = 0

    for pdf_stem, qlist in tqdm(queries_by_pdf.items(), desc="per-pdf retrieval"):
        chunks = pdf_to_chunks.get(pdf_stem, [])
        chunk_ids, chunk_texts, chunk_metas = _build_chunk_payload(chunks, pdf_stem)
        cached_emb, cached_meta = _load_cache(cache_dir, pdf_stem)
        use_cache = cached_emb is not None and cached_meta and cached_meta.get("chunk_count") == len(chunk_texts)
        if use_cache and cached_meta.get("chunk_hash") != _compute_chunk_hash(chunk_ids, chunk_texts):
            use_cache = False
        if use_cache:
            chunk_embeddings = cached_emb
            total_cache_hits += 1
        else:
            chunk_embeddings = retriever.encode_chunks(chunk_texts)
            total_cache_miss += 1
            meta = {
                "model_name": retrieval_cfg["embedding_model"],
                "normalize_embeddings": retrieval_cfg.get("normalize_embeddings", True),
                "max_seq_length": retrieval_cfg.get("max_seq_length"),
                "precision_mode": retrieval_cfg.get("precision_mode"),
                "section_blacklist_enabled": retrieval_cfg.get("section_blacklist_enabled", False),
                "embedding_dim": int(chunk_embeddings.shape[1]) if chunk_embeddings.size > 0 else 0,
                "chunk_count": len(chunk_texts),
                "chunk_ids": list(chunk_ids),
                "chunk_hash": _compute_chunk_hash(chunk_ids, chunk_texts),
                "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            }
            _save_cache(cache_dir, pdf_stem, chunk_embeddings, meta)

        query_texts = [q["query"] for q in qlist]
        query_embeddings = retriever.encode_queries(query_texts)

        emb_results = retriever.retrieve_top_k(query_embeddings, chunk_embeddings, chunk_metas, top_k_train)
        for q, hits in zip(qlist, emb_results):
            rec_base = {
                "query_id": q["query_id"],
                "query": q["query"],
                "pdf": q["pdf"],
                "company": q.get("company"),
                "year": q.get("year"),
                "answers": q.get("answers", []),
            }
            emb_hits_50 = _format_hits(hits)
            emb_hits_20 = emb_hits_50[:top_k_eval]
            emb_records_50.append({**rec_base, "hits": emb_hits_50})
            emb_records_20.append({**rec_base, "hits": emb_hits_20})

            bm_hits_50 = _format_hits(bm25.retrieve(q["query"], q["pdf"], top_k_train))
            bm_hits_20 = bm_hits_50[:top_k_eval]
            bm25_records_50.append({**rec_base, "hits": bm_hits_50})
            bm25_records_20.append({**rec_base, "hits": bm_hits_20})

            hybrid_50 = _format_hits(
                rrf_fuse(emb_hits_50, bm_hits_50, rrf_k=retrieval_cfg.get("hybrid", {}).get("rrf_k", 60), missing_rank=retrieval_cfg.get("hybrid", {}).get("missing_rank", 9999), top_k=top_k_train)
            )
            hybrid_20 = _format_hits(
                rrf_fuse(emb_hits_20, bm_hits_20, rrf_k=retrieval_cfg.get("hybrid", {}).get("rrf_k", 60), missing_rank=retrieval_cfg.get("hybrid", {}).get("missing_rank", 9999), top_k=top_k_eval)
            )
            hybrid_records_50.append({**rec_base, "hits": hybrid_50})
            hybrid_records_20.append({**rec_base, "hits": hybrid_20})

    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / f"embedding_top{top_k_train}_{args.stage}.jsonl", emb_records_50)
    _write_jsonl(output_dir / f"bm25_top{top_k_train}_{args.stage}.jsonl", bm25_records_50)
    _write_jsonl(output_dir / f"hybrid_rrf_top{top_k_train}_{args.stage}.jsonl", hybrid_records_50)
    _write_jsonl(output_dir / f"embedding_top{top_k_eval}_{args.stage}.jsonl", emb_records_20)
    _write_jsonl(output_dir / f"bm25_top{top_k_eval}_{args.stage}.jsonl", bm25_records_20)
    _write_jsonl(output_dir / f"hybrid_rrf_top{top_k_eval}_{args.stage}.jsonl", hybrid_records_20)

    stats = {
        "stage": args.stage,
        "step": "5b",
        "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "params": {
            "top_k_train": top_k_train,
            "top_k_eval": top_k_eval,
            "rrf_k": retrieval_cfg.get("hybrid", {}).get("rrf_k", 60),
            "missing_rank": retrieval_cfg.get("hybrid", {}).get("missing_rank", 9999),
        },
        "input_files": {
            "qa_mapping": data_cfg["qa_mapping"],
            "answers": data_cfg["answers"],
            "summary": data_cfg["summary"],
            "chunk_dir": data_cfg["chunk_output"],
            "exclude_pdfs": args.exclude_pdfs,
        },
        "stats": {
            "query_count": len(queries),
            "pdf_count": len(pdf_stems),
            "cache_hits": total_cache_hits,
            "cache_miss": total_cache_miss,
            "raw_query_count": raw_query_count,
            "filtered_query_count": filtered_query_count,
            "bm25_pdfs_indexed": len(pdf_to_chunks),
        },
        "output_files": {
            "embedding_top50": str(output_dir / f"embedding_top{top_k_train}_{args.stage}.jsonl"),
            "bm25_top50": str(output_dir / f"bm25_top{top_k_train}_{args.stage}.jsonl"),
            "hybrid_top50": str(output_dir / f"hybrid_rrf_top{top_k_train}_{args.stage}.jsonl"),
            "embedding_top20": str(output_dir / f"embedding_top{top_k_eval}_{args.stage}.jsonl"),
            "bm25_top20": str(output_dir / f"bm25_top{top_k_eval}_{args.stage}.jsonl"),
            "hybrid_top20": str(output_dir / f"hybrid_rrf_top{top_k_eval}_{args.stage}.jsonl"),
        },
    }
    if args.checkpoint_path:
        Path(args.checkpoint_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        ckpt_dir = Path(REPO_ROOT) / "data/output/checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = ckpt_dir / f"{args.stage}_step_5b_three_way_retrieval_{ts}.json"
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成三路检索，queries={len(queries)}, pdfs={len(pdf_stems)}, cache_hit/miss={total_cache_hits}/{total_cache_miss}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三路检索：Embedding + BM25 + Hybrid (RRF)")
    parser.add_argument("--stage", type=str, default="stage1", help="Stage name, e.g., stage1")
    parser.add_argument("--config", type=str, default="config/weak_supervision_config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="data/output/retrieval", help="输出目录")
    parser.add_argument("--top-k-train", type=int, default=50, help="训练口径 TopK（默认50）")
    parser.add_argument("--top-k-eval", type=int, default=20, help="评测口径 TopK（默认20）")
    parser.add_argument("--exclude-pdfs", type=str, default="data/output/quality/problematic_pdfs_stage1.json", help="问题 PDF 列表 JSON（可空）")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="可选：自定义 checkpoint 路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_three_way_retrieval(args)
