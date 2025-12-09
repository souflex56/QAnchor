#!/usr/bin/env python3
"""Stage0 embedding-only 检索脚本（Step 3a）。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# 确保仓库根目录可导入
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_answers, load_qa_mapping, select_pdf_subset  # noqa: E402
from src.embedding_retriever import EmbeddingRetriever  # noqa: E402


def _pdf_stem(path_str: str) -> str:
    path = Path(path_str)
    stem = path.stem
    return stem


def _chunk_path_for_pdf(pdf_path: str, chunk_dir: Path) -> Path:
    stem = _pdf_stem(pdf_path)
    return chunk_dir / f"{stem}_chunks.json"


def _load_chunk_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_child_chunks(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    data = _load_chunk_file(path)
    children = data.get("children") or data.get("chunks")
    if children is None:
        print(f"警告：{path.name} 缺少 children/chunks 字段，已跳过")
        return [], []
    pdf_from_source = data.get("source") or ""
    pdf_stem = path.stem[:-7] if path.stem.endswith("_chunks") else path.stem

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for child in children:
        text = child.get("content") or child.get("text") or ""
        meta = child.get("metadata") or {}
        pages = meta.get("page_numbers") or []
        if not pages and child.get("page_number"):
            pages = [child["page_number"]]
        metas.append(
            {
                "chunk_id": child.get("chunk_id"),
                "parent_id": child.get("parent_id"),
                "pdf": pdf_from_source or pdf_stem,
                "pdf_stem": pdf_stem,
                "page_numbers": pages,
                "section_path": meta.get("section_path") or [],
            }
        )
        texts.append(text)
    return texts, metas


def _build_answers_index(answer_records: Iterable[Dict[str, Any]]) -> Dict[int, List[str]]:
    index: Dict[int, List[str]] = {}
    for rec in answer_records:
        mid = rec.get("master_id")
        if mid is None:
            continue
        answers = rec.get("answers") or []
        if answers:
            index[int(mid)] = list(answers)
    return index


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


def _section_is_blacklisted(section_path: Sequence[str], blacklist: Sequence[str]) -> bool:
    """简单子串过滤，section_path 中若含任一黑名单子串则过滤。"""
    if not section_path or not blacklist:
        return False
    # section_path 可能是字符串或列表，这里统一为字符串再判断
    try:
        path_str = "/".join(map(str, section_path))
    except Exception:
        path_str = str(section_path)
    return any(key in path_str for key in blacklist)


def _print_distribution(scores: List[float]) -> None:
    if not scores:
        print("无相似度数据可统计")
        return
    arr = np.array(scores)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(arr, percentiles)
    pct_str = ", ".join(f"p{p}={v:.4f}" for p, v in zip(percentiles, pct_values))
    print(f"Similarity percentiles: {pct_str}, max={arr.max():.4f}, min={arr.min():.4f}")

    bins = np.arange(0.0, 1.01, 0.1)
    hist, edges = np.histogram(arr, bins=bins)
    hist_parts = []
    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        hist_parts.append(f"[{left:.1f},{right:.1f}): {int(count)}")
    print("Histogram (0.1 bins): " + "; ".join(hist_parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage0 embedding-only 检索")
    parser.add_argument("--stage", default="stage0", help="阶段名称，默认 stage0")
    parser.add_argument(
        "--config",
        default="config/weak_supervision_config.yaml",
        help="主配置文件路径",
    )
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="覆盖配置的 top_k")
    parser.add_argument("--limit-queries", type=int, default=None, help="限制处理的 query 数量（调试用）")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSONL 路径；默认 data/output/retrieval/embedding_{stage}.jsonl",
    )
    parser.add_argument(
        "--embedding-model",
        dest="embedding_model",
        default=None,
        help="覆盖配置的 embedding 模型名",
    )
    parser.add_argument("--device", default=None, help="覆盖配置的设备，如 cpu 或 cuda:0")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    retrieval_cfg = cfg.get("retrieval", {})

    chunk_dir = Path(data_cfg["chunk_output"])
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output or f"data/output/retrieval/embedding_{args.stage}.jsonl").parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output or f"data/output/retrieval/embedding_{args.stage}.jsonl")

    subset = select_pdf_subset(
        stage=args.stage,
        summary_path=data_cfg["summary"],
        pdf_dir=data_cfg["pdf_dir"],
        stage_config=cfg.get("stages"),
    )
    pdf_stems = [Path(rec["pdf_path"]).stem for rec in subset["records"]]
    print(f"[Stage: {args.stage}] 选中 PDF 数: {len(pdf_stems)}")

    # 加载 QA 与答案
    qa_df = load_qa_mapping(data_cfg["qa_mapping"])
    qa_df = _filter_qa_by_pdfs(qa_df, pdf_stems)
    answers_index = _build_answers_index(load_answers(data_cfg["answers"]))
    queries = _prepare_queries(qa_df, answers_index, limit=args.limit_queries)
    if not queries:
        print("未找到匹配选中 PDF 的 query，退出。")
        return
    print(f"处理 query 数: {len(queries)}")
    # 仅保留本次 queries 涉及的 pdf_stem，避免无谓编码其他 PDF 的 chunks
    used_pdfs = {str(q["pdf"]) for q in queries}
    print(f"本次查询涉及 PDF 数: {len(used_pdfs)}")

    # 加载 chunk 候选（只加载 used_pdfs）
    chunk_texts: List[str] = []
    chunk_metas: List[Dict[str, Any]] = []
    missing_chunks: List[str] = []
    section_blacklist_enabled = retrieval_cfg.get("section_blacklist_enabled", False)
    section_blacklist = retrieval_cfg.get("section_blacklist") or []
    dropped_by_blacklist = 0
    for stem in pdf_stems:
        if stem not in used_pdfs:
            continue
        cpath = chunk_dir / f"{stem}_chunks.json"
        if not cpath.exists():
            missing_chunks.append(stem)
            continue
        texts, metas = _extract_child_chunks(cpath)
        if section_blacklist_enabled and section_blacklist:
            for text, meta in zip(texts, metas):
                if _section_is_blacklisted(meta.get("section_path") or [], section_blacklist):
                    dropped_by_blacklist += 1
                    continue
                chunk_texts.append(text)
                chunk_metas.append(meta)
        else:
            chunk_texts.extend(texts)
            chunk_metas.extend(metas)
    if missing_chunks:
        print(f"警告：缺少 {len(missing_chunks)} 个 chunk 文件，已跳过: {', '.join(missing_chunks[:5])}{' ...' if len(missing_chunks) > 5 else ''}")
    if dropped_by_blacklist:
        print(f"根据 section_blacklist 过滤掉 {dropped_by_blacklist} 个 chunks")
    print(f"候选 child chunks: {len(chunk_metas)}")
    # 预先建立 pdf_stem 到 chunk 索引的映射，方便同文档内检索
    chunks_by_pdf: Dict[str, List[int]] = defaultdict(list)
    for idx, meta in enumerate(chunk_metas):
        pdf_key = meta.get("pdf_stem") or meta.get("pdf")
        if pdf_key:
            chunks_by_pdf[str(pdf_key)].append(idx)
    if not chunk_texts:
        print("未找到可用的 chunk 数据，退出。")
        return

    # 编码
    embedding_model = args.embedding_model or retrieval_cfg.get("embedding_model") or "qwen3-embedding-0.6b"
    device = args.device or retrieval_cfg.get("device")
    batch_size = retrieval_cfg.get("batch_size", 32)
    normalize = retrieval_cfg.get("normalize_embeddings", True)
    top_k = args.top_k or retrieval_cfg.get("top_k", 30)
    restrict_to_query_pdf = retrieval_cfg.get("restrict_to_query_pdf", True)

    retriever = EmbeddingRetriever(
        model_name=embedding_model,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
    )

    print(f"编码 {len(chunk_texts)} 个 chunks ...")
    chunk_embs = retriever.encode_chunks(chunk_texts)
    print(f"编码 {len(queries)} 个 queries ...")
    query_texts = [q["query"] for q in queries]
    query_embs = retriever.encode_queries(query_texts)

    print(f"检索 Top-{top_k} ...")
    if restrict_to_query_pdf:
        all_hits: List[List[Dict[str, Any]]] = []
        missing_candidates = 0
        for query_vec, query_rec in zip(query_embs, queries):
            pdf_key = str(query_rec.get("pdf"))
            idxs = chunks_by_pdf.get(pdf_key, [])
            if not idxs:
                missing_candidates += 1
                all_hits.append([])
                continue
            hits = retriever.retrieve_top_k(
                query_vec.reshape(1, -1),
                chunk_embs[idxs],
                [chunk_metas[i] for i in idxs],
                top_k=top_k,
            )[0]
            all_hits.append(hits)
        if missing_candidates:
            print(f"警告：{missing_candidates} 个 query 未找到同 PDF 的 chunk 候选，返回空结果")
    else:
        all_hits = retriever.retrieve_top_k(query_embs, chunk_embs, chunk_metas, top_k=top_k)

    # 输出 JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for query_rec, hits in zip(queries, all_hits):
            out = dict(query_rec)
            out["hits"] = hits
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"已写入检索结果：{output_path}")

    # 分布统计
    score_pool = [hit["score"] for hits in all_hits for hit in hits]
    print(f"总计候选: {len(score_pool)}")
    _print_distribution(score_pool)


if __name__ == "__main__":
    main()
