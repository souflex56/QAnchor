#!/usr/bin/env python3
"""Stage0 embedding-only 检索脚本（Step 3a）。"""

from __future__ import annotations

import argparse
import json
import sys
import hashlib
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from math import ceil
from tqdm import tqdm

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


def _compute_chunk_hash(chunk_ids: Sequence[str], texts: Sequence[str]) -> str:
    """对 chunk 内容做稳定 hash（按 chunk_id 排序后流式计算）。"""
    # 确保长度一致，缺失 chunk_id 用序号兜底
    norm_ids = [cid if cid is not None else f"idx_{i}" for i, cid in enumerate(chunk_ids)]
    pairs = list(zip(norm_ids, texts))
    pairs.sort(key=lambda x: str(x[0]))
    h = hashlib.sha1()
    for cid, text in pairs:
        h.update(str(cid).encode("utf-8"))
        h.update(b"::")
        h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _build_model_tag(
    model_name: str,
    normalize: bool,
    max_seq_length: int | None,
    precision_mode: str,
    section_blacklist_enabled: bool,
) -> str:
    """将关键超参编码为目录安全的 tag。"""
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


def _collect_mem_metrics() -> Dict[str, float]:
    """采集内存/显存信息，失败时返回已有字段。"""
    metrics: Dict[str, float] = {}
    try:
        import psutil
        import torch
    except Exception:
        return metrics
    try:
        vm = psutil.virtual_memory()
        metrics["ram_used_gb"] = vm.used / 1e9
        metrics["ram_avail_gb"] = vm.available / 1e9
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0)
        if torch.cuda.is_available():
            metrics["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            metrics["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        if torch.backends.mps.is_available():
            metrics["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1024**3
            metrics["mps_driver_gb"] = torch.mps.driver_allocated_memory() / 1024**3
    except Exception:
        return metrics
    return metrics


def _get_model_dtype(model: Any) -> str | None:
    """尝试从模型参数中获取 dtype。"""
    try:
        import torch
        params = None
        if hasattr(model, "parameters"):
            params = model.parameters()
        if params:
            first = next(iter(params))
            if isinstance(first, torch.Tensor):
                return str(first.dtype)
    except Exception:
        return None
    return None


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


def _calc_percentiles(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {}
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(np.array(scores), percentiles)
    return {f"p{p}": float(v) for p, v in zip(percentiles, pct_values)}


def _save_checkpoint(
    path: Path,
    *,
    stage: str,
    chunk_files: Sequence[Path],
    qa_mapping: Path,
    answers: Path,
    output_file: Path,
    params: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    checkpoint = {
        "stage": stage,
        "step": 3,
        "step_name": "embedding_retrieval",
        "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "input_files": [str(qa_mapping), str(answers), *[str(p) for p in chunk_files]],
        "output_files": [str(output_file)],
        "params": params,
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 checkpoint: {path}")


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
    parser.add_argument(
        "--output-format",
        dest="output_format",
        choices=["flat", "nested"],
        default="flat",
        help="输出格式：flat=每个hit独立一行（默认，适合Step4处理），nested=每个query一行含hits数组（适合标注模板）",
    )
    parser.add_argument(
        "--generate-annotation-template",
        dest="generate_annotation_template",
        action="store_true",
        help="快捷开关，等价于 nested 输出且包含 answers，输出文件名默认 embedding_{stage}_template.jsonl",
    )
    parser.add_argument(
        "--include-answers",
        dest="include_answers",
        action="store_true",
        help="flat格式时是否包含answers字段（默认不包含以减少冗余；Step4会从finglm_master.jsonl join）",
    )
    parser.add_argument(
        "--save-checkpoint",
        dest="save_checkpoint",
        action="store_true",
        help="写入 checkpoint 到 data/output/checkpoints/stage0_step_3.json（或按 stage 命名）",
    )
    parser.add_argument(
        "--checkpoint-path",
        dest="checkpoint_path",
        default=None,
        help="自定义 checkpoint 输出路径（默认 data/output/checkpoints/stage{stage}_step_3.json）",
    )
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 记录")
    parser.add_argument("--wandb-project", default="QAnchor", help="wandb 项目名")
    parser.add_argument("--wandb-run-name", default=None, help="wandb run 名称")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="wandb 模式（online/offline/disabled）",
    )
    parser.add_argument(
        "--wandb-log-interval",
        type=int,
        default=None,
        help="编码阶段每多少个 batch 记录一次内存信息到 wandb（依赖 mem_log_interval）",
    )
    parser.add_argument(
        "--exclude-pdfs",
        type=str,
        default=None,
        help="可选：按行列出 pdf_stem 的文件路径（txt 或包含 problematic[].pdf_stem 的 JSON），匹配的 PDF 将被跳过",
    )
    args = parser.parse_args()
    original_command = " ".join(sys.argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    retrieval_cfg = cfg.get("retrieval", {})

    chunk_dir = Path(data_cfg["chunk_output"])
    chunk_dir.mkdir(parents=True, exist_ok=True)
    default_output = f"data/output/retrieval/embedding_{args.stage}.jsonl"
    if args.generate_annotation_template:
        # 强制 nested + answers，默认输出文件改为 template 命名
        args.output_format = "nested"
        args.include_answers = True
        default_output = f"data/output/retrieval/embedding_{args.stage}_template.jsonl"
    output_path = Path(args.output or default_output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = select_pdf_subset(
        stage=args.stage,
        summary_path=data_cfg["summary"],
        pdf_dir=data_cfg["pdf_dir"],
        stage_config=cfg.get("stages"),
    )
    pdf_stems = [Path(rec["pdf_path"]).stem for rec in subset["records"]]

    # 可选：过滤问题 PDF
    if args.exclude_pdfs:
        exclude_path = Path(args.exclude_pdfs)
        exclude_stems: List[str] = []
        if exclude_path.suffix.lower() in {".json", ".jsonl"}:
            try:
                payload = json.loads(exclude_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "problematic" in payload:
                    exclude_stems = [
                        item.get("pdf_stem")
                        for item in payload.get("problematic", [])
                        if isinstance(item, dict) and item.get("pdf_stem")
                    ]
                elif isinstance(payload, list):
                    exclude_stems = [
                        item.get("pdf_stem") for item in payload if isinstance(item, dict) and item.get("pdf_stem")
                    ]
            except Exception:
                exclude_stems = []
        if not exclude_stems:
            # 兼容 txt，每行一个 pdf_stem
            exclude_stems = [line.strip() for line in exclude_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if exclude_stems:
            before = len(pdf_stems)
            exclude_set = set(exclude_stems)
            pdf_stems = [s for s in pdf_stems if s not in exclude_set]
            print(f"根据 exclude 列表过滤 PDF：{before} -> {len(pdf_stems)}")
        else:
            print(f"[警告] 未能从 {exclude_path} 读取有效的 pdf_stem，未应用过滤")
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
    per_pdf_chunks: Dict[str, Dict[str, Any]] = {}
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
        filtered_texts: List[str] = []
        filtered_metas: List[Dict[str, Any]] = []
        if section_blacklist_enabled and section_blacklist:
            for text, meta in zip(texts, metas):
                if _section_is_blacklisted(meta.get("section_path") or [], section_blacklist):
                    dropped_by_blacklist += 1
                    continue
                filtered_texts.append(text)
                filtered_metas.append(meta)
        else:
            filtered_texts = texts
            filtered_metas = metas
        per_pdf_chunks[stem] = {"texts": filtered_texts, "metas": filtered_metas}
    if missing_chunks:
        print(f"警告：缺少 {len(missing_chunks)} 个 chunk 文件，已跳过: {', '.join(missing_chunks[:5])}{' ...' if len(missing_chunks) > 5 else ''}")
    if dropped_by_blacklist:
        print(f"根据 section_blacklist 过滤掉 {dropped_by_blacklist} 个 chunks")
    total_candidate_chunks = sum(len(v["metas"]) for v in per_pdf_chunks.values())
    print(f"候选 child chunks: {total_candidate_chunks}")
    if total_candidate_chunks == 0:
        print("未找到可用的 chunk 数据，退出。")
        return

    # 编码
    embedding_model = args.embedding_model or retrieval_cfg.get("embedding_model") or "qwen3-embedding-0.6b"
    device = args.device or retrieval_cfg.get("device")
    batch_size = retrieval_cfg.get("batch_size", 32)
    normalize = retrieval_cfg.get("normalize_embeddings", True)
    top_k = args.top_k or retrieval_cfg.get("top_k", 30)
    restrict_to_query_pdf = retrieval_cfg.get("restrict_to_query_pdf", True)
    max_seq_length = retrieval_cfg.get("max_seq_length")
    empty_cache_interval = retrieval_cfg.get("empty_cache_interval", 10)
    precision_mode = retrieval_cfg.get("precision_mode", "auto")
    enable_empty_cache = retrieval_cfg.get("enable_empty_cache", True)
    use_inference_mode = retrieval_cfg.get("use_inference_mode", True)
    mem_log_interval = retrieval_cfg.get("mem_log_interval")
    wandb_log_interval = args.wandb_log_interval

    model_tag = _build_model_tag(
        embedding_model,
        normalize=normalize,
        max_seq_length=max_seq_length,
        precision_mode=precision_mode,
        section_blacklist_enabled=section_blacklist_enabled,
    )
    cache_dir = Path("data/output/embeddings") / model_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = None

    # 准备 tokenizer 长度函数（仅在需要记录 token 长度时使用，避免额外开销）
    token_length_fn = None

    wb = None
    if args.wandb and args.wandb_mode != "disabled":
        try:
            import wandb

            if args.wandb_mode == "offline":
                os.environ["WANDB_MODE"] = "offline"
            wb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name
                or f"{args.stage}-embed-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "stage": args.stage,
                    "output_format": args.output_format,
                    "top_k": top_k,
                    "embedding_model": embedding_model,
                    "model_tag": model_tag,
                    "batch_size": batch_size,
                    "max_seq_length": max_seq_length,
                    "normalize_embeddings": normalize,
                    "precision_mode": precision_mode,
                    "restrict_to_query_pdf": restrict_to_query_pdf,
                    "section_blacklist_enabled": section_blacklist_enabled,
                    "device": device,
                    "empty_cache_interval": empty_cache_interval,
                    "mem_log_interval": mem_log_interval,
                    "padding_strategy": None,
                    "truncation": None,
                    "model_dtype": None,
                    "emb_dtype": None,
                },
            )
        except ImportError:
            print("未安装 wandb，忽略 wandb 记录；如需开启请 pip install wandb")
            wb = None

    retriever = EmbeddingRetriever(
        model_name=embedding_model,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        max_seq_length=max_seq_length,
        empty_cache_interval=empty_cache_interval,
        precision_mode=precision_mode,
        enable_empty_cache=enable_empty_cache,
        use_inference_mode=use_inference_mode,
        mem_log_interval=mem_log_interval,
    )
    # 获取模型 dtype
    model_dtype = _get_model_dtype(retriever.model)
    if wb and model_dtype:
        try:
            wb.config.update({"model_dtype": model_dtype}, allow_val_change=True)
        except Exception:
            pass

    # 准备 tokenizer 长度统计函数（仅在日志需要时启用）
    log_token_length = bool(wb and mem_log_interval and wandb_log_interval)
    if log_token_length and hasattr(retriever.model, "tokenizer"):
        tokenizer = retriever.model.tokenizer

        def token_length_fn(text_batch: Sequence[str]) -> Sequence[int]:
            try:
                encoded = tokenizer(
                    list(text_batch),
                    padding=False,
                    truncation=False,
                    return_length=True,
                )
                lengths = encoded.get("length") or encoded.get("len")
                if lengths is None:
                    # 回退：尝试 input_ids 长度
                    if "input_ids" in encoded:
                        return [len(ids) for ids in encoded["input_ids"]]
                    return []
                return lengths
            except Exception:
                return []
    else:
        token_length_fn = None

    # 打印关键配置，便于确认运行口径
    print(
        "[Config] "
        f"model={embedding_model}, device={device}, model_dtype={model_dtype}, "
        f"batch_size={batch_size}, max_seq_length={max_seq_length}, normalize={normalize}, "
        f"precision_mode={precision_mode}, restrict_to_query_pdf={restrict_to_query_pdf}, "
        f"section_blacklist_enabled={section_blacklist_enabled}, "
        f"mem_log_interval={mem_log_interval}, wandb_log_interval={wandb_log_interval}, "
        f"empty_cache_interval={empty_cache_interval}, model_tag={model_tag}"
    )

    # 全局进度条（按 batch 计数）
    total_batches_chunks = sum(ceil(len(v["texts"]) / batch_size) for v in per_pdf_chunks.values())
    total_batches_queries = ceil(len(queries) / batch_size) if queries else 0
    global_pbar = tqdm(
        total=total_batches_chunks + total_batches_queries,
        desc="global encode",
        unit="batch",
        leave=True,
    )

    # 按 PDF 处理，复用/落盘缓存
    chunk_embs_list: List[np.ndarray] = []
    chunk_metas: List[Dict[str, Any]] = []
    chunks_by_pdf: Dict[str, List[int]] = defaultdict(list)
    cache_hits: List[str] = []
    cache_miss: List[str] = []
    cache_recomputed: List[str] = []
    emb_dtype_logged = False

    for stem, payload in per_pdf_chunks.items():
        texts = payload["texts"]
        metas = payload["metas"]
        if not texts:
            continue
        chunk_ids = [m.get("chunk_id") or f"idx_{i}" for i, m in enumerate(metas)]
        chunk_hash = _compute_chunk_hash(chunk_ids, texts)
        npz_path = cache_dir / f"{stem}.npz"
        sidecar_path = cache_dir / f"{stem}.json"

        emb_array: np.ndarray | None = None
        use_cache = False
        if npz_path.exists() and sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                emb_loaded = np.load(npz_path)
                emb_mat = emb_loaded["embeddings"]
                cache_valid = (
                    sidecar.get("model_name") == embedding_model
                    and sidecar.get("normalize_embeddings") == normalize
                    and sidecar.get("max_seq_length") == (max_seq_length or None)
                    and sidecar.get("precision_mode", "auto") == precision_mode
                    and sidecar.get("section_blacklist_enabled") == section_blacklist_enabled
                    and sidecar.get("chunk_hash") == chunk_hash
                    and sidecar.get("chunk_count") == len(chunk_ids)
                    and emb_mat.shape[0] == len(chunk_ids)
                )
                if cache_valid:
                    emb_array = emb_mat
                    use_cache = True
                    cache_hits.append(stem)
                    if global_pbar:
                        try:
                            global_pbar.update(ceil(len(texts) / batch_size))
                        except Exception:
                            pass
                else:
                    cache_recomputed.append(stem)
            except Exception:
                cache_recomputed.append(stem)
        else:
            cache_miss.append(stem)

        if emb_array is None:
            start_ts = time.time()
            log_fn = None
            if wb and mem_log_interval and wandb_log_interval:
                # 依赖 EmbeddingRetriever 内部 mem_log_interval 触发
                def log_fn(metrics: Dict[str, float]) -> None:
                    if metrics.get("batch_idx", 0) % wandb_log_interval == 0:
                        wb.log({"phase": "chunk_encode", **metrics})

            emb_array = retriever.encode_chunks(
                texts,
                log_fn=log_fn,
                token_length_fn=token_length_fn,
                progress_cb=lambda n: global_pbar.update(n),
            )
            elapsed = time.time() - start_ts
            # 持久化缓存
            np.savez(npz_path, embeddings=emb_array)
            sidecar = {
                "model_name": embedding_model,
                "normalize_embeddings": normalize,
                "max_seq_length": max_seq_length or None,
                "precision_mode": precision_mode,
                "dtype": str(emb_array.dtype),
                "section_blacklist_enabled": section_blacklist_enabled,
                "embedding_dim": int(emb_array.shape[1]) if emb_array.size else 0,
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "chunk_hash": chunk_hash,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
            if wb:
                metrics = _collect_mem_metrics()
                wb.log(
                    {
                        "phase": "chunk_encode",
                        "pdf_stem": stem,
                        "chunk_count": len(chunk_ids),
                        "chunk_encode_sec": elapsed,
                        "chunk_samples_per_sec": len(chunk_ids) / elapsed if elapsed > 0 else None,
                        **metrics,
                    }
                )
        if wb and not emb_dtype_logged and emb_array is not None:
            try:
                wb.config.update({"emb_dtype": str(emb_array.dtype)}, allow_val_change=True)
            except Exception:
                pass
            emb_dtype_logged = True

        start_idx = len(chunk_metas)
        chunk_embs_list.append(emb_array)
        chunk_metas.extend(metas)
        chunks_by_pdf[str(stem)] = list(range(start_idx, start_idx + len(metas)))

    if cache_hits or cache_miss or cache_recomputed:
        print(
            f"[缓存] 命中: {len(cache_hits)}, 缺失: {len(cache_miss)}, 失配重算: {len(cache_recomputed)}"
        )
        if cache_recomputed:
            print(f"[缓存] 失配重算 PDFs: {cache_recomputed[:5]}{' ...' if len(cache_recomputed) > 5 else ''}")
    if not chunk_metas:
        print("未能加载任何 chunk embedding，退出。")
        return

    chunk_embs = np.vstack(chunk_embs_list)

    print(f"编码 {len(queries)} 个 queries ...")
    query_texts = [q["query"] for q in queries]
    q_start = time.time()
    log_fn_q = None
    if wb and mem_log_interval and wandb_log_interval:
        def log_fn_q(metrics: Dict[str, float]) -> None:
            if metrics.get("batch_idx", 0) % wandb_log_interval == 0:
                wb.log({"phase": "query_encode", **metrics})

    query_embs = retriever.encode_queries(
        query_texts,
        log_fn=log_fn_q,
        token_length_fn=token_length_fn,
        progress_cb=lambda n: global_pbar.update(n),
    )
    q_elapsed = time.time() - q_start
    if wb:
        metrics = _collect_mem_metrics()
        wb.log(
            {
                "phase": "query_encode",
                "query_count": len(query_texts),
                "query_encode_sec": q_elapsed,
                "query_samples_per_sec": len(query_texts) / q_elapsed if q_elapsed > 0 else None,
                **metrics,
            }
        )

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

    nohit_records: List[Dict[str, Any]] = []

    # 输出 JSONL
    with output_path.open("w", encoding="utf-8") as f:
        if args.output_format == "nested":
            # 嵌套格式：每个 query 一行，hits 为数组
            for query_rec, hits in zip(queries, all_hits):
                out = dict(query_rec)
                out["hits"] = hits
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                if not hits:
                    nohit_records.append(
                        {
                            "query_id": query_rec.get("query_id"),
                            "query": query_rec.get("query"),
                            "pdf": query_rec.get("pdf"),
                            "company": query_rec.get("company"),
                            "year": query_rec.get("year"),
                        }
                    )
        else:
            # 扁平格式：每个 hit 独立一行
            total_records = 0
            for query_rec, hits in zip(queries, all_hits):
                # 提取 query 基础信息
                query_base = {
                    "query_id": query_rec.get("query_id"),
                    "query": query_rec.get("query"),
                    "pdf": query_rec.get("pdf"),
                    "company": query_rec.get("company"),
                    "year": query_rec.get("year"),
                }
                # 可选：包含 answers（调试用）
                if args.include_answers and "answers" in query_rec:
                    query_base["answers"] = query_rec["answers"]
                
                if not hits:
                    nohit_records.append(query_base)
                    continue

                # 展开每个 hit 为独立记录；优先保留 query 层的 pdf，命名 chunk 所在 pdf 为 source_pdf
                for hit in hits:
                    hit_pdf = hit.get("pdf")
                    hit_meta = {k: v for k, v in hit.items() if k != "pdf"}
                    flat_record = {**query_base, **hit_meta}
                    if hit_pdf is not None:
                        flat_record["source_pdf"] = hit_pdf
                    f.write(json.dumps(flat_record, ensure_ascii=False) + "\n")
                    total_records += 1

            print(f"已展开为 {total_records} 条扁平记录")

    format_desc = f"格式={args.output_format}"
    if args.output_format == "flat" and not args.include_answers:
        format_desc += "（不含answers，Step4需join finglm_master.jsonl）"
    print(f"已写入检索结果：{output_path} [{format_desc}]")

    if nohit_records:
        nohit_path = output_path.with_name(f"{output_path.stem}_nohits.jsonl")
        with nohit_path.open("w", encoding="utf-8") as nf:
            for rec in nohit_records:
                nf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"额外记录 {len(nohit_records)} 个无候选的 query → {nohit_path}")

    # 分布统计
    score_pool = [hit["score"] for hits in all_hits for hit in hits]
    print(f"总计候选: {len(score_pool)}")
    _print_distribution(score_pool)
    if wb:
        pct = _calc_percentiles(score_pool)
        wb.log(
            {
                "phase": "summary",
                "queries": len(queries),
                "total_chunks": len(chunk_metas),
                "total_hits": len(score_pool),
                "score_p50": pct.get("p50"),
                "score_p90": pct.get("p90"),
                "score_p99": pct.get("p99"),
                "score_min": min(score_pool) if score_pool else None,
                "score_max": max(score_pool) if score_pool else None,
                "cache_hits": len(cache_hits),
                "cache_miss": len(cache_miss),
                "cache_recomputed": len(cache_recomputed),
            }
        )

    if args.save_checkpoint:
        # 只记录本次实际使用的 chunk 文件
        chunk_files_used = [chunk_dir / f"{stem}_chunks.json" for stem in used_pdfs if (chunk_dir / f"{stem}_chunks.json").exists()]
        if args.checkpoint_path:
            checkpoint_path = Path(args.checkpoint_path)
        else:
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            checkpoint_path = Path(f"data/output/checkpoints/{args.stage}_step_3_{ts}.json")
        params = {
            "embedding_model": embedding_model,
            "top_k": top_k,
            "batch_size": batch_size,
            "normalize_embeddings": normalize,
            "restrict_to_query_pdf": restrict_to_query_pdf,
            "section_blacklist_enabled": section_blacklist_enabled,
            "output_format": args.output_format,
            "include_answers": args.include_answers,
            "model_tag": model_tag,
        }
        summary = {
            "queries": len(queries),
            "total_chunks": len(chunk_metas),
            "total_hits": len(score_pool),
            "missing_chunk_files": len(missing_chunks),
            "dropped_by_blacklist": dropped_by_blacklist,
            "nohit_queries": len(nohit_records),
            "score_percentiles": _calc_percentiles(score_pool),
            "score_min": min(score_pool) if score_pool else None,
            "score_max": max(score_pool) if score_pool else None,
            "cache_hits": cache_hits,
            "cache_miss": cache_miss,
            "cache_recomputed": cache_recomputed,
        }
        _save_checkpoint(
            checkpoint_path,
            stage=args.stage,
            chunk_files=chunk_files_used,
            qa_mapping=Path(data_cfg["qa_mapping"]),
            answers=Path(data_cfg["answers"]),
            output_file=output_path,
            params=params,
            summary={**summary, "original_command": original_command, "cwd": str(Path.cwd())},
        )

    try:
        global_pbar.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
