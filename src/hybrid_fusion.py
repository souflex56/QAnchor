"""Hybrid 融合：支持两路结果的 RRF / weighted_sum / max。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def _to_float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_weights(
    embedding_weight: float | None,
    bm25_weight: float | None,
) -> Tuple[float, float, bool]:
    """返回两路权重与“是否显式配置权重”的标记。"""
    has_explicit_weights = embedding_weight is not None or bm25_weight is not None
    emb_w = 1.0 if embedding_weight is None else float(embedding_weight)
    bm_w = 1.0 if bm25_weight is None else float(bm25_weight)
    return emb_w, bm_w, has_explicit_weights


def _minmax_normalize(values: Sequence[float | None]) -> List[float]:
    """按单一路径做 Min-Max 归一化。无差异时统一置 0。"""
    valid = [v for v in values if v is not None]
    if not valid:
        return [0.0] * len(values)
    min_v = min(valid)
    max_v = max(valid)
    if max_v <= min_v:
        return [0.0] * len(values)
    scale = max_v - min_v
    normalized: List[float] = []
    for v in values:
        if v is None:
            normalized.append(0.0)
        else:
            normalized.append((v - min_v) / scale)
    return normalized


def fuse_two_way(
    embedding_hits: Sequence[Dict],
    bm25_hits: Sequence[Dict],
    fusion_method: str = "rrf",
    embedding_weight: float | None = None,
    bm25_weight: float | None = None,
    rrf_k: int = 60,
    missing_rank: int = 9999,
    top_k: int | None = None,
) -> List[Dict]:
    """对单个 query 的两路结果做融合。"""
    method = (fusion_method or "rrf").lower()
    if method not in {"rrf", "weighted_sum", "max"}:
        method = "rrf"

    combined: Dict[str, Dict] = {}

    def _add_source(hits: Sequence[Dict], key: str) -> None:
        for hit in hits:
            cid = str(hit.get("chunk_id"))
            rank_val = _to_int_or_none(hit.get("rank"))
            score = _to_float_or_none(hit.get("score"))
            entry = combined.setdefault(
                cid,
                {
                    "chunk_id": hit.get("chunk_id"),
                    "parent_id": hit.get("parent_id"),
                    "pdf": hit.get("pdf"),
                    "pdf_stem": hit.get("pdf_stem"),
                    "page_numbers": hit.get("page_numbers") or [],
                    "section_path": hit.get("section_path") or [],
                    "embedding_rank": missing_rank,
                    "bm25_rank": missing_rank,
                    "embedding_score": None,
                    "bm25_score": None,
                },
            )
            if rank_val is not None:
                entry[f"{key}_rank"] = rank_val
            entry[f"{key}_score"] = score

    _add_source(embedding_hits, "embedding")
    _add_source(bm25_hits, "bm25")

    emb_w, bm_w, _ = _resolve_weights(embedding_weight, bm25_weight)
    entries = list(combined.values())
    fused: List[Dict] = []

    if method == "rrf":
        for entry in entries:
            emb_rank = entry.get("embedding_rank", missing_rank) or missing_rank
            bm_rank = entry.get("bm25_rank", missing_rank) or missing_rank
            score = emb_w / (rrf_k + emb_rank) + bm_w / (rrf_k + bm_rank)
            fused.append({**entry, "score": score})
    else:
        emb_scores = [_to_float_or_none(entry.get("embedding_score")) for entry in entries]
        bm_scores = [_to_float_or_none(entry.get("bm25_score")) for entry in entries]
        emb_norm = _minmax_normalize(emb_scores)
        bm_norm = _minmax_normalize(bm_scores)
        for idx, entry in enumerate(entries):
            emb_part = emb_w * emb_norm[idx]
            bm_part = bm_w * bm_norm[idx]
            if method == "weighted_sum":
                score = emb_part + bm_part
            else:  # method == "max"
                score = max(emb_part, bm_part)
            fused.append({**entry, "score": score})

    fused.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    for i, item in enumerate(fused, start=1):
        item["rank"] = i
    return fused


def rrf_fuse(
    embedding_hits: Sequence[Dict],
    bm25_hits: Sequence[Dict],
    rrf_k: int = 60,
    missing_rank: int = 9999,
    top_k: int | None = None,
) -> List[Dict]:
    """兼容旧接口：等权 RRF。"""
    return fuse_two_way(
        embedding_hits=embedding_hits,
        bm25_hits=bm25_hits,
        fusion_method="rrf",
        embedding_weight=None,
        bm25_weight=None,
        rrf_k=rrf_k,
        missing_rank=missing_rank,
        top_k=top_k,
    )


__all__ = ["fuse_two_way", "rrf_fuse"]
