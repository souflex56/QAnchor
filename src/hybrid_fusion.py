"""Hybrid 融合：支持 RRF 融合 Embedding 与 BM25 结果。"""

from __future__ import annotations

from typing import Dict, List, Sequence


def rrf_fuse(
    embedding_hits: Sequence[Dict],
    bm25_hits: Sequence[Dict],
    rrf_k: int = 60,
    missing_rank: int = 9999,
    top_k: int | None = None,
) -> List[Dict]:
    """对单个 query 的两路结果做 RRF 融合。"""
    combined: Dict[str, Dict] = {}

    def _add_source(hits: Sequence[Dict], key: str) -> None:
        for hit in hits:
            cid = str(hit.get("chunk_id"))
            rank_val = hit.get("rank")
            score = hit.get("score")
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
                },
            )
            if rank_val is not None:
                entry[f"{key}_rank"] = rank_val
            entry[f"{key}_score"] = score

    _add_source(embedding_hits, "embedding")
    _add_source(bm25_hits, "bm25")

    fused: List[Dict] = []
    for entry in combined.values():
        emb_rank = entry.get("embedding_rank", missing_rank) or missing_rank
        bm_rank = entry.get("bm25_rank", missing_rank) or missing_rank
        rrf_score = 1.0 / (rrf_k + emb_rank) + 1.0 / (rrf_k + bm_rank)
        fused.append({**entry, "score": rrf_score})

    fused.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    for i, item in enumerate(fused, start=1):
        item["rank"] = i
    return fused


__all__ = ["rrf_fuse"]
