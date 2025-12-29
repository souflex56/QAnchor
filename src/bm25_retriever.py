"""BM25 检索器：按 PDF 构建分词索引并返回 Top-K 结果。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def _ensure_jieba():
    try:
        import jieba  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("缺少 jieba，请先安装：pip install jieba") from e


def _ensure_rankbm25():
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("缺少 rank-bm25，请先安装：pip install rank-bm25") from e


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    # 统一全角/半角并压缩空白
    normalized = unicodedata.normalize("NFKC", str(text))
    return " ".join(normalized.split())


def _tokenize(text: str) -> List[str]:
    _ensure_jieba()
    import jieba

    return list(jieba.cut(text))


@dataclass
class ChunkDoc:
    chunk_id: str
    text: str
    meta: Dict


class BM25Retriever:
    """按 pdf_stem 管理 BM25 索引，仅同文档检索。"""

    def __init__(self) -> None:
        _ensure_rankbm25()
        from rank_bm25 import BM25Okapi

        self.BM25 = BM25Okapi
        self.indices: Dict[str, Tuple[object, List[ChunkDoc]]] = {}

    def add_pdf(self, pdf_stem: str, chunks: Sequence[Dict]) -> None:
        documents: List[ChunkDoc] = []
        tokenized_docs: List[List[str]] = []
        for child in chunks:
            cid = child.get("chunk_id") or ""
            text = _normalize_text(child.get("content") or child.get("text") or "")
            meta = child.get("metadata") or {}
            documents.append(
                ChunkDoc(
                    chunk_id=cid,
                    text=text,
                    meta={
                        "chunk_id": cid,
                        "parent_id": child.get("parent_id"),
                        "pdf": meta.get("pdf") or child.get("pdf") or f"{pdf_stem}.pdf",
                        "pdf_stem": pdf_stem,
                        "page_numbers": meta.get("page_numbers") or child.get("page_numbers") or [],
                        "section_path": meta.get("section_path") or child.get("section_path") or [],
                    },
                )
            )
            tokenized_docs.append(_tokenize(text))

        if not documents:
            self.indices[pdf_stem] = (None, [])
            return

        index = self.BM25(tokenized_docs)
        self.indices[pdf_stem] = (index, documents)

    def retrieve(self, query: str, pdf_stem: str, top_k: int) -> List[Dict]:
        index_tuple = self.indices.get(pdf_stem)
        if not index_tuple:
            return []
        index, documents = index_tuple
        if not index or not documents:
            return []

        tokenized_query = _tokenize(_normalize_text(query))
        scores = index.get_scores(tokenized_query)
        # 取 top_k
        k = min(top_k, len(scores))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[Dict] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            doc = documents[idx]
            results.append(
                {
                    **doc.meta,
                    "text": doc.text,
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )
        return results


__all__ = ["BM25Retriever"]
