"""Embedding 检索工具：封装向量编码与 Top-K 计算（Stage0 embedding-only）。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from tqdm import tqdm


class EmbeddingRetriever:
    """加载 embedding 模型并执行批量 Top-K 检索。"""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - 依赖缺失时抛错
            raise ImportError(
                "缺少 sentence-transformers，请先安装：pip install sentence-transformers"
            ) from e

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        embeddings: List[np.ndarray] = []
        try:
            for start in tqdm(
                range(0, len(texts), self.batch_size),
                desc=f"encoding ({len(texts)} items)",
                unit="batch",
            ):
                batch = texts[start : start + self.batch_size]
                emb = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                )
                embeddings.append(emb)
        except Exception as e:
            print(f"编码失败（共 {len(texts)} 条）：{e}")
            raise

        if not embeddings:
            return np.zeros((0, 0))
        return np.vstack(embeddings)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """批量编码 query 文本。"""
        return self._encode(queries)

    def encode_chunks(self, chunk_texts: Sequence[str]) -> np.ndarray:
        """批量编码 chunk 文本。"""
        return self._encode(chunk_texts)

    def retrieve_top_k(
        self,
        query_embeddings: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunk_metadata: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[List[Dict[str, Any]]]:
        """基于余弦相似度返回每个 query 的 Top-K 结果。"""
        if chunk_embeddings.shape[0] != len(chunk_metadata):
            raise ValueError("chunk_embeddings 与 chunk_metadata 数量不一致")
        if chunk_embeddings.size == 0:
            return [[] for _ in range(query_embeddings.shape[0])]

        score_matrix = np.matmul(query_embeddings, chunk_embeddings.T)  # 已归一化时即余弦
        actual_k = min(top_k, chunk_embeddings.shape[0])
        results: List[List[Dict[str, Any]]] = []

        for i in range(score_matrix.shape[0]):
            row = score_matrix[i]
            if actual_k < len(row):
                idx = np.argpartition(-row, actual_k - 1)[:actual_k]
                idx = idx[np.argsort(-row[idx])]
            else:
                idx = np.argsort(-row)

            hits: List[Dict[str, Any]] = []
            for rank, j in enumerate(idx, start=1):
                meta = dict(chunk_metadata[j])
                meta.update({"score": float(row[j]), "rank": rank})
                hits.append(meta)
            results.append(hits)

        return results


__all__ = ["EmbeddingRetriever"]
