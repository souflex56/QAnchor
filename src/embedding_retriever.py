"""Embedding 检索工具：封装向量编码与 Top-K 计算（Stage0 embedding-only）。"""

from __future__ import annotations

import gc
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import psutil
from tqdm import tqdm


class EmbeddingRetriever:
    """加载 embedding 模型并执行批量 Top-K 检索。"""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: int | None = None,
        empty_cache_interval: int = 10,
        precision_mode: str = "auto",  # auto | fp16 | fp32
        enable_empty_cache: bool = True,
        use_inference_mode: bool = True,
        mem_log_interval: int | None = None,
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
        self.empty_cache_interval = max(1, int(empty_cache_interval)) if empty_cache_interval else 10
        self.enable_empty_cache = enable_empty_cache
        self.use_inference_mode = use_inference_mode
        self.precision_mode = (precision_mode or "auto").lower()
        self.mem_log_interval = max(1, int(mem_log_interval)) if mem_log_interval else None

        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            try:
                self.model.max_seq_length = max_seq_length
            except Exception:
                # 某些模型未暴露该属性，忽略即可
                pass
        if device and device != "cpu":
            if self.precision_mode in ("auto", "fp16"):
                try:
                    self.model = self.model.half()
                except Exception:
                    # 个别模型可能不支持半精度，忽略
                    pass
            # precision_mode=fp32 时显式不做 half
        self.model.eval()

    def _log_memory(self, prefix: str) -> None:
        """可选的内存日志，便于观察显存压力。"""
        try:
            vm = psutil.virtual_memory()
            print(f"{prefix} RAM used={vm.used/1e9:.2f}G avail={vm.available/1e9:.2f}G")
            if torch.backends.mps.is_available():
                cur = torch.mps.current_allocated_memory() / 1024**3
                driver = torch.mps.driver_allocated_memory() / 1024**3
                print(f"{prefix} MPS current={cur:.2f}GiB driver={driver:.2f}GiB")
            if torch.cuda.is_available():
                cur = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"{prefix} CUDA allocated={cur:.2f}GiB reserved={reserved:.2f}GiB")
        except Exception:
            # 日志失败不影响主流程
            pass

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
                ctx = torch.inference_mode() if self.use_inference_mode else nullcontext()
                with ctx:
                    emb = self.model.encode(
                        batch,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                        show_progress_bar=False,
                    )
                embeddings.append(emb)

                # 周期性清理缓存，降低 MPS/CUDA 内存压力
                step_idx = start // self.batch_size
                if self.enable_empty_cache and step_idx % self.empty_cache_interval == 0:
                    gc.collect()
                    if torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                # 可选：打印内存使用
                if self.mem_log_interval and step_idx % self.mem_log_interval == 0:
                    self._log_memory(prefix=f"[batch {step_idx}]")
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
