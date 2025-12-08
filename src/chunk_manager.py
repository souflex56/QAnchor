"""分块管理工具：加载 ZenParse chunk JSON 并提供查询索引。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _pdf_keys(source: Optional[str], path: Path) -> List[str]:
    """生成用于索引的 pdf 关键字集合。"""
    keys = set()
    if source:
        keys.add(source)
        keys.add(Path(source).name)
    keys.add(path.name)
    # 去掉 _chunks 后缀的基名
    stem = path.stem
    if stem.endswith("_chunks"):
        keys.add(stem[: -len("_chunks")])
    else:
        keys.add(stem)
    return [k for k in keys if k]


class ChunkIndex:
    """为 chunk JSON 构建检索索引。"""

    def __init__(self, chunk_files: Iterable[Path]) -> None:
        self.raw: Dict[str, Dict[str, Any]] = {}
        self.by_pdf: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.by_id: Dict[str, Dict[str, Any]] = {}
        self.children_by_parent: Dict[str, List[Dict[str, Any]]] = {}

        for path in chunk_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            source = data.get("source")
            children = data.get("children") or data.get("chunks") or []
            parents = data.get("parents") or []
            keys = _pdf_keys(source, path)

            for key in keys:
                self.by_pdf[key] = {"children": children, "parents": parents}
                self.raw[key] = data

            for parent in parents:
                parent_id = parent.get("chunk_id")
                if parent_id:
                    self.by_id[parent_id] = parent

            for child in children:
                cid = child.get("chunk_id")
                if cid:
                    self.by_id[cid] = child
                pid = child.get("parent_id")
                if pid:
                    self.children_by_parent.setdefault(pid, []).append(child)

    def get_chunks_by_pdf(self, pdf_key: str) -> List[Dict[str, Any]]:
        bucket = self.by_pdf.get(pdf_key)
        if bucket:
            return bucket.get("children", []) or bucket.get("parents", [])
        # 兼容传入完整路径
        alt_key = Path(pdf_key).name
        bucket = self.by_pdf.get(alt_key)
        return (bucket.get("children", []) or bucket.get("parents", [])) if bucket else []

    def get_children(self, parent_id: str) -> List[Dict[str, Any]]:
        return self.children_by_parent.get(parent_id, [])

    def get_parent(self, child_id: str) -> Optional[Dict[str, Any]]:
        child = self.by_id.get(child_id)
        if not child:
            return None
        pid = child.get("parent_id")
        return self.by_id.get(pid) if pid else None

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.by_id.get(chunk_id)


def load_chunks(chunk_dir: str | Path) -> ChunkIndex:
    """从目录中加载 *_chunks.json 文件并返回索引。"""
    chunk_dir = Path(chunk_dir)
    chunk_files = sorted(chunk_dir.glob("*_chunks.json"))
    return ChunkIndex(chunk_files)


def get_chunks_by_pdf(index: ChunkIndex, pdf_key: str) -> List[Dict[str, Any]]:
    """便捷访问：按 pdf 基名或 source 获取 chunks。"""
    return index.get_chunks_by_pdf(pdf_key)


__all__ = ["ChunkIndex", "load_chunks", "get_chunks_by_pdf"]
