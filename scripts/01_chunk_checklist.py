#!/usr/bin/env python3
"""分块抽查 checklist（Step 2.1）。

从已生成的 chunk JSON 中抽样 3–5 份，输出页码连续性、section_path 覆盖、父子关联与粒度统计。
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# 确保仓库根目录在 sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import select_pdf_subset  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _page_gaps(pages: List[int]) -> List[List[int]]:
    gaps: List[List[int]] = []
    if not pages:
        return gaps
    pages = sorted(set(pages))
    for i in range(1, len(pages)):
        if pages[i] - pages[i - 1] > 1:
            gaps.append([pages[i - 1] + 1, pages[i] - 1])
    return gaps


def analyze_chunk_file(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    parents = data.get("parents", [])
    children = data.get("children", [])
    overview = data.get("overview", {}) or {}
    overview_pages = (overview.get("counts") or {}).get("page_count") or 0

    # 页码与 section 覆盖
    pages = []
    section_non_empty = 0
    section_level_non_empty = 0
    for chunk in parents + children:
        meta = chunk.get("metadata") or {}
        pages.extend(meta.get("page_numbers") or ([chunk.get("page_number")] if chunk.get("page_number") else []))
        if meta.get("section_path"):
            section_non_empty += 1
        if meta.get("section_level") is not None:
            section_level_non_empty += 1
    gaps = _page_gaps([p for p in pages if p is not None])

    # 父子关联
    parent_ids = {p.get("chunk_id") for p in parents if p.get("chunk_id")}
    child_with_parent = sum(1 for c in children if c.get("parent_id") in parent_ids)
    orphan_children = len(children) - child_with_parent
    parents_with_children = sum(1 for p in parents if p.get("child_ids"))

    # 粒度（基于 metadata.char_count 或 content 长度）
    def chunk_len(c: Dict[str, Any]) -> int:
        meta = c.get("metadata") or {}
        return int(meta.get("char_count") or len(c.get("content") or ""))

    lengths = [chunk_len(c) for c in children]
    stats = {}
    if lengths:
        stats = {
            "min": min(lengths),
            "p50": int(statistics.median(lengths)),
            "p90": int(statistics.quantiles(lengths, n=10)[8]) if len(lengths) >= 2 else int(lengths[0]),
            "max": max(lengths),
        }

    page_cov_ratio = round(len(set(pages)) / overview_pages, 3) if overview_pages else None

    gap_preview = gaps[:10]
    if len(gap_preview) < len(gaps):
        gap_preview.append(f"...共{len(gaps)}段缺口")

    return {
        "file": path.name,
        "parents": len(parents),
        "children": len(children),
        "pages_covered": len(set(pages)),
        "pages_total": overview_pages,
        "page_cov_ratio": page_cov_ratio,
        "page_gaps": gap_preview,
        "section_path_coverage": f"{section_non_empty}/{len(parents)+len(children)}",
        "section_level_filled": section_level_non_empty,
        "parents_with_children": parents_with_children,
        "orphan_children": orphan_children,
        "length_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分块抽查 checklist (Step 2.1)")
    parser.add_argument("--stage", default="stage0", help="阶段名称，默认 stage0")
    parser.add_argument("--config", default="config/weak_supervision_config.yaml", help="主配置文件")
    parser.add_argument("--sample", type=int, default=5, help="抽查文件数，默认 5")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    chunk_dir = Path(cfg["data"]["chunk_output"])

    subset = select_pdf_subset(
        stage=args.stage,
        summary_path=cfg["data"]["summary"],
        pdf_dir=cfg["data"]["pdf_dir"],
        stage_config=cfg["stages"],
    )

    expected = []
    for rec in subset["records"]:
        stem = Path(rec["pdf_path"]).stem
        f = chunk_dir / f"{stem}_chunks.json"
        if f.exists():
            expected.append(f)
    sample_files = expected[: args.sample]

    if not sample_files:
        print("未找到可抽查的 chunk 文件，请先运行 01_batch_chunking.py")
        return

    print(f"[Stage: {args.stage}] 抽查 {len(sample_files)} 份 chunk：")
    for f in sample_files:
        report = analyze_chunk_file(f)
        print(f"- {report['file']}")
        print(f"  parents/children: {report['parents']} / {report['children']}")
        cov_ratio = f"{report['page_cov_ratio']*100:.1f}%" if report['page_cov_ratio'] is not None else "未知"
        print(f"  pages_covered: {report['pages_covered']}/{report['pages_total'] or '未知'} (覆盖率: {cov_ratio}), gaps: {report['page_gaps'] or '无'}")
        print(f"  section_path_coverage: {report['section_path_coverage']}, section_level_filled: {report['section_level_filled']}")
        print(f"  parents_with_children: {report['parents_with_children']}, orphan_children: {report['orphan_children']}")
        if report["length_stats"]:
            s = report['length_stats']
            print(f"  child length chars (min/p50/p90/max): {s['min']}/{s['p50']}/{s['p90']}/{s['max']}")
        print("")


if __name__ == "__main__":
    main()
