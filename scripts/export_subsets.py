#!/usr/bin/env python3
"""Export predefined FinGLM subsets and dataset catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.subset_exporter import (  # noqa: E402
    SUBSET_SPECS,
    build_catalog,
    export_subset,
    load_dimension_stats,
    load_master,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 FinGLM 子集索引与 dataset catalog")
    parser.add_argument(
        "--master",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "finglm_master_dedup.jsonl",
        help="输入 master JSONL（建议使用去重后的版本）",
    )
    parser.add_argument(
        "--dimension-stats",
        type=Path,
        default=PROJECT_ROOT / "finglm_qa_dimension_analysis.json",
        help="维度分析 JSON（用于 catalog 快照）",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "index",
        help="子集索引输出目录",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "dataset_catalog.json",
        help="dataset catalog 输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_master(args.master)
    dim_stats = load_dimension_stats(args.dimension_stats)

    subsets_meta = []
    for spec in SUBSET_SPECS:
        meta = export_subset(records, spec, args.index_dir, args.master)
        subsets_meta.append(meta)

    catalog = build_catalog(subsets_meta, args.master, dim_stats)
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    with args.catalog.open("w", encoding="utf-8") as f:
        import json

        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print("✅ 子集索引与 catalog 已生成")
    print(f"- 索引目录: {args.index_dir}")
    print(f"- catalog: {args.catalog}")


if __name__ == "__main__":
    main()

