#!/usr/bin/env python3
"""Clean placeholder answers and deduplicate FinGLM master records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.text_processor import clean_and_dedup, load_records, save_clean_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗与去重 FinGLM master 表")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "finglm_master.jsonl",
        help="输入 master JSONL 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "finglm_master_dedup.jsonl",
        help="输出 cleaned/dedup JSONL 路径",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store",
        help="报告输出目录（clean_dedup_report.*）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    cleaned, stats = clean_and_dedup(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    json_path, md_path = save_clean_report(stats, args.report_dir)

    print(f"✅ 清洗+去重完成：输入 {stats['before_count']} 条 → 过滤后 {stats['after_filter']} → 去重后 {stats['after_dedup']}")
    print(f"👉 cleaned 输出: {args.output}")
    print(f"📄 报告: {json_path}, {md_path}")


if __name__ == "__main__":
    main()

