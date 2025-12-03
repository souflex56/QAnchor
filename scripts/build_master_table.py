#!/usr/bin/env python3
"""Build FinGLM master JSONL from raw data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.master_builder import RAW_DATA_DIR, build_master_records, write_jsonl, write_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 FinGLM 主表 JSONL")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="原始 FinGLM 数据目录（包含 pre/A/B/C）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "finglm_master.jsonl",
        help="输出 master JSONL 路径",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "finglm_master_stats.json",
        help="输出统计信息 JSON 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, stats = build_master_records(args.raw_dir)
    write_jsonl(records, args.output)
    print(f"✅ 写入 {len(records)} 条记录到 {args.output}")

    if args.stats_output:
        write_stats(stats, args.stats_output)
        print(f"📊 统计信息保存到 {args.stats_output}")


if __name__ == "__main__":
    main()

