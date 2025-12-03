#!/usr/bin/env python3
"""Analyze question type distribution (cleaned master)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.qa_analyzer import analyze_types, write_type_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinGLM 问答类型分析（使用 cleaned master）")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "finglm_master_dedup.jsonl",
        help="输入 JSONL（建议使用去重后的 master）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="输出目录（生成 finglm_data_type_analysis.{json,md}）",
    )
    parser.add_argument(
        "--clean-report",
        type=Path,
        default=PROJECT_ROOT / "finglm_data_store" / "clean_dedup_report.json",
        help="可选：清洗报告路径，用于在报告中显示清洗前后对比",
    )
    return parser.parse_args()


def load_cleaning_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    args = parse_args()
    report = analyze_types(args.input)
    cleaning_summary = load_cleaning_summary(args.clean_report)
    json_path, md_path = write_type_reports(report, args.output_dir, cleaning_summary=cleaning_summary)
    print(f"✅ 类型分析完成\nJSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()

