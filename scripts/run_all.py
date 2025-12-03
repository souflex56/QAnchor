#!/usr/bin/env python3
"""Run master build -> clean/dedup -> EDA -> subset export in one go."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

COMMANDS: List[List[str]] = [
    [
        PYTHON,
        "scripts/build_master_table.py",
        "--raw-dir",
        "data/input/finglm-data _raw",
        "--output",
        "finglm_data_store/finglm_master.jsonl",
        "--stats-output",
        "finglm_data_store/finglm_master_stats.json",
    ],
    [
        PYTHON,
        "scripts/clean_and_dedup.py",
        "--input",
        "finglm_data_store/finglm_master.jsonl",
        "--output",
        "finglm_data_store/finglm_master_dedup.jsonl",
        "--report-dir",
        "finglm_data_store",
    ],
    [
        PYTHON,
        "scripts/analyze_qa_types.py",
        "--input",
        "finglm_data_store/finglm_master_dedup.jsonl",
        "--output-dir",
        ".",
        "--clean-report",
        "finglm_data_store/clean_dedup_report.json",
    ],
    [
        PYTHON,
        "scripts/analyze_qa_dimensions.py",
        "--input",
        "finglm_data_store/finglm_master_dedup.jsonl",
        "--output-dir",
        ".",
        "--clean-report",
        "finglm_data_store/clean_dedup_report.json",
    ],
    [
        PYTHON,
        "scripts/export_subsets.py",
        "--master",
        "finglm_data_store/finglm_master_dedup.jsonl",
        "--dimension-stats",
        "finglm_qa_dimension_analysis.json",
        "--index-dir",
        "finglm_data_store/index",
        "--catalog",
        "finglm_data_store/dataset_catalog.json",
    ],
]


def run(cmd: List[str]) -> None:
    print(f"\n>>> 执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    for cmd in COMMANDS:
        run(cmd)
    print("\n✅ 全部步骤执行完毕！")


if __name__ == "__main__":
    main()

