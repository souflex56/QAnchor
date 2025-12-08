"""数据加载与选样工具，服务 Stage0/Stage1.

功能：
- 读取 FinGLM 问答映射（CSV）
- 读取标准答案（JSONL）
- 按 Stage 配置从 summary CSV 中选取多 QA 的 PDF 子集
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """验证 DataFrame 是否包含所需列，不满足则抛出可读错误。"""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {', '.join(missing)}")


def load_qa_mapping(path: str | Path) -> pd.DataFrame:
    """读取问答映射 CSV，返回 DataFrame。"""
    df = pd.read_csv(path)
    _ensure_columns(
        df,
        required=(
            "status",
            "master_id",
            "company",
            "year",
            "report_paths",
            "question",
        ),
    )
    return df


def load_answers(path: str | Path) -> List[Dict[str, Any]]:
    """读取标准答案 JSONL，每行解析为字典。"""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_pdf_subset(
    stage: str,
    summary_path: str | Path,
    pdf_dir: str | Path,
    stage_config: Optional[Dict[str, Any]] = None,
    prefer_multi_qa: bool = True,
) -> Dict[str, Any]:
    """按 Stage 配置选取 PDF 子集，优先多 QA。

    以 qa_count 为主目标，可选 pdf_count_cap 作为安全上限，防止长尾无限累加。
    返回：{"records": [ {...} ], "stats": {...}}
    """

    summary = pd.read_csv(summary_path)
    _ensure_columns(summary, required=("pdf_path", "company_year_key", "qa_count"))

    qa_target: Optional[int] = None
    pdf_cap: Optional[int] = None
    if stage_config and stage in stage_config:
        qa_target = stage_config[stage].get("qa_count")
        pdf_cap = stage_config[stage].get("pdf_count_cap")

    if prefer_multi_qa:
        sort_cols = ["qa_count"]
        # 有累积列时，用于稳定排序；没有时忽略
        if "qa_count_cumulative" in summary.columns:
            sort_cols.append("qa_count_cumulative")
        summary = summary.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    else:
        summary = summary.sort_values("qa_count_cumulative" if "qa_count_cumulative" in summary.columns else "qa_count")

    records: List[Dict[str, Any]] = []
    qa_sum = 0
    for _, row in summary.iterrows():
        if pdf_cap is not None and len(records) >= int(pdf_cap):
            break
        if qa_target is not None and qa_sum >= int(qa_target):
            break

        pdf_rel_path = str(row["pdf_path"])
        qa_count = int(row["qa_count"])
        records.append(
            {
                "pdf_path": str(Path(pdf_dir) / pdf_rel_path),
                "pdf_rel_path": pdf_rel_path,
                "company_year_key": row["company_year_key"],
                "qa_count": qa_count,
                "qa_count_cumulative": int(row.get("qa_count_cumulative", len(records) + 1)),
            }
        )
        qa_sum += qa_count

    stats = {
        "stage": stage,
        "total_pdfs": len(records),
        "total_qa": qa_sum,
        "qa_target": qa_target,
        "pdf_cap": pdf_cap,
    }

    return {"records": records, "stats": stats}


__all__ = ["load_qa_mapping", "load_answers", "select_pdf_subset"]
