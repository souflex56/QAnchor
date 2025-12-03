"""Export predefined FinGLM subsets and dataset catalog."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

SUBSET_SPECS = [
    {
        "name": "single_company_single_year_core",
        "description": "1公司+1年份，类型 ∈ {1,1-2,2-1,2-2,3-1}，RAG 主战场",
        "usage": ["retrieval", "reranker", "neutral_answering"],
        "filters": {
            "cross_dimension": ["1co_1yr"],
            "types": ["1", "1-2", "2-1", "2-2", "3-1"],
        },
        "notes": "用于检索/重排/中性回答蒸馏，优先覆盖已有 PDF 的公司年份。",
    },
    {
        "name": "single_company_multi_year",
        "description": "1公司 + 多年份（时序/趋势问题）",
        "usage": ["advanced_retrieval", "trend_reasoning"],
        "filters": {
            "cross_dimension": ["1co_multi_yr"],
        },
        "notes": "阶段 2 使用的数据，支持时序对比和增长率类问答。",
    },
    {
        "name": "multi_company_single_year",
        "description": "多公司 + 单一年份（横向对比）",
        "usage": ["analysis", "evaluation"],
        "filters": {
            "cross_dimension": ["multi_co_1yr"],
        },
        "notes": "当前阶段优先级低，可用于未来的横向对比检索。",
    },
    {
        "name": "multi_company_multi_year",
        "description": "多公司 + 多个年份（复杂对比）",
        "usage": ["analysis"],
        "filters": {
            "cross_dimension": ["multi_co_multi_yr"],
        },
        "notes": "数量最少（77），作为高难度 QA 预留。",
    },
    {
        "name": "type_3_2_concept",
        "description": "Type 3-2 概念题，可作为段永平风格 teacher",
        "usage": ["concept_knowledge", "duan_style_teacher"],
        "filters": {
            "types": ["3-2"],
        },
        "notes": "后续可改写成段式回答，用于风格 SFT。",
    },
    {
        "name": "generic_no_company_year",
        "description": "无公司、无年份信息的问答（cross_dimension == generic）",
        "usage": ["concept_knowledge"],
        "filters": {
            "cross_dimension": ["generic"],
        },
        "notes": "与 type_3_2 基本一致，用于快速定位概念题。",
    },
]


def load_master(master_path: Path) -> List[Dict[str, Any]]:
    if not master_path.exists():
        raise FileNotFoundError(f"未找到 master JSONL: {master_path}")
    records: List[Dict[str, Any]] = []
    with master_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_dimension_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def record_matches(record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True

    def match_list(field: str, allowed: Sequence[str]) -> bool:
        value = str(record.get(field, "") or "")
        return value in allowed

    if "types" in filters and not match_list("type", filters["types"]):
        return False
    if "cross_dimension" in filters and not match_list("cross_dimension", filters["cross_dimension"]):
        return False
    if "has_company" in filters:
        if bool(record.get("has_company")) != bool(filters["has_company"]):
            return False
    if "has_year" in filters:
        if bool(record.get("has_year")) != bool(filters["has_year"]):
            return False
    if "source_dataset" in filters:
        if record.get("source_dataset") not in filters["source_dataset"]:
            return False
    return True


def export_subset(
    records: List[Dict[str, Any]],
    spec: Dict[str, Any],
    output_dir: Path,
    master_path: Path,
) -> Dict[str, Any]:
    subset_records = [r for r in records if record_matches(r, spec.get("filters", {}))]
    ids = sorted(r.get("master_id", r.get("id")) for r in subset_records)

    index_data = {
        "name": spec["name"],
        "description": spec.get("description", ""),
        "usage": spec.get("usage", []),
        "notes": spec.get("notes"),
        "filters": spec.get("filters", {}),
        "count": len(ids),
        "ids": ids,
        "source": str(master_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec['name']}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    return {
        "name": spec["name"],
        "description": spec.get("description", ""),
        "usage": spec.get("usage", []),
        "notes": spec.get("notes"),
        "filters": spec.get("filters", {}),
        "count": len(ids),
        "index_path": str(output_path),
    }


def build_catalog(
    subsets: List[Dict[str, Any]],
    master_path: Path,
    dimension_stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "master_path": str(master_path),
        "dimension_stats_snapshot": dimension_stats.get("summary", {}),
        "cross_dimension_snapshot": dimension_stats.get("cross_analysis") or dimension_stats.get("cross_summary") or {},
        "subsets": subsets,
    }

