#!/usr/bin/env python3
"""Map type=1 & 1co_1yr QAs to report PDFs using normalized company names."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASTER = PROJECT_ROOT / "finglm_data_store" / "finglm_master_dedup.jsonl"
DEFAULT_REPORTS_CSV = PROJECT_ROOT / "data" / "input" / "finglm-data _raw" / "reports" / "reports_list.csv"
OUT_JSON = PROJECT_ROOT / "finglm_data_store" / "type1_1co1yr_pdf_map.json"
OUT_CSV = PROJECT_ROOT / "finglm_data_store" / "type1_1co1yr_pdf_map.csv"


def normalize_company(name: str) -> str:
    """Normalize company name for matching (remove parentheses chars, spaces)."""
    name = name.strip()
    if not name:
        return ""
    # Remove only the parentheses characters, keep inside content (e.g., "(集团)" -> "集团")
    name = re.sub(r"[()（）]", "", name)
    # Remove spaces and common separators
    name = re.sub(r"[\s\t\r\n]+", "", name)
    return name


def load_reports(csv_path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Load reports_list.csv into a mapping of normalized company#year -> report metadata."""
    mapping: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            name, path = row[0], row[1]
            parts = name.split("__")
            full = short = year = ""
            if len(parts) >= 5:
                _, full, _, short, year_part = parts[:5]
                year = (
                    year_part.replace("年", "")
                    .replace("年度报告.pdf", "")
                    .replace(".pdf", "")
                    .strip("_")
                )
            for comp in (full, short):
                norm = normalize_company(comp)
                if norm and year:
                    key = f"{norm}#{year}"
                    mapping[key].append({"name": name, "path": path})
    return mapping


def load_type1_1co1yr(master_path: Path) -> List[Dict[str, object]]:
    """Load type=1 & cross_dimension=1co_1yr records (retain full payload)."""
    records: List[Dict[str, object]] = []
    with master_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "1" or obj.get("cross_dimension") != "1co_1yr":
                continue
            company = (
                obj.get("primary_company")
                or obj.get("ent_name")
                or obj.get("ent_short_name")
                or ""
            )
            year = obj.get("primary_year") or obj.get("year") or ""
            records.append(
                {
                    "record": obj,  # full master record
                    "master_id": obj.get("master_id"),
                    "question": obj.get("question"),
                    "company": company,
                    "year": year,
                }
            )
    return records


def map_reports(
    records: List[Dict[str, object]], report_map: Dict[str, List[Dict[str, str]]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    matched: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for rec in records:
        company = rec.get("company") or ""
        year = rec.get("year") or ""
        norm = normalize_company(company)
        key = f"{norm}#{year}" if norm and year else None
        base_record = dict(rec.get("record", {}))
        base_record["company_norm"] = norm
        base_record["key_for_pdf"] = key
        if key and key in report_map:
            base_record["reports"] = report_map[key]
            matched.append(base_record)
        else:
            missing.append(base_record)
    return matched, missing


def write_outputs(matched: List[Dict[str, object]], missing: List[Dict[str, object]]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total": len(matched) + len(missing),
        "matched": len(matched),
        "missing": len(missing),
    }
    data = {"summary": summary, "matched": matched, "missing": missing}
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV (one row per record; multiple reports collapsed with |)
    rows = []
    for entry in matched:
        reports = entry.get("reports", [])
        names = "|".join(r["name"] for r in reports)
        paths = "|".join(r["path"] for r in reports)
        rows.append(
            {
                "status": "matched",
                "master_id": entry.get("master_id"),
                "company": entry.get("primary_company") or entry.get("ent_name") or entry.get("ent_short_name"),
                "company_norm": entry["company_norm"],
                "year": entry.get("primary_year") or entry.get("year"),
                "key": entry["key_for_pdf"],
                "report_names": names,
                "report_paths": paths,
                "question": entry.get("question"),
            }
        )
    for entry in missing:
        rows.append(
            {
                "status": "missing",
                "master_id": entry.get("master_id"),
                "company": entry.get("primary_company") or entry.get("ent_name") or entry.get("ent_short_name"),
                "company_norm": entry["company_norm"],
                "year": entry.get("primary_year") or entry.get("year"),
                "key": entry["key_for_pdf"],
                "report_names": "",
                "report_paths": "",
                "question": entry.get("question"),
            }
        )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "master_id",
                "company",
                "company_norm",
                "year",
                "key",
                "report_names",
                "report_paths",
                "question",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global OUT_JSON, OUT_CSV
    parser = argparse.ArgumentParser(
        description="Map type=1 & 1co_1yr QAs to reports_list.csv using normalized company names"
    )
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER, help="输入 master JSONL（去重后的）")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_CSV, help="reports_list.csv 路径")
    parser.add_argument("--output-json", type=Path, default=OUT_JSON, help="输出映射 JSON 路径")
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV, help="输出映射 CSV 路径")
    args = parser.parse_args()

    OUT_JSON = args.output_json
    OUT_CSV = args.output_csv

    report_map = load_reports(args.reports)
    records = load_type1_1co1yr(args.master)
    matched, missing = map_reports(records, report_map)
    write_outputs(matched, missing)

    print(
        f"✅ 映射完成：总 {len(records)}，匹配 {len(matched)}，未匹配 {len(missing)}\n"
        f"- JSON: {OUT_JSON}\n- CSV: {OUT_CSV}"
    )


if __name__ == "__main__":
    main()
