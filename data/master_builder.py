"""Build FinGLM master table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data.qa_analyzer import determine_cross_dimension, extract_company_and_year

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "input" / "finglm-data _raw"

SOURCE_FILES = [
    ("pre", Path("pre-data/answer.json")),
    ("A", Path("A-data/A-list-answer.json")),
    ("B", Path("B-data/B-list-answer.json")),
    ("C", Path("C-data/C-list-answer.json")),
]


@dataclass
class MasterRecord:
    master_id: int
    source_dataset: str
    source_file: str
    source_item_id: int
    question: str
    answers: List[str]
    type: str
    prompt: Dict[str, Any]
    ent_name: str
    ent_short_name: str
    year: str
    key_word: str
    primary_company: str
    primary_year: str
    has_company: bool
    has_year: bool
    company_count: int
    year_count: int
    all_companies: List[str]
    all_years: List[str]
    cross_dimension: str
    company_year_key: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "master_id": self.master_id,
            "source_dataset": self.source_dataset,
            "source_file": self.source_file,
            "source_item_id": self.source_item_id,
            "question": self.question,
            "answers": self.answers,
            "type": self.type,
            "prompt": self.prompt,
            "ent_name": self.ent_name,
            "ent_short_name": self.ent_short_name,
            "year": self.year,
            "key_word": self.key_word,
            "primary_company": self.primary_company,
            "primary_year": self.primary_year,
            "has_company": self.has_company,
            "has_year": self.has_year,
            "company_count": self.company_count,
            "year_count": self.year_count,
            "all_companies": self.all_companies,
            "all_years": self.all_years,
            "cross_dimension": self.cross_dimension,
            "company_year_key": self.company_year_key,
        }


def _iter_raw_items(raw_dir: Path) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    for source_label, relative_path in SOURCE_FILES:
        file_path = raw_dir / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"未找到源文件: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                yield source_label, relative_path, item


def normalize_answers(raw_item: Dict[str, Any]) -> List[str]:
    answers = raw_item.get("answers", raw_item.get("answer", []))
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers.strip()]
    if isinstance(answers, list):
        return [str(ans).strip() for ans in answers if str(ans).strip()]
    return [str(answers).strip()]


def build_master_records(raw_dir: Path) -> Tuple[List[MasterRecord], Dict[str, Any]]:
    records: List[MasterRecord] = []
    stats = {
        "total": 0,
        "by_source": {},
        "type_distribution": {},
        "cross_dimension_distribution": {},
    }

    for idx, (source_label, rel_path, raw_item) in enumerate(_iter_raw_items(raw_dir)):
        prompt = raw_item.get("prompt", {}) or {}
        answers = normalize_answers(raw_item)
        q_type = str(raw_item.get("type", "") or "").strip()

        (
            primary_company,
            primary_year,
            has_company,
            has_year,
            company_count,
            year_count,
            all_companies,
            all_years,
        ) = extract_company_and_year(raw_item)

        cross_dim = determine_cross_dimension(has_company, has_year, company_count, year_count)
        ent_name = str(prompt.get("ent_name", "") or "").strip()
        ent_short = str(prompt.get("ent_short_name", "") or "").strip()
        year = str(prompt.get("year", "") or "").strip()
        key_word = str(prompt.get("key_word", "") or "").strip()
        company_year_key = f"{primary_company}#{primary_year}" if primary_company and primary_year else None

        record = MasterRecord(
            master_id=idx,
            source_dataset=source_label,
            source_file=str(rel_path),
            source_item_id=int(raw_item.get("id", idx)),
            question=str(raw_item.get("question", "")).strip(),
            answers=answers,
            type=q_type,
            prompt=prompt,
            ent_name=ent_name,
            ent_short_name=ent_short,
            year=year,
            key_word=key_word,
            primary_company=primary_company,
            primary_year=primary_year,
            has_company=has_company,
            has_year=has_year,
            company_count=company_count,
            year_count=year_count,
            all_companies=all_companies,
            all_years=all_years,
            cross_dimension=cross_dim,
            company_year_key=company_year_key,
        )
        records.append(record)

        stats["total"] += 1
        stats["by_source"].setdefault(source_label, 0)
        stats["by_source"][source_label] += 1
        stats["type_distribution"].setdefault(q_type or "unknown", 0)
        stats["type_distribution"][q_type or "unknown"] += 1
        stats["cross_dimension_distribution"].setdefault(cross_dim, 0)
        stats["cross_dimension_distribution"][cross_dim] += 1

    return records, stats


def write_jsonl(records: List[MasterRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_stats(stats: Dict[str, Any], path: Path) -> None:
    stats_to_save = {"generated_at": datetime.now(timezone.utc).isoformat(), **stats}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats_to_save, f, ensure_ascii=False, indent=2)

