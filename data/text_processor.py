"""Cleaning and deduplication utilities for FinGLM master data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PLACEHOLDER_ANSWERS = {
    "未查询到2019年光云科技的相关信息，无法回答该问题。",
    "未查询到2020年长远锂科的相关信息，无法回答该问题。",
    "根据已有数据查询可知，未查询到该公司该年份的年报信息，无法回答您的问题。",
    "未查询到该公司该年份的年报信息，无法回答您的问题。",
    "非常抱歉，您的问题超出了我的回答范围，暂无法回答。",
    "非常抱歉，您的问题超出了我的回答范围，暂无法回答",  # 无句号变体
}

PLACEHOLDER_PATTERNS = (
    re.compile(r"^未查询到\d{4}年.+的相关信息，无法回答该问题。?$"),
    re.compile(r"^未查询到.+年报信息，无法回答您的问题。?$"),
)

def normalize_question(text: str) -> str:
    """Normalize question text for deduplication."""
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_placeholder_answer(answer: str) -> bool:
    normalized = answer.strip()
    if normalized in PLACEHOLDER_ANSWERS:
        return True
    return any(pat.match(normalized) for pat in PLACEHOLDER_PATTERNS)


def load_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def clean_and_dedup(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "before_count": 0,
        "after_filter": 0,
        "after_dedup": 0,
        "removed_counts": {
            "empty_question": 0,
            "empty_answers": 0,
            "placeholder_answer": 0,
            "dedup": 0,
        },
        "removed_samples": {
            "empty_question": [],
            "empty_answers": [],
            "placeholder_answer": [],
            "dedup": [],
        },
    }

    # First pass: filter invalid
    filtered: List[Dict[str, Any]] = []
    for item in records:
        stats["before_count"] += 1
        question = str(item.get("question", "")).strip()
        answers_raw = item.get("answers", [])
        answers: List[str]
        if isinstance(answers_raw, list):
            answers = [str(a).strip() for a in answers_raw if str(a).strip()]
        elif isinstance(answers_raw, str):
            answers = [answers_raw.strip()]
        else:
            answers = [str(answers_raw).strip()] if answers_raw else []

        if not question:
            stats["removed_counts"]["empty_question"] += 1
            if len(stats["removed_samples"]["empty_question"]) < 5:
                stats["removed_samples"]["empty_question"].append({"question": question, "id": item.get("master_id", item.get("id"))})
            continue

        if not answers:
            stats["removed_counts"]["empty_answers"] += 1
            if len(stats["removed_samples"]["empty_answers"]) < 5:
                stats["removed_samples"]["empty_answers"].append({"question": question, "id": item.get("master_id", item.get("id"))})
            continue

        valid_answers = [ans for ans in answers if not is_placeholder_answer(ans)]
        if not valid_answers:
            stats["removed_counts"]["placeholder_answer"] += 1
            if len(stats["removed_samples"]["placeholder_answer"]) < 5:
                stats["removed_samples"]["placeholder_answer"].append(
                    {"question": question, "answers": answers, "id": item.get("master_id", item.get("id"))}
                )
            continue

        item["answers"] = valid_answers
        filtered.append(item)

    stats["after_filter"] = len(filtered)

    # Second pass: dedup by (company, year, normalized_question)
    seen_keys = set()
    for item in filtered:
        primary_company = item.get("primary_company", "") or ""
        primary_year = item.get("primary_year", "") or ""
        norm_question = normalize_question(str(item.get("question", "")))
        key = (primary_company, primary_year, norm_question)
        if key in seen_keys:
            stats["removed_counts"]["dedup"] += 1
            if len(stats["removed_samples"]["dedup"]) < 5:
                stats["removed_samples"]["dedup"].append(
                    {
                        "question": item.get("question"),
                        "primary_company": primary_company,
                        "primary_year": primary_year,
                        "id": item.get("master_id", item.get("id")),
                    }
                )
            continue
        seen_keys.add(key)
        cleaned.append(item)

    stats["after_dedup"] = len(cleaned)
    return cleaned, stats


def save_clean_report(stats: Dict[str, Any], report_dir: Path) -> Tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "clean_dedup_report.json"
    md_path = report_dir / "clean_dedup_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# 清洗与去重报告\n")
    lines.append(f"- 清洗前: {stats.get('before_count', 0)}")
    lines.append(f"- 过滤后: {stats.get('after_filter', 0)}")
    lines.append(f"- 去重后: {stats.get('after_dedup', 0)}\n")

    lines.append("## 删除原因统计")
    for reason, count in stats.get("removed_counts", {}).items():
        lines.append(f"- {reason}: {count}")
    lines.append("")

    lines.append("## 被删除记录示例（每类最多5条）")
    for reason, samples in stats.get("removed_samples", {}).items():
        lines.append(f"### {reason}")
        if not samples:
            lines.append("无示例\n")
            continue
        for sample in samples:
            line = f"- ID: {sample.get('id')} | Q: {sample.get('question')}"
            if sample.get("primary_company") or sample.get("primary_year"):
                line += f" | 公司: {sample.get('primary_company','')} 年份: {sample.get('primary_year','')}"
            lines.append(line)
            if sample.get("answers"):
                lines.append(f"  - answers: {sample.get('answers')}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
