"""Analytics helpers for FinGLM QA data."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


YEAR_PATTERN = re.compile(r"(201[6-9]|202[0-4])")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_company_and_year(item: Dict[str, Any]) -> Tuple[str, str, bool, bool, int, int, List[str], List[str]]:
    """Extract company/year info from a raw QA item."""
    prompt = item.get("prompt", {}) or {}
    question = item.get("question", "") or ""

    company_name = prompt.get("ent_name", "") or prompt.get("ent_short_name", "")
    year = prompt.get("year", "")

    # years from question text
    year_matches = YEAR_PATTERN.findall(question)
    all_years = list({*year_matches}) if year_matches else ([year] if year else [])

    multi_company_keywords = [
        "多家公司",
        "哪家公司",
        "哪个公司",
        "上市公司",
        "第一",
        "第二",
        "第三",
        "第四",
        "第五",
        "最高",
        "最低",
        "排名",
        "对比",
    ]
    multi_year_keywords = ["同比", "环比", "增长率", "变化率", "对比"]

    is_multi_company = any(keyword in question for keyword in multi_company_keywords)
    has_comparison = any(keyword in question for keyword in multi_year_keywords)

    all_companies: List[str] = []
    if company_name and company_name != "unknown_company":
        all_companies = [company_name]
    elif any(k in question for k in ["公司", "股份", "企业", "集团"]):
        all_companies = ["multiple_companies"] if is_multi_company else ["unknown_company"]

    company_count = len(all_companies) if all_companies else 0
    year_count = len(all_years) if all_years else 0
    if has_comparison and year_count == 1:
        year_count = 2
    if is_multi_company:
        company_count = max(company_count, 2)

    has_company = company_count > 0
    has_year = year_count > 0
    main_company = all_companies[0] if all_companies else ""
    main_year = all_years[0] if all_years else ""

    return main_company, main_year, has_company, has_year, company_count, year_count, all_companies, all_years


def _render_type_markdown(report: Dict[str, Any], cleaning_summary: Dict[str, Any] | None = None) -> str:
    lines: List[str] = []
    total = sum(report["summary"]["type_distribution"].values()) or 1
    lines.append("# FinGLM Data 类型分析报告\n")
    lines.append("## 概述")
    lines.append(f"- 总问题数: {total}")
    lines.append(f"- 问题类型数: {report['summary']['total_types']}")
    if cleaning_summary:
        before = cleaning_summary.get("before_count")
        after = cleaning_summary.get("after_count")
        if before is not None and after is not None:
            lines.append(f"- 清洗前: {before} 条，清洗后: {after} 条")
    lines.append("")
    lines.append("### 类型分布")
    lines.append("| 类型 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for qtype, count in sorted(report["summary"]["type_distribution"].items(), key=lambda kv: kv[1], reverse=True):
        pct = count / total * 100
        lines.append(f"| {qtype} | {count} | {pct:.2f}% |")

    lines.append("\n## 各类型详细分析")
    for qtype in sorted(report["type_analysis"].keys()):
        analysis = report["type_analysis"][qtype]
        lines.append(f"### 类型 {qtype}")
        lines.append(f"- 数量: {analysis['count']} ({analysis['percentage']})")
        lines.append(f"- 描述: {analysis['description']}")
        if analysis["prompt_keys"]:
            lines.append(f"- Prompt 字段: {', '.join(analysis['prompt_keys'])}")
        patterns = analysis.get("patterns", {})
        if patterns:
            lines.append(f"- 平均长度: {patterns.get('avg_length', 0):.1f}")
            lines.append(f"- 包含公司比例: {patterns.get('has_company_ratio', 0) * 100:.1f}%")
            lines.append(f"- 包含年份比例: {patterns.get('has_year_ratio', 0) * 100:.1f}%")
            lines.append(f"- 计算相关比例: {patterns.get('has_calculation_ratio', 0) * 100:.1f}%")
        lines.append("#### 示例")
        for i, ex in enumerate(analysis["examples"][:5], 1):
            lines.append(f"{i}. (ID: {ex.get('id')}) {ex.get('question')}")
            if ex.get("prompt_keys"):
                lines.append(f"   - Prompt 字段: {', '.join(ex.get('prompt_keys'))}")
        lines.append("")

    lines.append("## 文件级别统计")
    for fname, breakdown in report["file_breakdown"].items():
        lines.append(f"- {fname}: {breakdown['total_count']} 条，类型: {', '.join(sorted(breakdown['types_present']))}")

    lines.append(f"\n*报告生成时间: {datetime.now().isoformat()}*")
    return "\n".join(lines)


def _infer_type_meaning(qtype: str, pattern: Dict[str, Any], prompt_keys: Iterable[str]) -> str:
    descriptions = {
        "1": "简单事实查询 - 直接查询单个具体数值或事实信息，通常有明确的答案",
        "1-2": "多值查询 - 查询多个相关的具体数值或事实信息",
        "2-1": "计算类问题 - 需要根据公式计算得出结果，通常涉及财务比率或增长率",
        "2-2": "对比类问题 - 需要对比两个时间点或两个实体的数据",
        "3-1": "分析类问题 - 需要基于年报数据进行综合分析、解释或讨论",
        "3-2": "概念性问题 - 询问财务、会计或业务概念的定义、含义或影响，不依赖特定公司数据",
    }
    base = descriptions.get(qtype, f"类型 {qtype} - 需要进一步分析")
    features: List[str] = []
    if pattern.get("has_company_ratio", 0) > 0.5:
        features.append("通常涉及特定公司")
    if pattern.get("has_year_ratio", 0) > 0.5:
        features.append("通常涉及特定年份")
    if pattern.get("has_calculation_ratio", 0) > 0.5:
        features.append("通常涉及计算")
    joined = ", ".join(sorted(prompt_keys))
    if "公式" in joined or "formula" in joined:
        features.append("包含计算公式")
    if not joined:
        features.append("不依赖额外prompt字段")
    if features:
        return f"{base} ({', '.join(features)})"
    return base


def _analyze_question_patterns(questions_by_type: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    patterns: Dict[str, Dict[str, Any]] = {}
    for qtype, questions in questions_by_type.items():
        lengths: List[int] = []
        has_company: List[bool] = []
        has_year: List[bool] = []
        has_calc: List[bool] = []
        words: List[str] = []
        for q in questions:
            text = q.get("question", "") or ""
            lengths.append(len(text))
            words.extend(text.split())
            has_company.append(any(k in text for k in ["公司", "企业", "股份", "有限"]))
            has_year.append(any(k in text for k in ["2019", "2020", "2021", "2022", "年"]))
            has_calc.append(any(k in text for k in ["计算", "多少", "比率", "增长率", "公式", "保留"]))
        patterns[qtype] = {
            "avg_length": (sum(lengths) / len(lengths)) if lengths else 0,
            "has_company_ratio": (sum(has_company) / len(has_company)) if has_company else 0,
            "has_year_ratio": (sum(has_year) / len(has_year)) if has_year else 0,
            "has_calculation_ratio": (sum(has_calc) / len(has_calc)) if has_calc else 0,
            "common_words": Counter(words).most_common(10),
        }
    return patterns


def analyze_types(input_path: Path) -> Dict[str, Any]:
    """Analyze type distribution from a master/cleaned JSONL."""
    type_counter: Counter[str] = Counter()
    type_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    type_prompts: Dict[str, set] = defaultdict(set)
    questions_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    file_breakdown: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total_count": 0, "type_distribution": Counter(), "types_present": set()})

    for item in _iter_jsonl(input_path):
        qtype = str(item.get("type", "unknown") or "unknown").strip()
        question = item.get("question", "") or ""
        prompt = item.get("prompt", {}) or {}
        source_file = item.get("source_file") or item.get("source_dataset") or "unknown"

        type_counter[qtype] += 1
        fb = file_breakdown[source_file]
        fb["total_count"] += 1
        fb["type_distribution"][qtype] += 1
        fb["types_present"].add(qtype)

        if len(type_examples[qtype]) < 10:
            type_examples[qtype].append(
                {
                    "id": item.get("master_id", item.get("id")),
                    "question": question,
                    "has_prompt": bool(prompt),
                    "prompt_keys": list(prompt.keys()) if isinstance(prompt, dict) else [],
                }
            )

        if isinstance(prompt, dict):
            for key in prompt.keys():
                if not isinstance(key, (int, float)) and not (isinstance(key, str) and len(key) > 20 and "股份有限公司" in key):
                    type_prompts[qtype].add(key)

        questions_by_type[qtype].append({"question": question, "prompt": prompt})

    patterns = _analyze_question_patterns(questions_by_type)
    total = sum(type_counter.values()) or 1

    report: Dict[str, Any] = {
        "summary": {
            "total_types": len(type_counter),
            "type_distribution": dict(type_counter),
            "type_percentages": {k: f"{(v / total * 100):.2f}%" for k, v in type_counter.items()},
        },
        "type_analysis": {},
        "file_breakdown": {},
    }

    for source, stats in file_breakdown.items():
        report["file_breakdown"][source] = {
            "total_count": stats["total_count"],
            "type_distribution": dict(stats["type_distribution"]),
            "types_present": list(stats["types_present"]),
        }

    for qtype in sorted(type_counter.keys()):
        prompt_keys = sorted(type_prompts[qtype])
        report["type_analysis"][qtype] = {
            "count": type_counter[qtype],
            "percentage": f"{(type_counter[qtype] / total * 100):.2f}%",
            "description": _infer_type_meaning(qtype, patterns.get(qtype, {}), prompt_keys),
            "examples": type_examples[qtype][:10],
            "prompt_keys": prompt_keys,
            "patterns": patterns.get(qtype, {}),
        }

    return report


def write_type_reports(report: Dict[str, Any], output_dir: Path, cleaning_summary: Dict[str, Any] | None = None) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "finglm_data_type_analysis.json"
    md_path = output_dir / "finglm_data_type_analysis.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path.write_text(_render_type_markdown(report, cleaning_summary), encoding="utf-8")
    return json_path, md_path


def determine_cross_dimension(has_company: bool, has_year: bool, company_count: int, year_count: int) -> str:
    if not has_company and not has_year:
        return "generic"
    if company_count <= 1 and year_count <= 1:
        return "1co_1yr"
    if company_count <= 1 and year_count > 1:
        return "1co_multi_yr"
    if company_count > 1 and year_count <= 1:
        return "multi_co_1yr"
    return "multi_co_multi_yr"


def analyze_dimensions(input_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze company/year dimensions from a master/cleaned JSONL."""
    by_year: Counter[str] = Counter()
    by_company: Counter[str] = Counter()

    all_qa: List[Dict[str, Any]] = []
    cross_buckets = {
        "single_company_single_year": [],
        "single_company_multi_year": [],
        "multi_company_single_year": [],
        "multi_company_multi_year": [],
        "generic": [],
    }

    for item in _iter_jsonl(input_path):
        company = item.get("primary_company") or ""
        year = item.get("primary_year") or ""
        all_companies = item.get("all_companies") or []
        all_years = item.get("all_years") or []
        company_count = item.get("company_count") or (len(all_companies) if all_companies else 0)
        year_count = item.get("year_count") or (len(all_years) if all_years else 0)
        has_company = item.get("has_company", bool(company_count))
        has_year = item.get("has_year", bool(year_count))

        if not company and not year:
            (
                company,
                year,
                has_company,
                has_year,
                company_count,
                year_count,
                all_companies,
                all_years,
            ) = extract_company_and_year(item)

        cross = item.get("cross_dimension") or determine_cross_dimension(has_company, has_year, company_count, year_count)

        qa_info = {
            "id": item.get("master_id", item.get("id")),
            "question": item.get("question"),
            "type": item.get("type"),
            "company": company,
            "year": year,
            "has_company": has_company,
            "has_year": has_year,
            "company_count": company_count,
            "year_count": year_count,
            "all_companies": all_companies,
            "all_years": all_years,
        }
        all_qa.append(qa_info)

        if has_year and year:
            by_year[year] += 1
        if has_company and company:
            by_company[company] += 1

        if cross == "generic":
            cross_buckets["generic"].append(qa_info)
        elif company_count <= 1 and year_count <= 1:
            cross_buckets["single_company_single_year"].append(qa_info)
        elif company_count <= 1 and year_count > 1:
            cross_buckets["single_company_multi_year"].append(qa_info)
        elif company_count > 1 and year_count <= 1:
            cross_buckets["multi_company_single_year"].append(qa_info)
        else:
            cross_buckets["multi_company_multi_year"].append(qa_info)

    total = len(all_qa) or 1
    cross_summary = {
        key: {
            "count": len(items),
            "percentage": len(items) / total * 100,
            "examples": items[:10],
        }
        for key, items in cross_buckets.items()
    }

    stats = {
        "all_qa": all_qa,
        "by_year": by_year,
        "by_company": by_company,
        "generic_qa": cross_buckets["generic"],
        "cross_summary": cross_summary,
    }
    return stats, cross_summary


def _render_dimension_markdown(stats: Dict[str, Any], cross: Dict[str, Any], cleaning_summary: Dict[str, Any] | None = None) -> str:
    total = len(stats["all_qa"]) or 1
    lines: List[str] = []
    lines.append("# FinGLM 问答对多维度统计分析报告\n")
    lines.append("## 概述")
    lines.append(f"- 总问答对数: {total}")
    if cleaning_summary:
        before = cleaning_summary.get("before_count")
        after = cleaning_summary.get("after_count")
        if before is not None and after is not None:
            lines.append(f"- 清洗前: {before} 条，清洗后: {after} 条")
    lines.append("")

    lines.append("## 1. 时间维度：按年份分类")
    lines.append("| 年份 | 问答对数量 | 占比 |")
    lines.append("|------|------------|------|")
    total_with_year = sum(stats["by_year"].values()) or 1
    for year, count in sorted(stats["by_year"].items()):
        pct = count / total_with_year * 100
        lines.append(f"| {year} | {count} | {pct:.2f}% |")
    lines.append(f"- 涉及年份数: {len(stats['by_year'])}")
    lines.append(f"- 无年份信息: {total - total_with_year}\n")

    lines.append("## 2. 公司维度：按公司分类 (Top 20)")
    lines.append("| 排名 | 公司名称 | 问答对数量 | 占比 |")
    lines.append("|------|----------|------------|------|")
    total_with_company = sum(stats["by_company"].values()) or 1
    for rank, (company, count) in enumerate(stats["by_company"].most_common(20), 1):
        pct = count / total_with_company * 100
        lines.append(f"| {rank} | {company[:50]} | {count} | {pct:.2f}% |")
    lines.append(f"- 涉及公司数: {len(stats['by_company'])}")
    lines.append(f"- 无公司信息: {total - total_with_company}\n")

    lines.append("## 3. 交叉维度分析（基于单个问答对涉及的公司/年份数量）")
    titles = {
        "single_company_single_year": "1公司+1年份",
        "single_company_multi_year": "1公司+多年份",
        "multi_company_single_year": "多公司+1年份",
        "multi_company_multi_year": "多公司+多年份",
        "generic": "通用类型",
    }
    for key in ["single_company_single_year", "single_company_multi_year", "multi_company_single_year", "multi_company_multi_year", "generic"]:
        entry = cross[key]
        lines.append(f"### {titles[key]}")
        lines.append(f"- 数量: {entry['count']} ({entry['percentage']:.2f}%)")
        lines.append("- 示例:")
        for i, ex in enumerate(entry.get("examples", [])[:10], 1):
            years_str = ", ".join(ex.get("all_years", []) or ([ex.get("year")] if ex.get("year") else []))
            lines.append(f"  {i}. (ID: {ex.get('id')}, 类型: {ex.get('type')}) 公司: {ex.get('company')} 年份: {years_str or ex.get('year')}")
            lines.append(f"     问题: {ex.get('question')}")
        lines.append("")

    lines.append("## 4. 通用类型问答对 (无公司&年份)")
    generic = cross["generic"]
    lines.append(f"- 数量: {generic['count']} ({generic['percentage']:.2f}%)")
    for i, qa in enumerate(generic.get("examples", [])[:20], 1):
        lines.append(f"{i}. (ID: {qa.get('id')}, 类型: {qa.get('type')}) {qa.get('question')}")
    lines.append("")

    lines.append("## 5. 总结统计")
    lines.append("| 维度 | 数量 | 占比 | 说明 |")
    lines.append("|------|------|------|------|")
    lines.append(f"| 总问答对数 | {total} | 100.00% | 所有记录 |")
    lines.append(f"| 1公司+1年份 | {cross['single_company_single_year']['count']} | {cross['single_company_single_year']['percentage']:.2f}% | 单公司单年份 |")
    lines.append(f"| 1公司+多年份 | {cross['single_company_multi_year']['count']} | {cross['single_company_multi_year']['percentage']:.2f}% | 单公司多年份 |")
    lines.append(f"| 多公司+1年份 | {cross['multi_company_single_year']['count']} | {cross['multi_company_single_year']['percentage']:.2f}% | 多公司单年份 |")
    lines.append(f"| 多公司+多年份 | {cross['multi_company_multi_year']['count']} | {cross['multi_company_multi_year']['percentage']:.2f}% | 多公司多年份 |")
    lines.append(f"| 通用类型 | {generic['count']} | {generic['percentage']:.2f}% | 无公司/年份 |")
    lines.append(f"\n*报告生成时间: {datetime.now().isoformat()}*")
    return "\n".join(lines)


def write_dimension_reports(stats: Dict[str, Any], cross: Dict[str, Any], output_dir: Path, cleaning_summary: Dict[str, Any] | None = None) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "finglm_qa_dimension_analysis.json"
    md_path = output_dir / "finglm_qa_dimension_analysis.md"

    json_report = {
        "summary": {
            "total_qa": len(stats["all_qa"]),
            "with_year": sum(stats["by_year"].values()),
            "with_company": sum(stats["by_company"].values()),
            "generic": cross["generic"]["count"],
        },
        "by_year": dict(stats["by_year"]),
        "by_company": dict(stats["by_company"]),
        "cross_analysis": {
            k: {"count": v["count"], "percentage": v["percentage"]} for k, v in cross.items()
        },
        "top_companies": dict(stats["by_company"].most_common(50)),
        "top_years": dict(stats["by_year"].most_common(10)),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    md_path.write_text(_render_dimension_markdown(stats, cross, cleaning_summary), encoding="utf-8")
    return json_path, md_path

