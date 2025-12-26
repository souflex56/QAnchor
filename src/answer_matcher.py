"""Type1 关键值抽取与 chunk 匹配工具（Step 4a）。

提供两个核心方法：
- `extract_key_values(record)`: 从标准答案记录中提取关键值（优先 prom_answer，fallback 白名单字段）。
- `match_chunk_to_answer(chunk_text, key_values)`: 归一化匹配关键值，返回匹配结果与置信度。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# 可扩展白名单：Stage0 实测后可补充
FIELD_WHITELIST: List[str] = [
    "电子信箱",
    "法定代表人",
    "注册地址",
    "证券代码",
    "股票代码",
    "股票简称",
    "公司网址",
    "联系电话",
    "办公地址",
    "邮政编码",
    "传真",
]


@dataclass
class KeyValue:
    value: str
    source: str  # prom_answer 或字段名


@dataclass
class MatchResult:
    is_match: bool
    match_type: str  # exact | fuzzy | regex | none
    matched_keys: List[str]
    match_evidence: str
    confidence: float
    unit_inferred: Optional[str] = None


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    norm = unicodedata.normalize("NFKC", str(text))
    norm = norm.lower()
    norm = re.sub(r"\s+", "", norm)
    return norm


def _strip_thousands(text: str) -> str:
    return text.replace(",", "").replace("，", "")


def _parse_number_with_unit(raw: str) -> Dict[str, Any]:
    """解析数字+单位，返回基础值（以元或无单位数值为基准）。

    支持：元/万元/亿元/万/亿/%；不解析失败时返回空。
    """

    if not raw:
        return {"value": None, "unit": None, "is_percent": False}
    text = _normalize_text(raw)
    text = _strip_thousands(text)
    is_percent = "%" in text

    m = re.search(r"([-+]?\d(?:[\d]*)(?:\.\d+)?)", text)
    if not m:
        return {"value": None, "unit": None, "is_percent": is_percent}

    num_str = m.group(1)
    try:
        value = float(num_str)
    except Exception:
        return {"value": None, "unit": None, "is_percent": is_percent}

    unit = None
    if "亿元" in raw or "亿" in raw:
        unit = "亿"
        value *= 1e8
    elif "万元" in raw or "万" in raw:
        unit = "万"
        value *= 1e4
    elif "元" in raw:
        unit = "元"

    if is_percent:
        unit = "%"
        value = value / 100.0

    return {"value": value, "unit": unit, "is_percent": is_percent}


def _format_number(num: float) -> str:
    if num == 0:
        return "0"
    if abs(num) >= 1:
        return f"{num:.6g}"
    return f"{num:.8g}"


def _value_variants(value: str) -> Dict[str, Any]:
    """生成用于匹配的多种归一化形态。"""

    variants: List[str] = []
    unit_hint: Optional[str] = None
    if not value:
        return {"variants": variants, "unit_hint": unit_hint}

    raw = str(value).strip()
    norm_basic = _normalize_text(raw)
    if norm_basic:
        variants.append(norm_basic)

    raw_clean = _normalize_text(_strip_thousands(raw))
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", raw_clean)
    if m:
        variants.append(m.group(1))

    parsed = _parse_number_with_unit(raw)
    if parsed.get("value") is not None:
        unit_hint = parsed.get("unit")
        base_val = parsed["value"]
        variants.append(_normalize_text(_format_number(base_val)))
        # 生成不同量级的数字字符串，防止单位差异错失匹配
        variants.append(_normalize_text(_format_number(base_val / 1e4)))  # 万元级
        variants.append(_normalize_text(_format_number(base_val / 1e8)))  # 亿元级

    # 去重，保持顺序
    dedup: List[str] = []
    seen = set()
    for v in variants:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return {"variants": dedup, "unit_hint": unit_hint}


def _has_stable_key(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.IGNORECASE):
        return True
    if re.search(r"(https?://|www\.)", text, re.IGNORECASE):
        return True
    if re.search(r"\d", text):
        return True
    return len(text) <= 64


def _build_norm_index_map(text: str) -> Tuple[str, List[int]]:
    """构建归一化文本并保留原文索引映射。"""
    norm_chars: List[str] = []
    index_map: List[int] = []
    for idx, ch in enumerate(text):
        norm = unicodedata.normalize("NFKC", ch)
        for n_ch in norm:
            if n_ch.isspace():
                continue
            if n_ch in {",", "，"}:
                continue
            n_ch = n_ch.lower()
            norm_chars.append(n_ch)
            index_map.append(idx)
    return "".join(norm_chars), index_map


def _find_match_span(norm_text: str, index_map: List[int], pattern: str) -> Optional[Tuple[int, int]]:
    if not pattern:
        return None
    start = norm_text.find(pattern)
    if start < 0:
        return None
    end = start + len(pattern) - 1
    if end >= len(index_map):
        return None
    return index_map[start], index_map[end]


def _extract_context(text: str, start: int, end: int, window: int = 120) -> str:
    if not text:
        return ""
    s = max(0, start - window)
    e = min(len(text), end + window + 1)
    return text[s:e]


def is_type1(record: Dict[str, Any]) -> bool:
    """判断是否 Type1（支持字符串/数字）。"""

    t = record.get("type")
    if t is None:
        t = (record.get("prompt") or {}).get("type")
    if t is None:
        return False
    try:
        t_str = str(t).strip().lower()
    except Exception:
        return False
    return t_str == "1"


def extract_key_values(answer_record: Dict[str, Any]) -> List[KeyValue]:
    """从答案记录中提取关键值（Type1-only）。"""

    kvs: List[KeyValue] = []
    prompt = answer_record.get("prompt") or {}

    prom_answer = prompt.get("prom_answer")
    has_stable_prom = False
    if prom_answer:
        val = str(prom_answer).strip()
        if val and _has_stable_key(val):
            kvs.append(KeyValue(value=val, source="prom_answer"))
            has_stable_prom = True

    if not has_stable_prom:
        for field in FIELD_WHITELIST:
            val = prompt.get(field)
            if not val:
                continue
            val_str = str(val).strip()
            if not val_str:
                continue
            kvs.append(KeyValue(value=val_str, source=field))

    # 去重（按 value 与 source 组合）
    dedup: List[KeyValue] = []
    seen = set()
    for kv in kvs:
        key = (kv.value, kv.source)
        if key in seen:
            continue
        dedup.append(kv)
        seen.add(key)
    return dedup


def _regex_email(text: str) -> bool:
    return bool(re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text, re.IGNORECASE))


def match_chunk_to_answer(chunk_text: str, key_values: Sequence[KeyValue]) -> MatchResult:
    """归一化匹配 chunk 与关键值，返回匹配结果。"""

    chunk_text = str(chunk_text or "")
    chunk_norm, index_map = _build_norm_index_map(chunk_text)
    if not chunk_norm or not key_values:
        return MatchResult(False, "none", [], "", 0.0)

    matched_keys: List[str] = []
    best_conf = 0.0
    best_type = "none"
    unit_hint: Optional[str] = None
    best_span: Optional[Tuple[int, int]] = None

    for kv in key_values:
        variants_info = _value_variants(kv.value)
        variants = variants_info.get("variants", [])
        if variants_info.get("unit_hint"):
            unit_hint = unit_hint or variants_info.get("unit_hint")

        # 宽松字符串包含
        for idx, v in enumerate(variants):
            span = _find_match_span(chunk_norm, index_map, v)
            if v and span:
                matched_keys.append(kv.source)
                conf = 1.0 if idx == 0 else 0.8
                best_conf = max(best_conf, conf)
                best_type = "exact"
                if best_span is None:
                    best_span = span
                break

        # 额外 regex 针对邮箱类字段
        if not matched_keys and (kv.source in ("电子信箱", "公司网址") or "@" in kv.value):
            email_match = re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", chunk_text, re.IGNORECASE)
            if email_match:
                matched_keys.append(kv.source)
                best_conf = max(best_conf, 0.7)
                best_type = "regex"
                if best_span is None:
                    best_span = (email_match.start(), email_match.end() - 1)

    is_match = len(matched_keys) > 0
    if is_match and best_span:
        evidence = _extract_context(chunk_text, best_span[0], best_span[1])
    else:
        evidence = chunk_text[:200] if chunk_text else ""
    if unit_hint and is_match:
        evidence = f"{evidence} [unit_inferred={unit_hint}]"
    confidence = min(max(best_conf, 0.0), 1.0)

    return MatchResult(is_match, best_type if is_match else "none", matched_keys, evidence, confidence, unit_hint)


__all__ = [
    "KeyValue",
    "MatchResult",
    "FIELD_WHITELIST",
    "is_type1",
    "extract_key_values",
    "match_chunk_to_answer",
]
