#!/usr/bin/env python3
"""
LLM-based annotation for Gold Eval candidates.

Reads a gold_eval_*_template.jsonl file and fills candidate labels:
  - evidence
  - related
  - irrelevant
Optionally adds short notes when needed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

LABELS = {"evidence", "related", "irrelevant"}

SYSTEM_PROMPT = (
    "You are a strict annotation assistant for retrieval evaluation. "
    "Label each candidate chunk for how well it answers the query."
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers=headers or {}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    return json.loads(raw)


class LLMClient:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, temperature: float, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        resp = http_post_json(url, payload, timeout=self.timeout)
        if isinstance(resp, dict):
            if "message" in resp and isinstance(resp["message"], dict):
                return resp["message"].get("content", "")
            if "response" in resp:
                return resp.get("response", "")
        raise RuntimeError(f"Unexpected Ollama response keys: {list(resp) if isinstance(resp, dict) else type(resp)}")


class SiliconFlowClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = http_post_json(url, payload, headers=headers, timeout=self.timeout)
        choices = resp.get("choices", []) if isinstance(resp, dict) else []
        if not choices:
            raise RuntimeError(f"Unexpected SiliconFlow response: {resp}")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return message.get("content", "")


def candidate_key(candidate: Dict[str, Any], idx: int) -> str:
    chunk_id = candidate.get("chunk_id")
    rank = candidate.get("rank")
    if chunk_id is not None:
        return f"{chunk_id}|{rank}"
    if rank is not None:
        return f"rank:{rank}"
    return f"idx:{idx}"


def build_existing_label_map(records: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Dict[str, str]]]:
    cache: Dict[Any, Dict[str, Dict[str, str]]] = {}
    for rec in records:
        qid = rec.get("query_id")
        if qid is None:
            continue
        cand_map: Dict[str, Dict[str, str]] = {}
        for idx, cand in enumerate(rec.get("candidates", []) or []):
            label = normalize_label(cand.get("label"))
            if label in LABELS:
                cand_map[candidate_key(cand, idx)] = {
                    "label": label,
                    "notes": cand.get("notes") or "",
                }
        if cand_map:
            cache[qid] = cand_map
    return cache


def normalize_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, str):
        value = label.strip().lower()
        if value.startswith("evid"):
            return "evidence"
        if value.startswith("rel"):
            return "related"
        if value.startswith("irr"):
            return "irrelevant"
        if value in LABELS:
            return value
    return None


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_payload(text: str) -> Any:
    text = strip_code_fence(text)
    if text.lower().startswith("json"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else ""
    for candidate in (text, extract_between(text, "[", "]"), extract_between(text, "{", "}")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("Failed to parse JSON from model response")


def extract_between(text: str, start_char: str, end_char: str) -> Optional[str]:
    start = text.find(start_char)
    end = text.rfind(end_char)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def normalize_results(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict):
        if "items" in parsed:
            parsed = parsed["items"]
        elif "results" in parsed:
            parsed = parsed["results"]
    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON is not a list")
    results: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Result item is not an object")
        idx = item.get("index")
        if isinstance(idx, str) and idx.strip().isdigit():
            idx = int(idx.strip())
        if not isinstance(idx, int):
            raise ValueError("Result item missing integer index")
        label = normalize_label(item.get("label"))
        if label not in LABELS:
            raise ValueError(f"Invalid label: {item.get('label')}")
        notes = item.get("notes") or ""
        results.append({"index": idx, "label": label, "notes": str(notes)})
    return results


def build_prompt(
    *,
    query: str,
    answers: List[str],
    candidates: List[Dict[str, Any]],
    indices: List[int],
    max_text_chars: int,
) -> str:
    lines: List[str] = []
    lines.append("Query:")
    lines.append(query)
    lines.append("")
    lines.append("Reference answers (paraphrases are possible):")
    for ans in answers:
        lines.append(f"- {ans}")
    lines.append("")
    lines.append("Label definitions:")
    lines.append("evidence: directly answers the query with required facts or numbers.")
    lines.append("related: same topic/entity/year but does not fully answer.")
    lines.append("irrelevant: not about the query.")
    lines.append("")
    lines.append("Return ONLY a JSON array. Each item: {\"index\": <int>, \"label\": \"evidence|related|irrelevant\", \"notes\": \"\"}.")
    lines.append("Use the same index shown with each candidate. Keep notes empty unless clarification is needed.")
    lines.append("")
    for idx in indices:
        cand = candidates[idx]
        lines.append(f"Candidate {idx}:")
        meta = {
            "chunk_id": cand.get("chunk_id"),
            "rank": cand.get("rank"),
            "page_numbers": cand.get("page_numbers"),
            "section_path": cand.get("section_path"),
        }
        lines.append(f"meta: {json.dumps(meta, ensure_ascii=False)}")
        text = str(cand.get("text") or "").strip()
        if max_text_chars > 0 and len(text) > max_text_chars:
            text = text[:max_text_chars] + " ...[truncated]"
        lines.append("text:")
        lines.append(text if text else "(empty)")
        lines.append("")
    return "\n".join(lines)


def label_batch(
    client: LLMClient,
    *,
    query: str,
    answers: List[str],
    candidates: List[Dict[str, Any]],
    indices: List[int],
    max_text_chars: int,
    max_retries: int,
    delay: float,
) -> Dict[int, Dict[str, str]]:
    prompt = build_prompt(query=query, answers=answers, candidates=candidates, indices=indices, max_text_chars=max_text_chars)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response_text = client.chat(messages)
            parsed = extract_json_payload(response_text)
            results = normalize_results(parsed)
            result_map = {item["index"]: item for item in results}
            missing = [idx for idx in indices if idx not in result_map]
            if missing:
                raise ValueError(f"Missing indices in response: {missing}")
            if delay > 0:
                time.sleep(delay)
            return result_map
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.0 + attempt)
            else:
                break
    raise RuntimeError(f"Failed to label batch after {max_retries + 1} attempts: {last_error}")


def chunk_list(values: List[int], size: int) -> List[List[int]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based annotation for gold eval JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/output/annotations/gold_eval_50_template.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/annotations/gold_eval_50_codex5.2_v1.jsonl"),
    )
    parser.add_argument("--provider", choices=["ollama", "siliconflow"], required=True)
    parser.add_argument("--model", default=os.getenv("MODEL", ""))
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-text-chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep after each request.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N queries.")
    parser.add_argument("--force", action="store_true", help="Ignore existing labels and re-annotate.")
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> LLMClient:
    if not args.model:
        raise ValueError("--model is required (or set MODEL env var).")
    if args.provider == "ollama":
        base_url = args.base_url or "http://localhost:11434"
        return OllamaClient(base_url=base_url, model=args.model, temperature=args.temperature, timeout=args.timeout)
    if args.provider == "siliconflow":
        api_key = args.api_key or os.getenv("SILICONFLOW_API_KEY", "")
        if not api_key:
            raise ValueError("SiliconFlow requires --api-key or SILICONFLOW_API_KEY env var.")
        base_url = args.base_url or "https://api.siliconflow.cn/v1"
        return SiliconFlowClient(
            base_url=base_url, api_key=api_key, model=args.model, temperature=args.temperature, timeout=args.timeout
        )
    raise ValueError(f"Unknown provider: {args.provider}")


def main() -> None:
    args = parse_args()
    input_records = load_jsonl(args.input)
    if args.limit > 0:
        input_records = input_records[: args.limit]

    existing_cache: Dict[Any, Dict[str, Dict[str, str]]] = {}
    if args.output.exists() and not args.force:
        existing_cache = build_existing_label_map(load_jsonl(args.output))

    client = build_client(args)
    output_records: List[Dict[str, Any]] = []

    total_queries = len(input_records)
    total_candidates = sum(len(rec.get("candidates") or []) for rec in input_records)
    print(f"Loaded {total_queries} queries, {total_candidates} candidates.")

    for idx, rec in enumerate(input_records, 1):
        qid = rec.get("query_id")
        candidates = rec.get("candidates") or []
        cached = existing_cache.get(qid, {})

        for c_idx, cand in enumerate(candidates):
            cached_item = cached.get(candidate_key(cand, c_idx))
            if cached_item and not args.force:
                cand["label"] = cached_item.get("label")
                cand["notes"] = cached_item.get("notes", "")

        todo_indices = [i for i, cand in enumerate(candidates) if normalize_label(cand.get("label")) not in LABELS]

        empty_indices = []
        for i in todo_indices:
            text = str(candidates[i].get("text") or "").strip()
            if not text:
                candidates[i]["label"] = "irrelevant"
                candidates[i]["notes"] = "empty_text"
                empty_indices.append(i)
        todo_indices = [i for i in todo_indices if i not in empty_indices]

        if todo_indices:
            answer_texts = [a.get("text") for a in rec.get("answers", []) if isinstance(a, dict) and a.get("text")]
            if not answer_texts:
                answer_texts = [str(a) for a in rec.get("answers", []) if a]

            batches = chunk_list(todo_indices, max(1, args.batch_size))
            for batch in batches:
                result_map = label_batch(
                    client,
                    query=str(rec.get("query") or ""),
                    answers=answer_texts,
                    candidates=candidates,
                    indices=batch,
                    max_text_chars=args.max_text_chars,
                    max_retries=args.max_retries,
                    delay=args.delay,
                )
                for cand_idx, result in result_map.items():
                    candidates[cand_idx]["label"] = result["label"]
                    candidates[cand_idx]["notes"] = result["notes"]

        annotated = sum(1 for cand in candidates if normalize_label(cand.get("label")) in LABELS)
        print(f"[{idx}/{total_queries}] query_id={qid} labeled {annotated}/{len(candidates)} candidates.")
        output_records.append(rec)

    write_jsonl(args.output, output_records)
    print(f"Saved annotated file to: {args.output}")


if __name__ == "__main__":
    main()
