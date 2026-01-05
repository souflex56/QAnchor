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
import re
import time
from datetime import datetime
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
        error_msg = f"HTTP {exc.code} for {url}"
        if body:
            error_msg += f": {body}"
        if exc.code == 502:
            error_msg += "\n\n诊断建议："
            error_msg += "\n1. 检查 Ollama 服务是否运行: ollama serve 或检查服务状态"
            error_msg += "\n2. 确认模型是否存在: ollama list"
            error_msg += "\n3. 如果模型不存在，请先下载: ollama pull <model_name>"
            error_msg += "\n4. 检查 Ollama 是否在正确的端口运行（默认 11434）"
        raise RuntimeError(error_msg) from exc
    except urlerror.URLError as exc:
        error_msg = f"无法连接到 {url}: {exc.reason}"
        if "localhost" in url or "127.0.0.1" in url:
            error_msg += "\n\n诊断建议："
            error_msg += "\n1. 确认 Ollama 服务是否已启动"
            error_msg += "\n2. 检查服务是否在正确的地址和端口运行"
            error_msg += "\n3. 尝试运行: ollama serve"
        raise RuntimeError(error_msg) from exc
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

    def test_connection(self) -> None:
        """Test if Ollama service is accessible and model exists."""
        try:
            # Test if Ollama service is running by making a simple request
            url = f"{self.base_url}/api/tags"
            req = urlrequest.Request(url, method="GET")
            with urlrequest.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
                resp_data = json.loads(raw)
            
            if not isinstance(resp_data, dict) or "models" not in resp_data:
                raise RuntimeError("无法获取模型列表，Ollama 服务可能异常")
            
            # Check if model exists
            available_models = [m.get("name", "") for m in resp_data.get("models", [])]
            model_found = any(self.model in name or name == self.model for name in available_models)
            if not model_found:
                print(f"警告: 模型 '{self.model}' 可能不存在。")
                if available_models:
                    print(f"可用模型: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}")
                print(f"提示: 如果模型不存在，请运行: ollama pull {self.model}")
            
            # Check GPU/compute resources
            try:
                ps_url = f"{self.base_url}/api/ps"
                ps_req = urlrequest.Request(ps_url, method="GET")
                with urlrequest.urlopen(ps_req, timeout=3) as ps_resp:
                    ps_raw = ps_resp.read().decode("utf-8")
                    ps_data = json.loads(ps_raw)
                    if isinstance(ps_data, dict) and "models" in ps_data:
                        for model_info in ps_data.get("models", []):
                            if isinstance(model_info, dict):
                                compute_info = model_info.get("compute", {})
                                if compute_info:
                                    compute_type = compute_info.get("type", "unknown")
                                    print(f"GPU/计算资源: {compute_type}")
                                    if compute_type == "discrete" or "Metal" in str(compute_info):
                                        print("  ✓ 正在使用 GPU 加速 (Metal)")
                                    break
            except Exception:
                # Ignore errors in GPU check, it's optional
                pass
        except urlerror.URLError as exc:
            raise RuntimeError(f"无法连接到 Ollama 服务 ({self.base_url}): {exc.reason}\n请确认 Ollama 服务已启动 (运行 'ollama serve' 或检查服务状态)") from exc
        except Exception as exc:
            raise RuntimeError(f"测试 Ollama 连接时出错: {exc}") from exc

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
    original_text = text
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
    # If we get here, show what we tried to parse
    error_msg = "Failed to parse JSON from model response"
    if original_text:
        preview = original_text[:500] if len(original_text) > 500 else original_text
        error_msg += f"\n\n模型返回的前500字符:\n{preview}"
        if len(original_text) > 500:
            error_msg += "\n...(已截断)"
    raise ValueError(error_msg)


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
    lines.append("IMPORTANT: Return ONLY a valid JSON array, no other text. Format:")
    lines.append("[{\"index\": 0, \"label\": \"evidence\", \"notes\": \"\"}, {\"index\": 1, \"label\": \"related\", \"notes\": \"\"}, ...]")
    lines.append("")
    lines.append("CRITICAL Requirements:")
    lines.append(f"- You MUST return exactly {len(indices)} items in the JSON array (one for each candidate below)")
    lines.append(f"- The array MUST include indices: {sorted(indices)}")
    lines.append("- Return a JSON array with one object per candidate")
    lines.append("- Use the exact index number shown with each candidate (e.g., Candidate 0 -> index: 0)")
    lines.append("- Label must be exactly one of: \"evidence\", \"related\", or \"irrelevant\"")
    lines.append("- Keep notes empty string \"\" unless clarification is needed")
    lines.append("- Do NOT include any explanation, markdown formatting, or text outside the JSON array")
    lines.append("- Do NOT skip any candidate - you must label ALL candidates shown below")
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
    last_response: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            response_text = client.chat(messages)
            last_response = response_text
            parsed = extract_json_payload(response_text)
            results = normalize_results(parsed)
            result_map = {item["index"]: item for item in results}
            missing = [idx for idx in indices if idx not in result_map]
            if missing:
                received_indices = sorted(result_map.keys())
                error_detail = f"Missing indices in response: {missing}\n"
                error_detail += f"Expected indices: {sorted(indices)}\n"
                error_detail += f"Received indices: {received_indices}\n"
                error_detail += f"Number of results: {len(results)}, expected: {len(indices)}"
                raise ValueError(error_detail)
            if delay > 0:
                time.sleep(delay)
            return result_map
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.0 + attempt)
            else:
                break
    # Build detailed error message
    error_msg = f"Failed to label batch after {max_retries + 1} attempts: {last_error}"
    if last_response and isinstance(last_error, ValueError):
        if "Failed to parse JSON" in str(last_error) or "Missing indices" in str(last_error):
            error_msg += f"\n\n完整模型响应:\n{last_response}"
            if "Missing indices" in str(last_error):
                error_msg += f"\n\n提示: 模型可能没有为所有候选者返回结果。请检查模型响应是否包含所有必需的索引。"
    raise RuntimeError(error_msg)


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
        default=None,
        help="Output path. If not specified, will be auto-generated based on model name and timestamp.",
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


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file paths."""
    # Replace common special characters with safe alternatives
    model = re.sub(r'[^\w\-_.]', '_', model)
    # Replace multiple underscores with single underscore
    model = re.sub(r'_+', '_', model)
    # Remove leading/trailing underscores
    model = model.strip('_')
    return model or "unknown_model"


def main() -> None:
    args = parse_args()
    
    # Auto-generate output path if not specified
    if args.output is None:
        model_name = sanitize_model_name(args.model or os.getenv("MODEL", "unknown"))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = Path(f"data/output/annotations/gold_eval_50_{model_name}_{timestamp}.jsonl")
    
    input_records = load_jsonl(args.input)
    if args.limit > 0:
        input_records = input_records[: args.limit]

    existing_cache: Dict[Any, Dict[str, Dict[str, str]]] = {}
    if args.output.exists() and not args.force:
        existing_cache = build_existing_label_map(load_jsonl(args.output))

    client = build_client(args)
    
    # Test connection for Ollama clients
    if isinstance(client, OllamaClient):
        print("正在测试 Ollama 连接...")
        try:
            client.test_connection()
            print(f"✓ Ollama 连接成功，使用模型: {args.model}")
        except Exception as exc:
            print(f"✗ Ollama 连接测试失败: {exc}")
            raise
    
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
