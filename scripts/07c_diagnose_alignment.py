#!/usr/bin/env python3
"""
诊断B类query（全irrelevant）的数据对齐问题
检查：1) chunk_id一致性 2) 缺失投票 3) pdf_stem匹配
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional

# B类query IDs（全irrelevant）
B_CLASS_QUERIES = {9908, 2393, 9462, 7774, 9927, 8102, 8032, 8323}

def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_gemini_csv(path: Path) -> Dict[Tuple[int, str], Dict]:
    """加载Gemini CSV，返回 {(query_id, chunk_id): {label, notes, ...}}"""
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["query_id"]) if row.get("query_id") else None
            chunk = row.get("chunk_id", "").strip()
            if qid is None or not chunk:
                continue
            # 补齐前导0
            if chunk.isdigit() and len(chunk) < 8:
                chunk = chunk.zfill(8)
            key = (qid, chunk)
            rows[key] = {
                "label": (row.get("label") or "").strip().lower(),
                "notes": (row.get("notes") or "").strip(),
                "query": (row.get("query") or "").strip(),
            }
    return rows

def build_label_map(records: List[Dict], source_name: str) -> Dict[Tuple[int, str], Optional[str]]:
    """从JSONL构建 {(query_id, chunk_id): label} 映射"""
    label_map = {}
    for rec in records:
        qid = rec.get("query_id")
        if qid is None:
            continue
        for cand in rec.get("candidates") or []:
            chunk_id = cand.get("chunk_id")
            if not chunk_id:
                continue
            key = (qid, chunk_id)
            label = cand.get("label")
            if label:
                label = label.strip().lower()
                if label.startswith("evid"):
                    label = "evidence"
                elif label.startswith("rel"):
                    label = "related"
                elif label.startswith("irr"):
                    label = "irrelevant"
            label_map[key] = label if label in {"evidence", "related", "irrelevant"} else None
    return label_map

def check_query_alignment(
    template_path: Path,
    gemini_csv_path: Path,
    qwen_jsonl_path: Path,
    codex_jsonl_path: Path,
    target_qids: Set[int],
) -> None:
    """检查目标query的数据对齐问题"""
    
    # 加载模板（获取期望的chunk_id列表）
    template_records = load_jsonl(template_path)
    template_by_qid = {rec.get("query_id"): rec for rec in template_records if rec.get("query_id") in target_qids}
    
    # 加载三份标注
    gemini_data = load_gemini_csv(gemini_csv_path)
    qwen_data = build_label_map(load_jsonl(qwen_jsonl_path), "qwen")
    codex_data = build_label_map(load_jsonl(codex_jsonl_path), "codex")
    
    print("=" * 80)
    print("B类Query数据对齐诊断报告")
    print("=" * 80)
    
    for qid in sorted(target_qids):
        template_rec = template_by_qid.get(qid)
        if not template_rec:
            print(f"\n[ERROR] Query {qid}: 模板中未找到")
            continue
        
        candidates = template_rec.get("candidates") or []
        expected_chunk_ids = {cand.get("chunk_id") for cand in candidates if cand.get("chunk_id")}
        expected_pdf_stem = template_rec.get("pdf_stem")
        query_text = template_rec.get("query", "")
        
        print(f"\n{'='*80}")
        print(f"Query ID: {qid}")
        print(f"Query: {query_text[:60]}...")
        print(f"Expected PDF: {expected_pdf_stem}")
        print(f"Expected chunks: {len(expected_chunk_ids)}")
        
        # 1. 检查chunk_id一致性
        gemini_chunks = {cid for (q, cid) in gemini_data.keys() if q == qid}
        qwen_chunks = {cid for (q, cid) in qwen_data.keys() if q == qid}
        codex_chunks = {cid for (q, cid) in codex_data.keys() if q == qid}
        
        all_chunks = expected_chunk_ids | gemini_chunks | qwen_chunks | codex_chunks
        
        print(f"\n[1] Chunk ID一致性检查:")
        print(f"  模板: {len(expected_chunk_ids)} chunks")
        print(f"  Gemini: {len(gemini_chunks)} chunks")
        print(f"  Qwen: {len(qwen_chunks)} chunks")
        print(f"  Codex: {len(codex_chunks)} chunks")
        
        missing_in_gemini = expected_chunk_ids - gemini_chunks
        missing_in_qwen = expected_chunk_ids - qwen_chunks
        missing_in_codex = expected_chunk_ids - codex_chunks
        
        if missing_in_gemini:
            print(f"  [WARNING] Gemini缺失: {sorted(missing_in_gemini)[:5]}...")
        if missing_in_qwen:
            print(f"  [WARNING] Qwen缺失: {sorted(missing_in_qwen)[:5]}...")
        if missing_in_codex:
            print(f"  [WARNING] Codex缺失: {sorted(missing_in_codex)[:5]}...")
        
        extra_in_gemini = gemini_chunks - expected_chunk_ids
        extra_in_qwen = qwen_chunks - expected_chunk_ids
        extra_in_codex = codex_chunks - expected_chunk_ids
        
        if extra_in_gemini:
            print(f"  [WARNING] Gemini多余: {sorted(extra_in_gemini)[:5]}...")
        if extra_in_qwen:
            print(f"  [WARNING] Qwen多余: {sorted(extra_in_qwen)[:5]}...")
        if extra_in_codex:
            print(f"  [WARNING] Codex多余: {sorted(extra_in_codex)[:5]}...")
        
        # 2. 检查缺失投票（被当成irrelevant）
        print(f"\n[2] 缺失投票检查（None/null被当成irrelevant）:")
        missing_votes = []
        for chunk_id in expected_chunk_ids:
            gemini_label = gemini_data.get((qid, chunk_id), {}).get("label")
            qwen_label = qwen_data.get((qid, chunk_id))
            codex_label = codex_data.get((qid, chunk_id))
            
            if gemini_label is None or gemini_label == "":
                missing_votes.append((chunk_id, "gemini", gemini_label))
            if qwen_label is None:
                missing_votes.append((chunk_id, "qwen", qwen_label))
            if codex_label is None:
                missing_votes.append((chunk_id, "codex", codex_label))
        
        if missing_votes:
            print(f"  [WARNING] 发现 {len(missing_votes)} 个缺失投票:")
            for chunk_id, source, label in missing_votes[:10]:
                print(f"    {chunk_id} ({source}): {label}")
        else:
            print(f"  [OK] 所有chunk都有三份标注")
        
        # 3. 检查pdf_stem匹配
        print(f"\n[3] PDF Stem匹配检查:")
        pdf_mismatches = []
        for cand in candidates[:5]:  # 只检查前5个作为样本
            chunk_id = cand.get("chunk_id")
            chunk_pdf_stem = cand.get("pdf_stem")
            if chunk_pdf_stem and chunk_pdf_stem != expected_pdf_stem:
                pdf_mismatches.append((chunk_id, chunk_pdf_stem, expected_pdf_stem))
        
        if pdf_mismatches:
            print(f"  [WARNING] 发现PDF不匹配:")
            for chunk_id, actual, expected in pdf_mismatches:
                print(f"    {chunk_id}: {actual} != {expected}")
        else:
            print(f"  [OK] 前5个chunk的pdf_stem匹配")
        
        # 4. 统计标签分布
        print(f"\n[4] 标签分布统计:")
        label_counts = {"evidence": 0, "related": 0, "irrelevant": 0, "missing": 0}
        for chunk_id in expected_chunk_ids:
            gemini_label = gemini_data.get((qid, chunk_id), {}).get("label")
            qwen_label = qwen_data.get((qid, chunk_id))
            codex_label = codex_data.get((qid, chunk_id))
            
            # 统计各来源的标签
            for label in [gemini_label, qwen_label, codex_label]:
                if label in label_counts:
                    label_counts[label] += 1
                elif label is None or label == "":
                    label_counts["missing"] += 1
        
        print(f"  Evidence: {label_counts['evidence']}")
        print(f"  Related: {label_counts['related']}")
        print(f"  Irrelevant: {label_counts['irrelevant']}")
        print(f"  Missing: {label_counts['missing']}")
        
        # 5. 检查是否有任何正例（evidence或related）
        has_positive = False
        for chunk_id in expected_chunk_ids:
            gemini_label = gemini_data.get((qid, chunk_id), {}).get("label")
            qwen_label = qwen_data.get((qid, chunk_id))
            codex_label = codex_data.get((qid, chunk_id))
            
            if gemini_label in {"evidence", "related"} or \
               qwen_label in {"evidence", "related"} or \
               codex_label in {"evidence", "related"}:
                has_positive = True
                print(f"\n  [INFO] 发现正例: chunk_id={chunk_id}")
                print(f"    Gemini: {gemini_label}, Qwen: {qwen_label}, Codex: {codex_label}")
                break
        
        if not has_positive:
            print(f"\n  [CONFIRMED] 确实全为irrelevant，无正例")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True, help="gold_eval_50_template.jsonl")
    parser.add_argument("--gemini-csv", type=Path, required=True, help="gold-eval-gemini.csv")
    parser.add_argument("--qwen-jsonl", type=Path, required=True, help="gold_eval_50_qwen*.jsonl")
    parser.add_argument("--codex-jsonl", type=Path, required=True, help="gold_eval_50_codex*.jsonl")
    args = parser.parse_args()
    
    check_query_alignment(
        args.template,
        args.gemini_csv,
        args.qwen_jsonl,
        args.codex_jsonl,
        B_CLASS_QUERIES,
    )

