# 架构设计

> QAnchor Pipeline 的五大核心模块、技术架构与证据路径。

---

## Pipeline 一览

一张图看懂整体流程（从 PDF 分块到评测）：

```text
┌─────────────────────────────────────────────────────────────┐
│                      QAnchor Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Step 1-2: PDF 分块 + 质检                                   │
│    └── ZenParse 分块 → parent/child 层级结构                 │
├─────────────────────────────────────────────────────────────┤
│  Step 3-5: 三路检索基线                                      │
│    ├── Qwen3-Embedding-0.6B (向量检索)                       │
│    ├── BM25 + jieba (关键词检索)                             │
│    └── RRF 融合 (k=60) → Top-50 候选                         │
├─────────────────────────────────────────────────────────────┤
│  Step 6: Reverse Mining                                      │
│    └── Answer Matching → 正例/负例自动标注                    │
├─────────────────────────────────────────────────────────────┤
│  Step 7-8: Gold Eval + Train/Dev 划分                        │
│    ├── 多标注裁决 (Gemini/Qwen/Codex)                        │
│    └── blacklist 隔离评测集                                  │
├─────────────────────────────────────────────────────────────┤
│  Step 9: Reranker LoRA 训练                                  │
│    ├── Qwen3-Reranker-0.6B-seq-cls/BGE-v2-m3                 │
│    ├── Listwise Softmax-CE Loss                              │
│    └── PEFT + Accelerate + wandb                             │
├─────────────────────────────────────────────────────────────┤
│  Step 10: 评测 (4组对照)                                     │
│    └── MRR@10 / NDCG@10 / P@10                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 五大核心模块（Input / Output / Implementation）

### 1) 数据分块与管理（Chunking & Management）
- **Input**：PDF 财报
- **Output**：含 parent/child 层级的 JSON chunks
- **Implementation**：
  - ZenParse `hybrid_table` 策略，DocLayout-YOLO 表格检测（`config/zenparse_config.yaml`）
  - parent_size=4000、child_size=1200、overlap=200
  - `ChunkIndex` 构建 `by_pdf / by_id / children_by_parent` 索引（`src/chunk_manager.py`）

### 2) 多路召回（Embedding + BM25 + RRF）
- **Input**：Query + 同文档 chunks
- **Output**：Top-50 候选（含 rank/score）
- **Implementation**：
  - `SentenceTransformer` 批量向量检索，显存控制（`src/embedding_retriever.py`）
  - `jieba + rank_bm25` 按 `pdf_stem` 建索引（`src/bm25_retriever.py`）
  - RRF 融合，`rrf_k=60`（`src/hybrid_fusion.py`）

### 3) Reverse Mining（弱监督核心）
- **Input**：Top-50 检索结果 + `finglm_master.jsonl`
- **Output**：Triplets（query + positive + hard negatives）
- **Implementation**：
  - key-value 规则匹配（优先 `prom_answer`，fallback 白名单字段）
  - 正例阈值 `confidence >= 0.5`
  - 高 rank 未匹配样本 → hard negatives，`neg_ratio=3`
  - 代码：`scripts/06_reverse_mining.py`、`src/answer_matcher.py`

### 4) Reranker 微调（LoRA + Listwise）
- **Input**：Triplets
- **Output**：LoRA Adapter
- **Implementation**：
  - `AutoModelForSequenceClassification`（Cross-Encoder）
  - LoRA 目标层：`q_proj/k_proj/v_proj/o_proj`
  - Listwise Softmax-CE Loss（直接优化排序）
  - 默认模板：`qwen3_template`（system/user/assistant 全量模板）
  - 代码：`scripts/09_train_reranker.py`

### 5) Gold Eval 评测
- **Input**：Reranker + Gold Eval
- **Output**：MRR@10 / NDCG@10 / P@10
- **Implementation**：
  - Gemini / Qwen / Codex 多模型投票裁决
  - 多数票 + all-diff 规则兜底
  - 代码：`scripts/07b_adjudicate_gold_eval.py`、`scripts/10_evaluate.py`

---

## SSOT（事实源）与证据路径
- **指标结果**：`data/output/eval/metrics_comparison_*.json`
- **评测配置**：`data/output/eval/eval_config_*.json`
- **对照报告**：`data/output/eval/stage1_reranker_comparison_report_20260118_v2.md`
- **统计显著性报告**：`data/output/eval/qwen3-reranker-0.6b/significance_report_*.json|md`
- **Reverse Mining 统计**：`data/output/mining/mining_stats_stage1.json`

### 统计稳健性补充（embedding_only vs hybrid_rrf）
- 数据来源：`per_query_scores_qwen3_template_20260117.jsonl` + `significance_report_qwen3_template_20260117_20260303-175859.md`
- **MRR@10**：`0.4115 -> 0.5756`（`+39.89%`），95% CI `[0.0889, 0.2413]`，显著为正
- **NDCG@10**：`+26.92%`，95% CI `[0.0814, 0.2067]`，显著为正
- **P@10**：`+21.13%`，提升为正，但配对 sign-flip `p=0.0518`（边界显著）
- 结论：`hybrid_rrf` 对 `embedding_only` 的提升在 **MRR/NDCG 维度统计稳健**；P@10 提升趋势存在但证据强度相对弱

---

## 检索配置契约

详见 [`retrieval_config_contract.md`](retrieval_config_contract.md)，描述三路检索的融合模式（RRF / 加权和 / 最大值）及参数生效规则。
