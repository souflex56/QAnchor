# QAnchor

> 基于弱监督学习的金融领域 RAG 检索排序 Pipeline（最佳 MRR@10: 0.7758）

一个端到端的金融财报问答检索排序 pipeline，通过 Reverse Mining 自动挖掘训练数据，结合多路检索融合与 Reranker 微调，实现高质量证据排序。评测基于：1,274 条训练 query–chunk triplets、247 条验证 triplets、50 条黄金评测查询（详见评测报告附录）。

## 项目目标与范围
QAnchor 是一个弱监督 Query–Chunk 训练数据生成 pipeline，用于支撑 ZenSeeker（A 股财报问答系统）的检索与排序能力。核心目标：
- Phase1 仅覆盖 Type1（type==1），任务为同文档证据排序（reranker），不做跨 PDF 检索
- 通过 Reverse Mining 从 FinGLM 数据（2,613 条 Type1 QA）中自动挖掘正例 + hard negatives，生成 triplets 训练数据
- 建立三路检索基线（Embedding-only / BM25 / Hybrid RRF），用于支撑候选召回
- 构建 Gold Eval（50→100）多标注裁决评测集，验证 Reranker 微调效果
- 训练 Reranker 并完成 4 组对照实验（Embedding-only / Hybrid / Base Reranker / Fine-tuned Reranker）

## 🎯 项目亮点

### 1. Reverse Mining 弱监督框架
- **挑战**：训练数据标注成本高，周期长
- **方法**：从 FinGLM QA 自动匹配答案并挖掘 hard negatives，形成 triplets
- **证据（本次评测）**：1,274 条训练 triplets，平均 14.4 个 negatives/查询

### 2. Hybrid 检索 + Reranker 二阶段排序
- **召回**：Embedding + BM25 + RRF 融合（Top-50 候选）
- **精排**：Qwen3-Reranker-0.6B + LoRA（r=16）
- **指标（本次评测）**：Base MRR 0.6115 → Finetuned 0.7758（+26.9%）

### 3. 完整评测体系与可复现性
- **Gold Eval 多标注裁决**：Gemini / Qwen / Codex 多模型投票
- **指标**：MRR@10 / NDCG@10 / P@10
- **SSOT（事实源）**：以 `data/output/eval/metrics_comparison_*.json` 与 `data/output/eval/eval_config_*.json` 为准

## 📊 关键结果（本次评测）
- **最佳配置**：Qwen3-0.6B + `qwen3_template`, max_len=768
- **核心指标**：MRR@10 0.7758 / NDCG@10 0.8761 / P@10 0.228
- **对比基线**：Embedding-only MRR@10 0.4115；Hybrid RRF MRR@10 0.5756
- **数据来源**：`data/output/eval/*reranker_comparison_report_20260118_v2.md`

| 模型配置 | 阶段 | MRR@10 | NDCG@10 | P@10 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B (qwen3_template, 768) | Base | 0.6115 | 0.7572 | 0.192 |
| Qwen3-0.6B (qwen3_template, 768) | Finetuned | 0.7758 | 0.8761 | 0.228 |
| BGE-v2-m3 (hf_pair, 768) | Base | 0.6443 | 0.7986 | 0.190 |
| BGE-v2-m3 (hf_pair, 768) | Finetuned | 0.7096 | 0.8522 | 0.226 |
| BGE-v2-m3 (hf_pair, 512) | Base | 0.6553 | 0.7840 | 0.182 |
| BGE-v2-m3 (hf_pair, 512) | Finetuned | 0.7103 | 0.8282 | 0.224 |

## 🛠️ 技术架构
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

## 🚀 复现指南（全流程）

> 面向学术验收的最小复现路径（从原始数据到评测结果）。

先设置阶段变量（用于路径与文件名）：
```bash
export STAGE=<stage>
export FINETUNED_RERANKER=<path_to_adapter>
```

### 0. 环境与版本
- 训练设备：RTX 4090（见 `data/output/eval/*reranker_comparison_report_20260118_v2.md` 附录 A.1）
- 评测设备：Apple M2 Max（MPS，见附录 A.4）
- Python：3.11.9（见 value-test.yml）
- 代码版本：`release` / `2d7fc99`

### 1. 安装依赖
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 原始数据准备
- 原始数据目录：`data/input/`
- 数据索引与统计：`finglm_data_store/`
- 若数据外置下载，请补充来源与校验和。

### 3. 分块（ZenParse）
```bash
python scripts/01_batch_chunking.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml \
  --zen-root Reference/external/ZenParse \
  --zen-config config/zenparse_config.yaml
```
输出：`data/output/chunks/`

可选：分块质检
```bash
python scripts/01_chunk_checklist.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml
```

### 4. 检索与弱监督挖掘
```bash
python scripts/02_embedding_retrieval.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml \
  --generate-annotation-template \
  --output data/output/retrieval/embedding_stage1_template.jsonl \
  --save-checkpoint

# 可选：自动标注模板（用于对齐/诊断）
python scripts/02_5_auto_label.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml \
  --input data/output/retrieval/embedding_stage1_template.jsonl \
  --output data/output/annotations/auto_label_stage1.jsonl \
  --save-checkpoint

python scripts/05_three_way_retrieval.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml \
  --exclude-pdfs data/output/quality/problematic_pdfs_stage1.json

# 若出现 ModuleNotFoundError，可先设置：export PYTHONPATH=$(pwd)
python scripts/06_reverse_mining.py \
  --stage stage1 \
  --retrieval-input data/output/retrieval/hybrid_rrf_top50_stage1.jsonl \
  --master-path finglm_data_store/finglm_master.jsonl \
  --chunk-dir data/output/chunks \
  --output-dir data/output/mining \
  --checkpoint-dir data/output/checkpoints \
  --neg-ratio 3 \
  --confidence-threshold 0.5 \
  --save-checkpoint
```
输出：`data/output/retrieval/`、`data/output/mining/`

### 5. Gold Eval 构建
```bash
python scripts/07_prepare_gold_eval.py \
  --stage stage1 \
  --input data/output/retrieval/hybrid_rrf_top20_stage1.jsonl \
  --chunks-dir data/output/chunks \
  --size 50 \
  --seed 42 \
  --output data/output/annotations/gold_eval_50_template.jsonl \
  --blacklist config/eval_blacklist.json \
  --checkpoint data/output/checkpoints/stage1_step_7_gold_eval.json \
  --blacklist-mode replace

# Qwen 标注（需外部 API）
python scripts/07_llm_gold_eval_annotate.py \
  --input data/output/annotations/gold_eval_50_template.jsonl \
  --provider siliconflow \
  --model qwen2.5-7b-instruct \
  --api-key "$SILICONFLOW_API_KEY" \
  --output data/output/annotations/gold_eval_50_qwen2.5_7b-instruct_20260101-232811.jsonl

# Gemini/Codex 标注为外部流程，产物路径如下：
# - data/output/annotations/gold-eval-gemini.csv
# - data/output/annotations/gold_eval_50_codex5.2_v1.jsonl

python scripts/07c_diagnose_alignment.py \
  --template data/output/annotations/gold_eval_50_template.jsonl \
  --gemini-csv data/output/annotations/gold-eval-gemini.csv \
  --qwen-jsonl data/output/annotations/gold_eval_50_qwen2.5_7b-instruct_20260101-232811.jsonl \
  --codex-jsonl data/output/annotations/gold_eval_50_codex5.2_v1.jsonl

python scripts/07b_adjudicate_gold_eval.py \
  --template data/output/annotations/gold_eval_50_template.jsonl \
  --gemini-csv data/output/annotations/gold-eval-gemini.csv \
  --qwen-jsonl data/output/annotations/gold_eval_50_qwen2.5_7b-instruct_20260101-232811.jsonl \
  --codex-jsonl data/output/annotations/gold_eval_50_codex5.2_v1.jsonl \
  --extended-out data/output/annotations/gold_eval_50_extended.jsonl \
  --core-out data/output/annotations/gold_eval_50_core.jsonl \
  --adjudication-out data/output/annotations/gold_eval_50_adjudication.jsonl

python scripts/07d_resample_gold_eval.py \
  --input data/output/retrieval/hybrid_rrf_top20_stage1.jsonl \
  --template data/output/annotations/gold_eval_50_template.jsonl \
  --extended data/output/annotations/gold_eval_50_extended.jsonl \
  --chunks-dir data/output/chunks \
  --size 50 \
  --seed 42 \
  --output data/output/annotations/gold_eval_50_template_resampled.jsonl \
  --blacklist-out config/eval_blacklist_resampled.json \
  --report-out data/output/annotations/gold_eval_50_resample_report.json \
  --exclude-dropped config/gold_eval_dropped.json

python scripts/07e_prepare_gold_eval_additions.py \
  --input data/output/retrieval/hybrid_rrf_top20_stage1.jsonl \
  --chunks-dir data/output/chunks \
  --add-size 15 \
  --seed 42 \
  --exclude-template data/output/annotations/gold_eval_50_template.jsonl \
  --exclude-blacklist config/eval_blacklist.json \
  --exclude-dropped config/gold_eval_dropped.json \
  --output data/output/annotations/gold_eval_50_template_additions_15.jsonl \
  --report-out data/output/annotations/gold_eval_50_additions_15_report.json
```
输出：`data/output/annotations/`
说明：最终评测默认使用 `gold_eval_50_extended_final.jsonl`；如只生成到 `gold_eval_50_extended.jsonl`，可先以该文件进行评测并标注差异。

### 6. 训练数据准备
```bash
python scripts/08_prepare_train_data.py \
  --stage stage1 \
  --input data/output/mining/mined_triplets_stage1.jsonl \
  --blacklist config/eval_blacklist.json \
  --confidence-threshold 0.7 \
  --train-ratio 0.9 \
  --seed 42 \
  --output-train data/output/train/train_triplets_stage1.jsonl \
  --output-dev data/output/train/dev_triplets_stage1.jsonl \
  --checkpoint data/output/checkpoints/stage1_step_8_train_data.json
```
输出：`data/output/train/`

### 7. 微调训练
```bash
python scripts/09_train_reranker.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml \
  --train-data data/output/train/train_triplets_stage1.jsonl \
  --dev-data data/output/train/dev_triplets_stage1.jsonl \
  --gold-eval data/output/annotations/gold_eval_50_extended_final.jsonl \
  --blacklist config/eval_blacklist.json \
  --model-name tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
  --pair-format qwen3_template \
  --max-length 768 \
  --max-neg 7 \
  --learning-rate 2e-5 \
  --num-epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --seed 42 \
  --save-checkpoint
```
输出：`data/output/checkpoints/`（可选：`data/output/artifacts/`）

### 8. 评测
```bash
python scripts/10_evaluate.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml \
  --gold-eval data/output/annotations/gold_eval_50_extended_final.jsonl \
  --embedding-results data/output/retrieval/embedding_top20_stage1.jsonl \
  --hybrid-results data/output/retrieval/hybrid_rrf_top20_stage1.jsonl \
  --base-reranker tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
  --finetuned-reranker data/output/artifacts/reranker/tomaarsen-qwen3-reranker-0.6b-seq-cls/cuda_nvidia-geforce-rtx-4090/run_20260117-103627_stage1_lr2e-05_e3_bs1_ga8_len768_neg7_seed42/adapter \
  --pair-format qwen3_template \
  --device mps \
  --batch-size 8 \
  --max-length 768 \
  --save-checkpoint
```
输出：`data/output/eval/`

### 9. 复现一致性检查
- 关键配置：`config/weak_supervision_config.yaml`、`config/zenparse_config.yaml`、`value-test.yml`
- 核对指标：`data/output/eval/metrics_comparison_stage1.json`、`data/output/eval/eval_report_stage1.md`
- 如需提交证明，可保存指标文件的校验和（如 SHA256）。

### 10. 验收材料建议归档
- 训练数据：`data/output/train/`
- 评测集：`data/output/annotations/`
- 训练记录：`data/output/checkpoints/`
- 评测结果：`data/output/eval/`
