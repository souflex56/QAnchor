# QAnchor

> 基于弱监督学习的金融领域 RAG 检索排序 Pipeline（最佳 MRR@10: 0.7758）

QAnchor 是一个端到端的检索优化闭环：在缺乏高质量标注数据的前提下，利用现有 QA 对 + Reverse Mining 自动构建训练数据，微调 Reranker，并通过 Gold Eval 量化检索排序增益。

核心故事线：**PDF 结构化分块 → 多路召回 → 反向挖掘训练集 → Listwise 微调 → Gold Eval 评测**。

## 📌 评测结果
| 模型配置 | 阶段 | max_length | MRR@10 | NDCG@10 | P@10 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Reranker-0.6B-seq-cls | Base | 768 | 0.6115 | 0.7572 | 0.192 |
| **★ Qwen3-Reranker-0.6B-seq-cls** | **Finetuned** | **768** | **0.7758 (+26.9%)** | **0.8761 (+15.7%)** | **0.228 (+18.8%)** |
| BGE-v2-m3 | Base | 768 | 0.6443 | 0.7986 | 0.190 |
| BGE-v2-m3 | Finetuned | 768 | 0.7096 (+10.1%) | 0.8522 (+6.7%) | 0.226 (+18.9%) |
| BGE-v2-m3 | Base | 512 | 0.6553 | 0.7840 | 0.182 |
| BGE-v2-m3 | Finetuned | 512 | 0.7103 (+8.4%) | 0.8282 (+5.6%) | 0.224 (+23.1%) |

本次最优为 **Qwen3-Reranker-0.6B-seq-cls Finetuned**。  `max_length` 为 reranker 输入序列的最大长度（Query + Document 拼接后的 token 序列，超出会截断）。

完整报告：`data/output/eval/stage1_reranker_comparison_report_20260118_v2.md`
统计显著性报告（配对 bootstrap / sign-flip）：`data/output/eval/qwen3-reranker-0.6b/significance_report_*.md`
## 项目目标与范围
QAnchor 是一个弱监督 Query–Chunk 训练数据生成与检索排序 pipeline，用于支撑 ZenSeeker（A 股财报问答系统）的检索与排序能力。

**边界条件（Phase1）**
- 仅覆盖 Type1（type==1）
- 仅做**同文档证据排序**（不做跨 PDF 检索）
- 检索范围限制为 query 对应 `pdf_stem` 的 chunks

## 数据规模（本次评测）
- **训练/验证**：1,274 训练 triplets、247 验证 triplets（`data/output/train/*`）
- **Gold Eval**：50 条黄金评测查询（`data/output/annotations/gold_eval_50_extended_final.jsonl`）
- **Reverse Mining 统计（Stage1）**：
  - queries_with_pos=244 / total_queries=300（pos_found_rate=81.33%）
  - mined_triplets=1,871
  - **平均 hard negatives / 有效查询 ≈ 12.58**（`3069 / 244`）
  - avg_best_pos_rank=9.79
  - 统计来源：`data/output/mining/mining_stats_stage1.json`

---

## 🧩 五大核心模块（Input / Output / Implementation）

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

---


## 🚀 复现指南（全流程）

> 面向学术验收的最小复现路径（从原始数据到评测结果）。

先设置阶段变量（用于路径与文件名）：
```bash
export STAGE=<stage>
export FINETUNED_RERANKER=<path_to_adapter>
```

### 0. 环境与版本（必做）
- 训练设备：RTX 4090（见 `data/output/eval/*reranker_comparison_report_20260118_v2.md` 附录 A.1）
- 评测设备：Apple M2 Max（MPS，见附录 A.4）
- Python：3.11.9（见 value-test.yml）
- 代码版本：`release` / `2d7fc99`

### 1. 安装依赖（必做）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 原始数据准备（必做）
- 原始数据目录：`data/input/`
- 数据索引与统计：`finglm_data_store/`
- 若数据外置下载，请补充来源与校验和。

### 3. 分块（ZenParse）（必做）
```bash
python scripts/01_batch_chunking.py \
  --stage "$STAGE" \
  --config config/weak_supervision_config.yaml \
  --zen-root Reference/external/ZenParse \
  --zen-config config/zenparse_config.yaml
```
输出：`data/output/chunks/`

可选：分块质检
```bash
python scripts/01_chunk_checklist.py \
  --stage "$STAGE" \
  --config config/weak_supervision_config.yaml
```

### 4. 检索与弱监督挖掘（必做）
```bash
python scripts/02_embedding_retrieval.py \
  --stage "$STAGE" \
  --config config/weak_supervision_config.yaml \
  --generate-annotation-template \
  --output data/output/retrieval/embedding_${STAGE}_template.jsonl \
  --save-checkpoint

# 可选：自动标注模板（用于对齐/诊断）
python scripts/02_5_auto_label.py \
  --stage "$STAGE" \
  --config config/weak_supervision_config.yaml \
  --input data/output/retrieval/embedding_${STAGE}_template.jsonl \
  --output data/output/annotations/auto_label_${STAGE}.jsonl \
  --save-checkpoint

python scripts/05_three_way_retrieval.py \
  --stage "$STAGE" \
  --config config/weak_supervision_config.yaml \
  --exclude-pdfs data/output/quality/problematic_pdfs_${STAGE}.json

# 若出现 ModuleNotFoundError，可先设置：export PYTHONPATH=$(pwd)
python scripts/06_reverse_mining.py \
  --stage "$STAGE" \
  --retrieval-input data/output/retrieval/hybrid_rrf_top50_${STAGE}.jsonl \
  --master-path finglm_data_store/finglm_master.jsonl \
  --chunk-dir data/output/chunks \
  --output-dir data/output/mining \
  --checkpoint-dir data/output/checkpoints \
  --neg-ratio 3 \
  --confidence-threshold 0.5 \
  --save-checkpoint
```
输出：`data/output/retrieval/`、`data/output/mining/`

### 5. Gold Eval 构建（必做）
```bash
python scripts/07_prepare_gold_eval.py \
  --stage "$STAGE" \
  --input data/output/retrieval/hybrid_rrf_top20_${STAGE}.jsonl \
  --chunks-dir data/output/chunks \
  --size 50 \
  --seed 42 \
  --output data/output/annotations/gold_eval_50_template.jsonl \
  --blacklist config/eval_blacklist.json \
  --checkpoint data/output/checkpoints/${STAGE}_step_7_gold_eval.json \
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
  --input data/output/retrieval/hybrid_rrf_top20_${STAGE}.jsonl \
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
  --input data/output/retrieval/hybrid_rrf_top20_${STAGE}.jsonl \
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

### 6. 训练数据准备（必做）
```bash
python scripts/08_prepare_train_data.py \
  --stage "$STAGE" \
  --input data/output/mining/mined_triplets_${STAGE}.jsonl \
  --blacklist config/eval_blacklist.json \
  --confidence-threshold 0.7 \
  --train-ratio 0.9 \
  --seed 42 \
  --output-train data/output/train/train_triplets_${STAGE}.jsonl \
  --output-dev data/output/train/dev_triplets_${STAGE}.jsonl \
  --checkpoint data/output/checkpoints/${STAGE}_step_8_train_data.json
```
输出：`data/output/train/`

### 7. 微调训练（必做）
```bash
python scripts/09_train_reranker.py \
  --stage "$STAGE" \
  --config config/weak_supervision_config.yaml \
  --train-data data/output/train/train_triplets_${STAGE}.jsonl \
  --dev-data data/output/train/dev_triplets_${STAGE}.jsonl \
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

### 8. 评测（必做）
```bash
python scripts/10_evaluate.py \
  --stage "$STAGE" \
  --config config/weak_supervision_config.yaml \
  --gold-eval data/output/annotations/gold_eval_50_extended_final.jsonl \
  --embedding-results data/output/retrieval/embedding_top20_${STAGE}.jsonl \
  --hybrid-results data/output/retrieval/hybrid_rrf_top20_${STAGE}.jsonl \
  --base-reranker tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
  --finetuned-reranker "$FINETUNED_RERANKER" \
  --pair-format qwen3_template \
  --device mps \
  --batch-size 8 \
  --max-length 768 \
  --save-checkpoint
```
输出：`data/output/eval/`

### 8.1 统计显著性验证（推荐）
> 用 per-query 结果做配对 bootstrap / sign-flip，验证增益不是随机波动。

```bash
python scripts/10a_eval_significance.py \
  --per-query data/output/eval/qwen3-reranker-0.6b/per_query_scores_qwen3_template_20260117.jsonl \
  --baseline embedding_only \
  --treatment hybrid_rrf \
  --bootstrap-samples 20000 \
  --permutation-samples 20000 \
  --seed 42
```
输出：`data/output/eval/qwen3-reranker-0.6b/significance_report_*.json|md`

### 9. 复现一致性检查（必做）
- 关键配置：`config/weak_supervision_config.yaml`、`config/zenparse_config.yaml`、`value-test.yml`
- 核对指标：`data/output/eval/metrics_comparison_${STAGE}.json`、`data/output/eval/eval_report_${STAGE}.md`
- 如需提交证明，可保存指标文件的校验和（如 SHA256）。

### 10. 验收材料建议归档（可选）
- 训练数据：`data/output/train/`
- 评测集：`data/output/annotations/`
- 训练记录：`data/output/checkpoints/`
- 评测结果：`data/output/eval/`

---

## ✅ SSOT（事实源）与证据路径
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

## 🎯 面试摘要（Q&A 版）
- **Q：这不就是一个 RAG demo 吗？**  
  **A：不是。**我解决的是“金融长文档 + 缺标注数据”的落地难题：通过结构化分块、多路召回与排序闭环，把检索从“能跑通”提升到“有指标保障”。  
- **Q：没有标注数据怎么训练 Reranker？**  
  **A：用 Reverse Mining 自动构建训练集。**从 QA 标准答案中抽取关键值，匹配 Top-50 召回结果（confidence ≥ 0.5），并把高排名但不匹配的片段当作 hard negatives（neg_ratio=3）。Stage1 产出 1,871 条 triplets，平均每个有效查询 12.58 个 hard negatives。  
- **Q：为什么用 Listwise Loss，而不是 Pairwise？**  
  **A：Pairwise 只学“局部胜负”，Listwise 直接对齐“整组排序”。**下面是更细节的解释：  
  **1) “拆成 N 组二选一”是什么意思（Pairwise）？**  
  假设一个 query 有 1 个正例 P 和 3 个负例 N1/N2/N3。Pairwise 会拆成 3 条训练样本：  
  - (P vs N1)  
  - (P vs N2)  
  - (P vs N3)  
  每次只学“P 比某个 Ni 好”。这叫“局部比较”。  
  **问题**：它不关心 N1、N2、N3 之间的排序，也不保证 P 一定是全场第一。只要 P 比每个负例“略高一点”，pairwise 就满足了，但这不一定对应检索排序中“正例必须排第一”的目标。  
  **2) “压低所有 hard negatives”是什么意思（Listwise）？**  
  Listwise 把 P + 所有 hard negatives 当作一个整体候选列表。模型输出一串分数 `[s_pos, s_n1, s_n2, ...]`，然后做 softmax，让正例成为整组里概率最大的一项：  
  ```text
  loss = -log_softmax(scores)[0]
  ```  
  这会迫使模型**同时把所有负例压下去**，因为只要任何一个负例分数靠得太近，softmax 概率就会被“抢走”。  
  **3) 为什么说 Listwise 和业务指标对齐？**  
  检索评测看的是排名位置（MRR/NDCG），尤其是“正例排第几”。Listwise 的训练目标是“在一组候选里，正例排第一”，与评测目标一致。  
  **4) 项目中 Listwise 的具体实现？**（`scripts/09_train_reranker.py`）  
  - `TripletCollator` 组织候选：`candidates = [pos_text] + neg_texts`，并记录 `group_sizes`。  
  - `ListwiseTrainer.compute_loss`：`loss_i = -torch.log_softmax(group_scores, dim=0)[0]`，index 0 即正例。  
- **Q：你如何证明增益是真的？**  
  **A：两层证据。**先看同口径 Gold Eval：`embedding_only -> hybrid_rrf` 的 MRR@10 从 0.4115 到 0.5756（+39.9%），且 `unjudged_rate=0`。再看配对统计报告：MRR/NDCG 的 bootstrap 95%CI 均为正，说明不是随机抖动；P@10 也上升，但显著性更边界。  
- **Q：工程约束下怎么保证检索质量？**  
  **A：限定同文档检索 + Embedding/BM25/RRF 三路召回。**避免跨 PDF 噪声，同时提升召回覆盖与精度上限。  
- **Q：为什么要做 parent/child 分块？**  
  **A：检索粒度和理解粒度需要分离。**child 用于精确召回（`chunk_level=child`），parent 作为更完整上下文（parent_size=4000 / child_size=1200）。  
- **Q：为什么不只用向量检索？**  
  **A：语义召回和精确匹配各有盲点。**Embedding 擅长语义同义，BM25 擅长年份/数值等精确匹配，RRF 融合提升稳健性。  
- **Q：如何保证实验可复现？**  
  **A：所有关键配置与指标都固化为 SSOT 文件。**评测结果与配置分别落在 `metrics_comparison_*.json` 与 `eval_config_*.json`，可直接复核。
- **Q：为什么 Phase1 不做跨 PDF 检索？**  
  **A：这是问题域约束。**Type1 任务对应单公司单年份财报，跨文档会引入噪声与错误负例，因此限制 `pdf_stem` 能提升训练与评测稳定性。  
- **Q：为什么必须用 `qwen3_template`？**  
  **A：模板对性能影响显著。**对照报告显示 legacy 模板会显著拉低 Base MRR，而 `qwen3_template` 与 Qwen3 官方格式一致，能稳定释放模型能力。  
- **Q：训练成本如何控制？**  
  **A：用 LoRA + 小 batch + 梯度累积。**只微调投影层参数（q/k/v/o），再配合小 batch 与梯度累积；检索侧使用 embedding 缓存减少重复编码。
- **Q：如何避免评测数据泄露？**  
  **A：用 blacklist + 按 query_id 划分。**Gold Eval 构建时引入 blacklist 隔离；训练集准备阶段会过滤 blacklist 并按 query_id 划分 train/dev。  
- **Q：弱监督噪声怎么控制？**  
  **A：分两层阈值过滤。**Reverse Mining 阶段用 confidence≥0.5 判正例，训练准备阶段再以 confidence≥0.7 过滤噪声样本。  
- **Q：Qwen3 和 BGE 怎么取舍？**  
  **A：性能 vs 资源的权衡。**Qwen3-768 在本次评测中 MRR 最高；BGE-512 在资源受限时更稳妥（性能接近、推理成本低）。  

---

## 🔭 Future Work（尚未在主线评测脚本落地）
- **A1 数据去噪（CE 复审）**：对“规则未匹配”的候选再做 Cross-Encoder 复核，隔离 uncertain 样本，输出更干净的 `pos/verified_neg`，减少假 Hard Negative 污染。
- **A2 Hard Negative 策略升级**：在当前“检索排名选负例”之外，引入混合策略 / Curriculum / Online Mining，让负例难度随模型能力动态提升。
- **B1 `unjudged_rate>0` 敏感性分析**：当存在未标注候选时，分别给出下界（全负）/先验场景/上界（全正）区间，而不是只报单点分数。
- **B2 跨季度稳定性评估**：按 query 类型与时间切片做 out-of-time 分层评测，结合滚动窗口与季度 CI，防止“单次评测撞运气”。

完整路线图与实施细节见：`docs/future_work.md`
