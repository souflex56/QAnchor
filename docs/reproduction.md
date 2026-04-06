# 复现指南（全流程）

> 面向学术验收的最小复现路径（从原始数据到评测结果）。

先设置阶段变量（用于路径与文件名）：
```bash
export STAGE=<stage>
export FINETUNED_RERANKER=<path_to_adapter>
```

## 0. 环境与版本（必做）
- 训练设备：RTX 4090（见 `data/output/eval/*reranker_comparison_report_20260118_v2.md` 附录 A.1）
- 评测设备：Apple M2 Max（MPS，见附录 A.4）
- Python：3.11.9（见 value-test.yml）
- 代码版本：建议使用 commit `2d7fc99`，这样最容易和报告中的结果保持一致

## 1. 安装依赖（必做）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 原始数据准备（必做）
- 原始数据目录：`data/input/`
- 数据索引与统计：`finglm_data_store/`
- 若数据外置下载，请补充来源与校验和。

## 3. 分块（ZenParse）（必做）
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

## 4. 检索与弱监督挖掘（必做）
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

## 5. Gold Eval 构建（必做）
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

## 6. 训练数据准备（必做）
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

## 7. 微调训练（必做）
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

## 8. 评测（必做）
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

## 8.1 统计显著性验证（推荐）
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

## 9. 复现一致性检查（必做）
- 关键配置：`config/weak_supervision_config.yaml`、`config/zenparse_config.yaml`、`value-test.yml`
- 核对指标：`data/output/eval/metrics_comparison_${STAGE}.json`、`data/output/eval/eval_report_${STAGE}.md`
- 如需提交证明，可保存指标文件的校验和（如 SHA256）。

## 10. 验收材料建议归档（可选）
- 训练数据：`data/output/train/`
- 评测集：`data/output/annotations/`
- 训练记录：`data/output/checkpoints/`
- 评测结果：`data/output/eval/`
