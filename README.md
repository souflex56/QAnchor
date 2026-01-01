1. 项目目标和范围理解

QAnchor 是一个弱监督 Query–Chunk 训练数据生成 pipeline，用于支撑 ZenSeeker（A 股财报问答系统）的检索与排序能力。核心目标：
•  Phase1 仅覆盖 Type1（type==1），任务为同文档证据排序（reranker），不做跨 PDF 检索
•  通过 Reverse Mining 从 FinGLM 数据（2,613 条 Type1 QA）中自动挖掘正例+hard negatives，生成 triplets 训练数据
•  建立三路检索基线（Embedding-only / BM25 / Hybrid RRF），用于支撑候选召回
•  构建 Gold Eval（50→100）人工评测集，验证 Reranker 微调效果
•  训练 Reranker 并完成 4 组对照实验（Embedding-only / Hybrid / Base Reranker / Fine-tuned Reranker）

# 使用步骤 
## 1. 分块 

```python scripts/01_batch_chunking.py \    
  --stage stage1 \
  --config config/weak_supervision_config.yaml  
```

质检

```python scripts/01_chunk_checklist.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml  
```

## 2. embedding 
```
python scripts/02_embedding_retrieval.py \
  --stage stage1 \
  --config config/weak_supervision_config.yaml \
  --top-k 50 \
  --output data/output/retrieval/embedding_stage1_baseline_top50.jsonl \
  --output-format nested \
  --save-checkpoint \
  --checkpoint-path data/output/checkpoints/stage1_embedding_cache_$(date +%Y%m%d-%H%M%S).json \
  --exclude-pdfs data/output/quality/problematic_pdfs_stage1.json


```