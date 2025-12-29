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