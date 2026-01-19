# stage1 Reranker Evaluation Report

## 1. Metrics

| Group | MRR@10 | NDCG@10 | P@10 |
| --- | --- | --- | --- |
| 1. Embedding-only | 0.4115 | 0.5341 | 0.1420 |
| 2. Hybrid (RRF) | 0.5756 | 0.6778 | 0.1720 |
| 3. Hybrid + Base Reranker | 0.6443 | 0.7986 | 0.1900 |
| 4. Hybrid + Fine-tuned Reranker | 0.7096 | 0.8522 | 0.2260 |

## 2. Improvements

- Hybrid vs Embedding-only: MRR 0.1641
- Fine-tuned vs Base: MRR 0.0652
