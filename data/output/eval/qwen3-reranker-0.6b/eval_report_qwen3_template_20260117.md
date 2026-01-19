# stage1 Reranker Evaluation Report

## 1. Metrics

| Group | MRR@10 | NDCG@10 | P@10 |
| --- | --- | --- | --- |
| 1. Embedding-only | 0.4115 | 0.5341 | 0.1420 |
| 2. Hybrid (RRF) | 0.5756 | 0.6778 | 0.1720 |
| 3. Hybrid + Base Reranker | 0.6115 | 0.7572 | 0.1920 |
| 4. Hybrid + Fine-tuned Reranker | 0.7758 | 0.8761 | 0.2280 |

## 2. Improvements

- Hybrid vs Embedding-only: MRR 0.1641
- Fine-tuned vs Base: MRR 0.1643
