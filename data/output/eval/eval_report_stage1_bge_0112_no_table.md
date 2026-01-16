# stage1 Reranker Evaluation Report

## 1. Metrics

| Group | MRR@10 | NDCG@10 | P@10 |
| --- | --- | --- | --- |
| 1. Embedding-only | 0.4115 | 0.5341 | 0.1420 |
| 2. Hybrid (RRF) | 0.5457 | 0.6842 | 0.1540 |
| 3. Hybrid + Base Reranker | 0.6312 | 0.7915 | 0.1620 |
| 4. Hybrid + Fine-tuned Reranker | 0.6572 | 0.8249 | 0.1880 |

## 2. Improvements

- Hybrid vs Embedding-only: MRR 0.1342
- Fine-tuned vs Base: MRR 0.0260
