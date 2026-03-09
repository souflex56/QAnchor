# Significance Report

- Per-query: `data/output/eval/qwen3-reranker-0.6b/per_query_scores_qwen3_template_20260117.jsonl`
- Baseline: `embedding_only`
- Treatment: `hybrid_rrf`
- Sample count: `50`

| Metric | Baseline | Treatment | Delta | Rel Delta | 95% CI | p(diff<=0) | Sign-flip p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mrr@10 | 0.411500 | 0.575635 | 0.164135 | 39.89% | [0.088944, 0.241278] | 5e-05 | 0 |
| ndcg@10 | 0.534078 | 0.677839 | 0.143760 | 26.92% | [0.081431, 0.206673] | 0 | 5e-05 |
| p@10 | 0.142000 | 0.172000 | 0.030000 | 21.13% | [0.002000, 0.060000] | 0.01735 | 0.0518 |
