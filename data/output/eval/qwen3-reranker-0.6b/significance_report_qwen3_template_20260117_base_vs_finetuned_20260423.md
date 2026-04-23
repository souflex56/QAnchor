# Significance Report

- Per-query: `data/output/eval/qwen3-reranker-0.6b/per_query_scores_qwen3_template_20260117.jsonl`
- Baseline: `hybrid_base_reranker`
- Treatment: `hybrid_finetuned_reranker`
- Sample count: `50`

| Metric | Baseline | Treatment | Delta | Rel Delta | 95% CI | p(diff<=0) | Sign-flip p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mrr@10 | 0.611524 | 0.775833 | 0.164310 | 26.87% | [0.071333, 0.264976] | 0 | 0.0016 |
| ndcg@10 | 0.757155 | 0.876117 | 0.118962 | 15.71% | [0.047800, 0.195956] | 0.00025 | 0.00175 |
| p@10 | 0.192000 | 0.228000 | 0.036000 | 18.75% | [0.020000, 0.056000] | 0 | 0.0002 |
