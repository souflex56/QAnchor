#!/usr/bin/env python3
"""Step 10a: paired significance analysis for retrieval metrics."""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _now_stamp() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y%m%d-%H%M%S")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_mean(values: Sequence[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(q * (len(sorted_values) - 1))
    return float(sorted_values[idx])


def _bootstrap_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float, float, float]:
    if len(baseline) != len(treatment):
        raise ValueError("baseline/treatment length mismatch")
    n = len(baseline)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    obs = _safe_mean([t - b for b, t in zip(baseline, treatment)])
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for __ in range(n)]
        d = _safe_mean([treatment[i] - baseline[i] for i in idx])
        deltas.append(d)
    deltas.sort()

    lo = _quantile(deltas, alpha / 2.0)
    hi = _quantile(deltas, 1.0 - alpha / 2.0)
    p_le_zero = sum(1 for x in deltas if x <= 0.0) / len(deltas)
    return obs, lo, hi, p_le_zero


def _paired_signflip_pvalue(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    n_perm: int,
    seed: int,
) -> float:
    if len(baseline) != len(treatment):
        raise ValueError("baseline/treatment length mismatch")
    n = len(baseline)
    if n == 0:
        return 1.0

    obs = abs(_safe_mean([t - b for b, t in zip(baseline, treatment)]))
    rng = random.Random(seed + 97)
    count = 0
    for _ in range(n_perm):
        s = 0.0
        for b, t in zip(baseline, treatment):
            d = t - b
            if rng.random() < 0.5:
                d = -d
            s += d
        if abs(s / n) >= obs:
            count += 1
    return count / n_perm


def _infer_suffix(per_query_path: Path) -> str:
    stem = per_query_path.stem
    m = re.match(r"per_query_scores_(.+)$", stem)
    return m.group(1) if m else stem


def _resolve_metric_list(raw: str) -> List[str]:
    metrics = [x.strip() for x in raw.split(",") if x.strip()]
    if not metrics:
        raise ValueError("metrics must not be empty")
    return metrics


def _extract_metric_series(
    rows: Sequence[Dict[str, Any]],
    group: str,
    metric: str,
) -> List[float]:
    series: List[float] = []
    for row in rows:
        series.append(float(row.get(group, {}).get(metric, 0.0)))
    return series


def _write_md(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired significance analysis for eval metrics.")
    parser.add_argument(
        "--per-query",
        type=Path,
        required=True,
        help="Path to per_query_scores_*.jsonl",
    )
    parser.add_argument("--baseline", default="embedding_only")
    parser.add_argument("--treatment", default="hybrid_rrf")
    parser.add_argument("--metrics", default="mrr@10,ndcg@10,p@10")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.per_query)
    metrics = _resolve_metric_list(args.metrics)

    suffix = _infer_suffix(args.per_query)
    default_dir = args.per_query.parent
    stamp = _now_stamp()
    out_json = args.output_json or default_dir / f"significance_report_{suffix}_{stamp}.json"
    out_md = args.output_md or default_dir / f"significance_report_{suffix}_{stamp}.md"

    per_metric: Dict[str, Any] = {}
    for metric in metrics:
        base = _extract_metric_series(rows, args.baseline, metric)
        treat = _extract_metric_series(rows, args.treatment, metric)
        obs, ci_lo, ci_hi, p_boot_le_zero = _bootstrap_ci(
            base,
            treat,
            n_bootstrap=args.bootstrap_samples,
            alpha=args.alpha,
            seed=args.seed,
        )
        p_perm = _paired_signflip_pvalue(
            base,
            treat,
            n_perm=args.permutation_samples,
            seed=args.seed,
        )
        diffs = [t - b for b, t in zip(base, treat)]
        per_metric[metric] = {
            "baseline_mean": _safe_mean(base),
            "treatment_mean": _safe_mean(treat),
            "abs_delta": obs,
            "rel_delta_vs_baseline": (obs / _safe_mean(base)) if _safe_mean(base) else None,
            "improved_queries": sum(1 for d in diffs if d > 1e-12),
            "degraded_queries": sum(1 for d in diffs if d < -1e-12),
            "tied_queries": sum(1 for d in diffs if abs(d) <= 1e-12),
            "bootstrap_ci": {
                "alpha": args.alpha,
                "low": ci_lo,
                "high": ci_hi,
                "p_diff_le_zero": p_boot_le_zero,
            },
            "paired_signflip_p_two_sided": p_perm,
        }

    payload = {
        "per_query_path": str(args.per_query),
        "baseline": args.baseline,
        "treatment": args.treatment,
        "sample_count": len(rows),
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed": args.seed,
        "metrics": per_metric,
        "created_at": _now_stamp(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines: List[str] = [
        "# Significance Report",
        "",
        f"- Per-query: `{args.per_query}`",
        f"- Baseline: `{args.baseline}`",
        f"- Treatment: `{args.treatment}`",
        f"- Sample count: `{len(rows)}`",
        "",
        "| Metric | Baseline | Treatment | Delta | Rel Delta | 95% CI | p(diff<=0) | Sign-flip p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for metric, stat in per_metric.items():
        ci = stat["bootstrap_ci"]
        rel = stat["rel_delta_vs_baseline"]
        rel_str = "NA" if rel is None else f"{rel * 100:.2f}%"
        md_lines.append(
            "| "
            + f"{metric} | {stat['baseline_mean']:.6f} | {stat['treatment_mean']:.6f} | "
            + f"{stat['abs_delta']:.6f} | {rel_str} | "
            + f"[{ci['low']:.6f}, {ci['high']:.6f}] | {ci['p_diff_le_zero']:.6g} | "
            + f"{stat['paired_signflip_p_two_sided']:.6g} |"
        )
    _write_md(out_md, md_lines)

    print(f"[10a] JSON report: {out_json}")
    print(f"[10a] Markdown report: {out_md}")


if __name__ == "__main__":
    main()

