#!/usr/bin/env python3
"""Step 9a: Train reranker with LoRA and listwise softmax-CE."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback

from peft import LoraConfig, TaskType, get_peft_model


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _beijing_now_iso() -> str:
    tz_bj = timezone(timedelta(hours=8))
    return datetime.now(tz_bj).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _model_tag(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("/", "-")
        .replace(" ", "-")
        .replace("@", "-")
    )


def _get_git_commit(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head.split(" ", 1)[1].strip()
        ref_path = repo_root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return head


def _normalize_qid(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        return str(value)


def _count_query_ids(records: Sequence[Dict[str, Any]]) -> int:
    qids = set()
    for rec in records:
        qid = _normalize_qid(rec.get("query_id"))
        if qid is not None:
            qids.add(qid)
    return len(qids)


def _neg_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = [len(rec.get("neg_texts") or []) for rec in records]
    if not counts:
        return {"min": 0, "max": 0, "avg": 0, "median": 0, "p75": 0, "p95": 0}
    sorted_counts = sorted(counts)

    def _percentile(p: float) -> int:
        if not sorted_counts:
            return 0
        idx = int(p * (len(sorted_counts) - 1))
        return int(sorted_counts[idx])

    return {
        "min": min(counts),
        "max": max(counts),
        "avg": sum(counts) / len(counts),
        "median": statistics.median(sorted_counts),
        "p75": _percentile(0.75),
        "p95": _percentile(0.95),
    }


def _load_blacklist_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "count": 0,
            "source": "missing",
            "updated_at": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "count": payload.get("count", len(payload.get("query_ids", []))),
        "source": payload.get("source", "unknown"),
        "updated_at": payload.get("updated_at"),
    }


def _resolve_max_length(
    args: argparse.Namespace, cfg: Dict[str, Any], tokenizer: Any
) -> int:
    if args.max_length:
        return int(args.max_length)
    retrieval_cfg = cfg.get("retrieval") or {}
    cfg_len = retrieval_cfg.get("max_seq_length")
    if cfg_len:
        return int(cfg_len)
    model_max = getattr(tokenizer, "model_max_length", None)
    if model_max and model_max < 100000:
        return int(model_max)
    return 512


def _filter_lora_targets(model: torch.nn.Module, targets: Sequence[str]) -> List[str]:
    available = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if last in targets:
            available.add(last)
    return [t for t in targets if t in available]


def _scores_from_logits(logits: torch.Tensor, config: Any) -> torch.Tensor:
    if logits.dim() == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits.view(-1)
    if logits.shape[-1] == 2:
        id2label = getattr(config, "id2label", {}) or {}
        label_map = {str(v).lower(): int(k) for k, v in id2label.items()}
        if "yes" in label_map and "no" in label_map:
            return logits[:, label_map["yes"]] - logits[:, label_map["no"]]
        if "pos" in label_map and "neg" in label_map:
            return logits[:, label_map["pos"]] - logits[:, label_map["neg"]]
        return logits[:, 1]
    return logits[:, 0]


class TripletDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        max_neg: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.max_neg = max_neg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        query = rec["query"]
        pos_text = rec["pos_text"]
        neg_texts = rec.get("neg_texts") or []
        if self.max_neg is not None:
            neg_texts = list(neg_texts)[: self.max_neg]
        return {
            "query": query,
            "pos_text": pos_text,
            "neg_texts": list(neg_texts),
        }


class TripletCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        max_neg: Optional[int],
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_neg = max_neg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pairs: List[Tuple[str, str]] = []
        group_sizes: List[int] = []

        for item in features:
            pos_text = item["pos_text"]
            neg_texts = item.get("neg_texts") or []
            candidates = [pos_text] + list(neg_texts)
            group_sizes.append(len(candidates))
            for cand in candidates:
                pairs.append((item["query"], cand))

        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]
        batch = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["group_sizes"] = group_sizes
        return batch


class ListwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        group_sizes = inputs.pop("group_sizes")
        outputs = model(**inputs)
        scores = _scores_from_logits(outputs.logits, model.config)

        if isinstance(group_sizes, torch.Tensor):
            group_sizes = group_sizes.tolist()

        total = sum(group_sizes)
        if total != scores.shape[0]:
            raise ValueError(
                f"group_sizes sum {total} != scores {scores.shape[0]}"
            )

        losses: List[torch.Tensor] = []
        offset = 0
        for size in group_sizes:
            if size <= 0:
                continue
            group_scores = scores[offset : offset + size]
            offset += size
            loss_i = -torch.log_softmax(group_scores, dim=0)[0]
            losses.append(loss_i)

        if not losses:
            raise ValueError(
                f"No valid groups for loss computation. group_sizes={group_sizes}"
            )
        loss = torch.stack(losses).mean()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        if prediction_loss_only:
            return (loss, None, None)
        logits = outputs.logits.detach()
        return (loss, logits, None)


class BestAdapterSaver(TrainerCallback):
    def __init__(
        self,
        output_dir: Path,
        tokenizer: Any,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric: Optional[float] = None

    def _is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return value > self.best_metric
        return value < self.best_metric

    def _save(self, model, metrics: Dict[str, Any], state: Any) -> None:
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))

        tok_dir = self.output_dir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(tok_dir))

        payload = {
            "metric_name": self.metric_name,
            "best_metric": self.best_metric,
            "metrics": metrics,
            "best_at_step": state.global_step,
            "best_at_epoch": state.epoch,
            "updated_at": _beijing_now_iso(),
        }
        (self.output_dir / "metrics_dev.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return
        metric_value = metrics[self.metric_name]
        if self._is_better(metric_value):
            self.best_metric = metric_value
            model = kwargs.get("model")
            if model is not None:
                self._save(model, metrics, state)


def _load_triplets(
    path: Path,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    raw = _load_jsonl(path)
    if max_samples is not None:
        raw = raw[: max_samples]
    records: List[Dict[str, Any]] = []
    for rec in raw:
        query = str(rec.get("query", "")).strip()
        pos_text = str(rec.get("pos_text", "")).strip()
        if not query or not pos_text:
            continue
        neg_texts = rec.get("neg_texts") or []
        neg_texts = [str(t).strip() for t in neg_texts if str(t).strip()]
        records.append(
            {
                "query_id": rec.get("query_id"),
                "query": query,
                "pos_text": pos_text,
                "neg_texts": neg_texts,
            }
        )
    return records


def _build_data_manifest(
    train_path: Path,
    dev_path: Path,
    train_records: Sequence[Dict[str, Any]],
    dev_records: Sequence[Dict[str, Any]],
    blacklist_meta: Dict[str, Any],
    gold_eval_path: Optional[Path],
) -> Dict[str, Any]:
    return {
        "train_data": str(train_path),
        "dev_data": str(dev_path),
        "train_samples": len(train_records),
        "dev_samples": len(dev_records),
        "train_query_ids": _count_query_ids(train_records),
        "dev_query_ids": _count_query_ids(dev_records),
        "blacklist_file": blacklist_meta.get("path"),
        "blacklist_count": blacklist_meta.get("count"),
        "blacklist_source": blacklist_meta.get("source"),
        "gold_eval_file": str(gold_eval_path) if gold_eval_path else None,
        "training_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "git_commit": _get_git_commit(REPO_ROOT),
        "created_at": _beijing_now_iso(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 9a: train reranker with LoRA and listwise loss."
    )
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument(
        "--config",
        type=str,
        default="config/weak_supervision_config.yaml",
        help="Config path for defaults (max_seq_length).",
    )
    parser.add_argument("--train-data", type=Path, default=None)
    parser.add_argument("--dev-data", type=Path, default=None)
    parser.add_argument(
        "--blacklist",
        type=Path,
        default=Path("config/eval_blacklist.json"),
    )
    parser.add_argument(
        "--gold-eval",
        type=Path,
        default=Path("data/output/annotations/gold_eval_50_extended_final.jsonl"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-neg", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Save HF checkpoints for resume; default saves adapter only.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Write Step9a checkpoint JSON.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint path. If omitted, filename includes model tag + timestamp.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="QAnchor")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    train_path = args.train_data or Path(
        f"data/output/train/train_triplets_{args.stage}.jsonl"
    )
    dev_path = args.dev_data or Path(
        f"data/output/train/dev_triplets_{args.stage}.jsonl"
    )

    train_records = _load_triplets(train_path, args.max_train_samples)
    dev_records = _load_triplets(dev_path, args.max_dev_samples)

    if not train_records:
        raise ValueError(f"Empty train records: {train_path}")
    if not dev_records:
        raise ValueError(f"Empty dev records: {dev_path}")

    blacklist_meta = _load_blacklist_meta(args.blacklist)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = _model_tag(args.model_name)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path("data/output/artifacts/reranker") / model_tag / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    added_tokens = 0
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = _filter_lora_targets(model, args.lora_target_modules)
    removed = [m for m in args.lora_target_modules if m not in target_modules]
    if removed:
        print(f"[LoRA] Dropped missing target modules: {removed}")
    if not target_modules:
        raise ValueError("No valid LoRA target modules found in model.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_length = _resolve_max_length(args, cfg, tokenizer)
    data_collator = TripletCollator(tokenizer, max_length, args.max_neg)

    report_to: List[str] = ["wandb"] if args.wandb else []
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "warmup_ratio": args.warmup_ratio,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.keep_checkpoints,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": report_to,
        "prediction_loss_only": True,
        "remove_unused_columns": False,
    }
    if args.wandb and args.wandb_run_name:
        training_kwargs["run_name"] = args.wandb_run_name
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"
    if "save_strategy" in sig.parameters:
        training_kwargs["save_strategy"] = "steps" if args.keep_checkpoints else "no"
    allowed = set(sig.parameters)
    training_kwargs = {k: v for k, v in training_kwargs.items() if k in allowed}

    training_args = TrainingArguments(**training_kwargs)

    best_saver: Optional[BestAdapterSaver] = None
    if not args.keep_checkpoints:
        best_saver = BestAdapterSaver(output_dir, tokenizer)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": TripletDataset(train_records, args.max_neg),
        "eval_dataset": TripletDataset(dev_records, args.max_neg),
        "data_collator": data_collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    allowed = set(trainer_sig.parameters)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in allowed}

    trainer = ListwiseTrainer(**trainer_kwargs)

    initial_metrics = trainer.evaluate()
    if best_saver:
        trainer.add_callback(best_saver)

    trainer.train()
    metrics = trainer.evaluate()

    metrics_path = output_dir / "metrics_dev.json"
    if not metrics_path.exists():
        metrics_path.write_text(
            json.dumps(
                {
                    "metric_name": "eval_loss",
                    "best_metric": metrics.get("eval_loss"),
                    "metrics": metrics,
                    "best_at_step": trainer.state.global_step,
                    "best_at_epoch": trainer.state.epoch,
                    "updated_at": _beijing_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    should_save_final = True
    if not args.keep_checkpoints and best_saver and best_saver.best_metric is not None:
        should_save_final = False
        print("[Output] Best adapter already saved by BestAdapterSaver; skip final save.")

    if should_save_final:
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))

        tokenizer_dir = output_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_dir))

    (output_dir / "training_args.json").write_text(
        json.dumps(training_args.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    model.config.to_json_file(str(output_dir / "config.json"))

    data_manifest = _build_data_manifest(
        train_path=train_path,
        dev_path=dev_path,
        train_records=train_records,
        dev_records=dev_records,
        blacklist_meta=blacklist_meta,
        gold_eval_path=args.gold_eval if args.gold_eval else None,
    )
    data_manifest["neg_stats"] = _neg_stats(train_records)
    (output_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_checkpoint:
        initial_eval_loss = initial_metrics.get("eval_loss") if initial_metrics else None
        final_eval_loss = metrics.get("eval_loss") if metrics else None
        loss_drop_pct = None
        meets_acceptance = None
        if initial_eval_loss and final_eval_loss:
            loss_drop_pct = (initial_eval_loss - final_eval_loss) / initial_eval_loss * 100
            meets_acceptance = loss_drop_pct >= 20

        checkpoint = {
            "stage": args.stage,
            "step": 9,
            "completed": True,
            "completed_at": _beijing_now_iso(),
            "model_name": args.model_name,
            "lora_config": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": target_modules,
            },
            "training_args": {
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_epochs,
                "per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_ratio": args.warmup_ratio,
                "eval_steps": args.eval_steps,
                "save_steps": args.save_steps,
                "keep_checkpoints": args.keep_checkpoints,
                "max_length": max_length,
                "max_neg": args.max_neg,
            },
            "best_metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            "artifacts_dir": str(output_dir),
            "data_manifest": data_manifest,
            "validation": {
                "initial_eval_loss": initial_eval_loss,
                "final_eval_loss": final_eval_loss,
                "loss_drop_percentage": loss_drop_pct,
                "meets_acceptance": meets_acceptance,
            },
        }
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = Path(
                f"data/output/checkpoints/{args.stage}_step_9a_train_reranker_{model_tag}_{run_id}.json"
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[Output] Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
