#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one sentiment fine-tuning experiment and write train docs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bpe import BPETokenizer
from finetune import (
    GPTForSequenceClassification,
    ReviewSentimentDataset,
    evaluate_sentiment,
    train_epoch_sentiment,
)
from model import GPTModel


DEFAULT_CONFIG: dict[str, Any] = {
    "train_path": "data/ratings_train.txt",
    "val_path": None,
    "val_ratio": 0.08,
    "seed": 42,
    "corpus_size": 500_000,
    "train_data_size": 50_000,
    "val_data_size": 4_000,
    "vocab_size": 2_000,
    "context_length": 128,
    "max_length": 128,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 3,
    "drop_rate": 0.2,
    "qkv_bias": False,
    "activation_name": "gelu",
    "batch_size": 256,
    "num_workers": 4,
    "epoch_num": 6,
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "parent_run_id": None,
    "parent_checkpoint_path": None,
    "parent_tokenizer_path": None,
    "branch_on_epoch_overfit": True,
    "branch_probe_epochs": 1,
    "branch_max_events": 3,
    "overfit_generalization_gap_threshold": 0.15,
    "overfit_acc_gap_threshold": 0.08,
}


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def make_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")

    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = str(item.get("text", "")).strip()
                label = item.get("label")
                if text and label in {0, 1, "0", "1"}:
                    rows.append({"text": text, "label": int(label)})
        return rows

    rows = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            text = str(row.get("document", "")).strip()
            label = row.get("label")
            if text and label in {"0", "1"}:
                rows.append({"text": text, "label": int(label)})
    return rows


def split_train_val(
    train_path: Path,
    val_path: Path | None,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_path is not None:
        return read_rows(train_path), read_rows(val_path)

    rows = read_rows(train_path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * val_ratio))
    return rows[:-val_size], rows[-val_size:]


def truncate_rows(rows: list[dict[str, Any]], size: int | None) -> list[dict[str, Any]]:
    if size is None or size <= 0:
        return rows
    return rows[:size]


def build_corpus(rows: list[dict[str, Any]], char_limit: int) -> str:
    texts: list[str] = []
    total_chars = 0
    for row in rows:
        text = row["text"]
        if total_chars >= char_limit:
            break
        texts.append(text)
        total_chars += len(text) + 1
    return "\n".join(texts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def runtime_info(device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "batch_size": config["batch_size"],
        "max_length": config["max_length"],
        "context_length": config["context_length"],
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "drop_rate": config["drop_rate"],
    }


def model_config_from(config: dict[str, Any], tokenizer: BPETokenizer) -> dict[str, Any]:
    return {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": config["context_length"],
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "drop_rate": config["drop_rate"],
        "qkv_bias": config["qkv_bias"],
        "activation_name": config.get("activation_name", "gelu"),
    }


def torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def shape_keys(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config.get("vocab_size"),
        config.get("context_length"),
        config.get("emb_dim"),
        config.get("n_heads"),
        config.get("n_layers"),
        config.get("qkv_bias"),
    )


def load_checkpoint_for_experiment(
    model: GPTForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path | None,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"mode": "fresh", "loaded_model_keys": 0, "loaded_optimizer": False}

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    checkpoint_config = checkpoint.get("config", {})
    model_state = checkpoint["model_state_dict"]
    load_info: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_run_id": checkpoint.get("run_id"),
        "loaded_optimizer": False,
    }

    if shape_keys(checkpoint_config) == shape_keys(config):
        model.load_state_dict(model_state)
        load_info["mode"] = "strict"
        load_info["loaded_model_keys"] = len(model_state)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            load_info["loaded_optimizer"] = True
        except Exception as exc:  # noqa: BLE001
            load_info["optimizer_error"] = str(exc)
        return load_info

    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in model_state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    current_state.update(compatible_state)
    model.load_state_dict(current_state, strict=False)
    load_info["mode"] = "partial"
    load_info["loaded_model_keys"] = len(compatible_state)
    load_info["skipped_model_keys"] = len(model_state) - len(compatible_state)
    load_info["optimizer_note"] = "optimizer state was not reused after shape change"
    return load_info


def save_checkpoint(
    path: Path,
    model: GPTForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    tokenizer_path: Path,
    metrics: dict[str, list[float]],
    run_id: str,
    parent_run_id: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "tokenizer_path": str(tokenizer_path),
            "metrics": metrics,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
        },
        path,
    )


def increasing_twice(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    return values[-1] > values[-2] and values[-2] > values[-3]


def overfit_thresholds(config: dict[str, Any]) -> dict[str, float]:
    return {
        "generalization_gap": float(
            config.get("overfit_generalization_gap_threshold", 0.15)
        ),
        "acc_gap": float(config.get("overfit_acc_gap_threshold", 0.08)),
    }


def analyze_metrics(metrics: dict[str, list[float]]) -> dict[str, Any]:
    train_losses = metrics["train_loss"]
    val_losses = metrics["val_loss"]
    train_accs = metrics["train_acc"]
    val_accs = metrics["val_acc"]

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    generalization_gap = final_val_loss - final_train_loss
    acc_gap = final_train_acc - final_val_acc
    overfit_score = max(0.0, generalization_gap) + max(0.0, acc_gap)

    train_loss_decreasing = len(train_losses) >= 2 and final_train_loss < train_losses[0]
    val_loss_increasing = increasing_twice(val_losses)

    if (
        (train_loss_decreasing and val_loss_increasing)
        or generalization_gap > 0.15
        or acc_gap > 0.08
    ):
        fit_status = "overfit"
    elif (
        final_train_loss > 0.65
        and final_val_loss > 0.65
        and final_train_acc < 0.65
        and final_val_acc < 0.65
    ):
        fit_status = "underfit"
    elif (
        final_val_loss <= val_losses[0]
        and generalization_gap <= 0.15
        and acc_gap <= 0.08
    ):
        fit_status = "good_fit"
    else:
        fit_status = "unstable"

    return {
        "best_val_loss": min(val_losses),
        "best_val_acc": max(val_accs),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "generalization_gap": generalization_gap,
        "acc_gap": acc_gap,
        "overfit_score": overfit_score,
        "fit_status": fit_status,
        "val_loss_increasing_twice": val_loss_increasing,
        "train_loss_decreasing": train_loss_decreasing,
    }


def analyze_metrics_with_config(
    metrics: dict[str, list[float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    analysis = analyze_metrics(metrics)
    thresholds = overfit_thresholds(config)

    if (
        (analysis["train_loss_decreasing"] and analysis["val_loss_increasing_twice"])
        or analysis["generalization_gap"] > thresholds["generalization_gap"]
        or analysis["acc_gap"] > thresholds["acc_gap"]
    ):
        analysis["fit_status"] = "overfit"
    elif (
        analysis["final_train_loss"] > 0.65
        and analysis["final_val_loss"] > 0.65
        and analysis["final_train_acc"] < 0.65
        and analysis["final_val_acc"] < 0.65
    ):
        analysis["fit_status"] = "underfit"
    elif (
        analysis["final_val_loss"] <= metrics["val_loss"][0]
        and analysis["generalization_gap"] <= thresholds["generalization_gap"]
        and analysis["acc_gap"] <= thresholds["acc_gap"]
    ):
        analysis["fit_status"] = "good_fit"
    else:
        analysis["fit_status"] = "unstable"

    analysis["thresholds"] = thresholds
    return analysis


def next_hypothesis(analysis: dict[str, Any]) -> str:
    status = analysis["fit_status"]
    if status == "overfit":
        return "Increase dropout, reduce depth, or try both from the best save point."
    if status == "underfit":
        return "Model is not fitting train data yet; inspect learning rate, data size, and model capacity."
    if status == "good_fit":
        return "Keep this run as a candidate final model and avoid using test data until final selection."
    return "Metrics are mixed; repeat with the same seed or inspect the loss curve before changing capacity."


def branch_score(result: dict[str, Any]) -> float:
    analysis = result["analysis"]
    return float(analysis["best_val_acc"]) - float(analysis["overfit_score"])


def make_branch_candidates(
    base_config: dict[str, Any],
    parent_run_id: str,
    checkpoint_path: Path,
    tokenizer_path: Path,
    epoch: int,
) -> list[dict[str, Any]]:
    base = dict(base_config)
    base.pop("run_id", None)
    base["parent_run_id"] = parent_run_id
    base["parent_checkpoint_path"] = str(checkpoint_path)
    base["parent_tokenizer_path"] = str(tokenizer_path)
    base["epoch_num"] = int(base.get("branch_probe_epochs", 1))
    base["branch_on_epoch_overfit"] = False

    current_drop = float(base["drop_rate"])
    next_drop = min(current_drop + 0.1, 0.5)
    current_layers = int(base["n_layers"])
    candidates: list[dict[str, Any]] = []

    if next_drop != current_drop:
        candidate = dict(base)
        candidate["drop_rate"] = round(next_drop, 3)
        candidate["candidate_type"] = "drop_rate_only"
        candidate["run_id"] = f"{parent_run_id}_e{epoch + 1}_drop"
        candidates.append(candidate)

    if current_layers > 1:
        candidate = dict(base)
        candidate["n_layers"] = current_layers - 1
        candidate["candidate_type"] = "n_layers_only"
        candidate["run_id"] = f"{parent_run_id}_e{epoch + 1}_layers"
        candidates.append(candidate)

    if next_drop != current_drop and current_layers > 1:
        candidate = dict(base)
        candidate["drop_rate"] = round(next_drop, 3)
        candidate["n_layers"] = current_layers - 1
        candidate["candidate_type"] = "drop_rate_and_n_layers"
        candidate["run_id"] = f"{parent_run_id}_e{epoch + 1}_both"
        candidates.append(candidate)

    return candidates[:3]


def write_metrics_graph(result: dict[str, Any], runs_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    run_id = result["run_id"]
    metrics = result["metrics"]
    epochs = list(range(1, len(metrics["train_loss"]) + 1))
    if not epochs:
        return None

    graph_path = runs_dir / f"{run_id}_metrics.png"
    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], marker="o", label="train loss")
    plt.plot(epochs, metrics["val_loss"], marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.25)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["train_acc"], marker="o", label="train acc")
    plt.plot(epochs, metrics["val_acc"], marker="o", label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.xticks(epochs)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()

    plt.suptitle(f"Run {run_id} Metrics")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=180, bbox_inches="tight")
    plt.close()
    return graph_path.name


def write_branch_graph(result: dict[str, Any], runs_dir: Path) -> str | None:
    branch_events = result.get("branch_events", [])
    if not branch_events:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    run_id = result["run_id"]
    graph_path = runs_dir / f"{run_id}_branch_points.png"

    plt.figure(figsize=(10, 6))
    best_x: list[float] = []
    best_y: list[float] = []

    for event_index, event in enumerate(branch_events, start=1):
        candidates = event["candidates"]
        selected = event["selected_candidate"]
        xs = [float(item["analysis"]["overfit_score"]) for item in candidates]
        ys = [float(item["analysis"]["best_val_acc"]) for item in candidates]

        plt.scatter(xs, ys, s=90, alpha=0.75, label=f"epoch {event['epoch'] + 1} candidates")
        mid_x = (min(xs) + max(xs)) / 2
        left_label_count = 0
        right_label_count = 0
        for item, x_value, y_value in zip(candidates, xs, ys):
            cfg = item["config"]
            label = (
                f"{item['candidate_type']}\n"
                f"drop={cfg['drop_rate']}, layers={cfg['n_layers']}\n"
                f"acc={y_value:.3f}, score={item['selection_score']:.3f}"
            )
            if x_value > mid_x:
                x_offset = -165
                y_offset = 18 + right_label_count * 56
                right_label_count += 1
            else:
                x_offset = 10
                y_offset = 18 + left_label_count * 42
                left_label_count += 1
            plt.annotate(
                label,
                (x_value, y_value),
                textcoords="offset points",
                xytext=(x_offset, y_offset),
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.78},
                arrowprops={"arrowstyle": "-", "alpha": 0.35},
                annotation_clip=False,
            )

        selected_x = float(selected["analysis"]["overfit_score"])
        selected_y = float(selected["analysis"]["best_val_acc"])
        best_x.append(selected_x)
        best_y.append(selected_y)
        plt.scatter(
            [selected_x],
            [selected_y],
            s=220,
            marker="*",
            edgecolor="black",
            linewidth=1.0,
            label=f"selected {event_index}",
        )

    if len(best_x) >= 2:
        plt.plot(best_x, best_y, linewidth=2.5, marker="o", label="selected path")
    else:
        plt.plot(best_x, best_y, linewidth=2.5, marker="o", label="selected point")

    all_x = [
        float(item["analysis"]["overfit_score"])
        for event in branch_events
        for item in event["candidates"]
    ]
    all_y = [
        float(item["analysis"]["best_val_acc"])
        for event in branch_events
        for item in event["candidates"]
    ]
    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)
    x_margin = max(0.001, x_span * 0.3)
    y_margin = max(0.03, y_span * 0.8)
    plt.xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    plt.ylim(max(0.0, min(all_y) - y_margin), min(1.0, max(all_y) + y_margin))

    plt.xlabel("Overfit score lower is better")
    plt.ylabel("Best validation accuracy higher is better")
    plt.title("Overfit Branch Candidates and Selected Path")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path, dpi=180, bbox_inches="tight")
    plt.close()
    return graph_path.name


def write_run_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    analysis = result["analysis"]
    config = result["config"]
    runtime = result["runtime"]

    rows = [
        "| epoch | train_loss | val_loss | train_acc | val_acc |",
        "|---:|---:|---:|---:|---:|",
    ]
    for idx in range(len(metrics["train_loss"])):
        rows.append(
            "| "
            f"{idx + 1} | "
            f"{metrics['train_loss'][idx]:.4f} | "
            f"{metrics['val_loss'][idx]:.4f} | "
            f"{metrics['train_acc'][idx]:.4f} | "
            f"{metrics['val_acc'][idx]:.4f} |"
        )

    config_block = json.dumps(config, ensure_ascii=False, indent=2)
    runtime_block = json.dumps(runtime, ensure_ascii=False, indent=2)
    load_block = json.dumps(result["checkpoint_load"], ensure_ascii=False, indent=2)
    selected_branch_block = json.dumps(
        result.get("selected_branch_config"),
        ensure_ascii=False,
        indent=2,
    )
    metric_graph = result.get("metric_graph")
    branch_graph = result.get("branch_graph")

    graph_section = ""
    if metric_graph:
        graph_section += f"\n## Metric Graph\n\n![metric graph]({metric_graph})\n"
    if branch_graph:
        graph_section += f"\n## Branch Decision Graph\n\n![branch graph]({branch_graph})\n"

    branch_section = ""
    if result.get("branch_events"):
        branch_rows = [
            "| event_epoch | candidate | drop_rate | n_layers | val_acc | overfit_score | selection_score | selected |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
        for event in result["branch_events"]:
            selected_run_id = event["selected_candidate"]["run_id"]
            for candidate in event["candidates"]:
                cfg = candidate["config"]
                branch_rows.append(
                    "| "
                    f"{event['epoch'] + 1} | "
                    f"{candidate['candidate_type']} | "
                    f"{cfg['drop_rate']} | "
                    f"{cfg['n_layers']} | "
                    f"{candidate['analysis']['best_val_acc']:.4f} | "
                    f"{candidate['analysis']['overfit_score']:.4f} | "
                    f"{candidate['selection_score']:.4f} | "
                    f"{'yes' if candidate['run_id'] == selected_run_id else ''} |"
                )
        branch_section = f"""
## Branch Candidates

{os.linesep.join(branch_rows)}

## Selected Branch Config

```json
{selected_branch_block}
```
"""

    content = f"""# Run {result['run_id']}

- started_at: {result['started_at']}
- finished_at: {result['finished_at']}
- parent_run_id: {result.get('parent_run_id')}
- fit_status: {analysis['fit_status']}
- overfit_score: {analysis['overfit_score']:.4f}
- best_val_loss: {analysis['best_val_loss']:.4f}
- best_val_acc: {analysis['best_val_acc']:.4f}
- generalization_gap: {analysis['generalization_gap']:.4f}
- acc_gap: {analysis['acc_gap']:.4f}

## Config

```json
{config_block}
```

## Runtime

```json
{runtime_block}
```

## Checkpoint Load

```json
{load_block}
```

## Metrics

{os.linesep.join(rows)}
{graph_section}
{branch_section}

## Next Hypothesis

{result['next_hypothesis']}
"""
    path.write_text(content, encoding="utf-8")


def run_experiment(
    config: dict[str, Any] | None = None,
    docs_dir: Path | str = REPO_ROOT / "docs" / "train",
) -> dict[str, Any]:
    merged_config = dict(DEFAULT_CONFIG)
    if config:
        merged_config.update(config)

    run_id = merged_config.get("run_id") or make_run_id()
    parent_run_id = merged_config.get("parent_run_id")
    docs_path = resolve_path(docs_dir) or (REPO_ROOT / "docs" / "train")
    runs_dir = docs_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    start_time = time.perf_counter()
    seed = int(merged_config["seed"])
    set_seed(seed)

    train_path = resolve_path(merged_config["train_path"])
    val_path = resolve_path(merged_config.get("val_path"))
    if train_path is None:
        raise ValueError("train_path is required")

    train_rows, val_rows = split_train_val(
        train_path=train_path,
        val_path=val_path,
        val_ratio=float(merged_config["val_ratio"]),
        seed=seed,
    )
    train_rows = truncate_rows(train_rows, merged_config.get("train_data_size"))
    val_rows = truncate_rows(val_rows, merged_config.get("val_data_size"))
    if not train_rows:
        raise ValueError("train data is empty")
    if not val_rows:
        raise ValueError("val data is empty")
    if int(merged_config["max_length"]) > int(merged_config["context_length"]):
        raise ValueError("max_length must be <= context_length")

    tokenizer_path = runs_dir / f"{run_id}_tokenizer.json"
    parent_tokenizer_path = resolve_path(merged_config.get("parent_tokenizer_path"))
    tokenizer = BPETokenizer(vocab_size=int(merged_config["vocab_size"]))
    if parent_tokenizer_path is not None:
        tokenizer.load(parent_tokenizer_path)
        tokenizer_path.write_text(parent_tokenizer_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        corpus = build_corpus(train_rows, int(merged_config["corpus_size"]))
        tokenizer.train(corpus)
        tokenizer.save(tokenizer_path)

    model_config = model_config_from(merged_config, tokenizer)
    merged_config["vocab_size"] = model_config["vocab_size"]

    train_ds = ReviewSentimentDataset(
        train_rows,
        tokenizer,
        max_length=int(merged_config["max_length"]),
    )
    val_ds = ReviewSentimentDataset(
        val_rows,
        tokenizer,
        max_length=int(merged_config["max_length"]),
    )

    batch_size = int(merged_config["batch_size"])
    num_workers = int(merged_config["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = select_device()
    runtime = runtime_info(device, merged_config)
    model = GPTForSequenceClassification(
        GPTModel(model_config),
        drop_rate=float(merged_config["drop_rate"]),
    ).to(device)
    runtime["model_device"] = str(next(model.parameters()).device)
    runtime["train_rows"] = len(train_rows)
    runtime["val_rows"] = len(val_rows)
    runtime["model_params"] = sum(p.numel() for p in model.parameters())
    print(
        f"[{run_id}] start device={runtime['model_device']} "
        f"train={runtime['train_rows']} val={runtime['val_rows']} "
        f"epochs={merged_config['epoch_num']} batch={merged_config['batch_size']}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(merged_config["learning_rate"]),
        weight_decay=float(merged_config["weight_decay"]),
    )

    checkpoint_load = load_checkpoint_for_experiment(
        model=model,
        optimizer=optimizer,
        checkpoint_path=resolve_path(merged_config.get("parent_checkpoint_path")),
        config=model_config,
        device=device,
    )

    metrics: dict[str, list[float]] = {
        "train_epoch_loss": [],
        "train_epoch_acc": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    latest_checkpoint_path = runs_dir / f"{run_id}_latest.pt"
    best_checkpoint_path = runs_dir / f"{run_id}_best.pt"
    best_val_loss = float("inf")
    branch_events: list[dict[str, Any]] = []

    for epoch in range(int(merged_config["epoch_num"])):
        print(f"[{run_id}] epoch {epoch + 1} start", flush=True)
        train_epoch_loss, train_epoch_acc = train_epoch_sentiment(
            model,
            train_loader,
            optimizer,
            device,
        )
        train_loss, train_acc = evaluate_sentiment(model, train_eval_loader, device)
        val_loss, val_acc = evaluate_sentiment(model, val_loader, device)

        metrics["train_epoch_loss"].append(float(train_epoch_loss))
        metrics["train_epoch_acc"].append(float(train_epoch_acc))
        metrics["train_loss"].append(float(train_loss))
        metrics["train_acc"].append(float(train_acc))
        metrics["val_loss"].append(float(val_loss))
        metrics["val_acc"].append(float(val_acc))
        print(
            f"[{run_id}] epoch {epoch + 1} "
            f"train={train_epoch_loss:.4f}/{train_epoch_acc:.4f} "
            f"train_eval={train_loss:.4f}/{train_acc:.4f} "
            f"val={val_loss:.4f}/{val_acc:.4f}",
            flush=True,
        )

        epoch_checkpoint_path = runs_dir / f"{run_id}_epoch_{epoch + 1}.pt"
        save_checkpoint(
            epoch_checkpoint_path,
            model,
            optimizer,
            epoch,
            model_config,
            tokenizer_path,
            metrics,
            run_id,
            parent_run_id,
        )
        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            epoch,
            model_config,
            tokenizer_path,
            metrics,
            run_id,
            parent_run_id,
        )
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                model_config,
                tokenizer_path,
                metrics,
                run_id,
                parent_run_id,
            )

        epoch_analysis = analyze_metrics_with_config(metrics, merged_config)
        if (
            merged_config.get("branch_on_epoch_overfit", True)
            and epoch_analysis["fit_status"] == "overfit"
            and len(branch_events) < int(merged_config.get("branch_max_events", 3))
        ):
            candidates = make_branch_candidates(
                merged_config,
                run_id,
                epoch_checkpoint_path,
                tokenizer_path,
                epoch,
            )
            candidate_results = []
            for candidate_config in candidates:
                print(
                    f"[{run_id}] branch {candidate_config['candidate_type']} "
                    f"drop={candidate_config['drop_rate']} "
                    f"layers={candidate_config['n_layers']}",
                    flush=True,
                )
                candidate_result = run_experiment(candidate_config, docs_dir=docs_path)
                candidate_summary = {
                    "run_id": candidate_result["run_id"],
                    "candidate_type": candidate_config["candidate_type"],
                    "config": candidate_result["config"],
                    "analysis": candidate_result["analysis"],
                    "best_checkpoint_path": candidate_result["best_checkpoint_path"],
                    "tokenizer_path": candidate_result["tokenizer_path"],
                    "selection_score": branch_score(candidate_result),
                }
                candidate_results.append(candidate_summary)

            if candidate_results:
                selected_candidate = max(
                    candidate_results,
                    key=lambda item: item["selection_score"],
                )
                branch_events.append(
                    {
                        "epoch": epoch,
                        "trigger_analysis": epoch_analysis,
                        "checkpoint_path": str(epoch_checkpoint_path),
                        "candidates": candidate_results,
                        "selected_candidate": selected_candidate,
                    }
                )
                selected_config = dict(selected_candidate["config"])
                selected_config.pop("run_id", None)
                selected_config["parent_run_id"] = selected_candidate["run_id"]
                selected_config["parent_checkpoint_path"] = selected_candidate[
                    "best_checkpoint_path"
                ]
                selected_config["parent_tokenizer_path"] = selected_candidate[
                    "tokenizer_path"
                ]
                selected_config["branch_on_epoch_overfit"] = True
                merged_config["selected_branch_config"] = selected_config
                break

    analysis = analyze_metrics_with_config(metrics, merged_config)
    finished_at = utc_now()
    result = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": time.perf_counter() - start_time,
        "config": merged_config,
        "model_config": model_config,
        "runtime": runtime,
        "checkpoint_load": checkpoint_load,
        "metrics": metrics,
        "analysis": analysis,
        "next_hypothesis": next_hypothesis(analysis),
        "branch_events": branch_events,
        "selected_branch_config": merged_config.get("selected_branch_config"),
        "tokenizer_path": str(tokenizer_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
    }

    result["metric_graph"] = write_metrics_graph(result, runs_dir)
    result["branch_graph"] = write_branch_graph(result, runs_dir)

    write_json(runs_dir / f"{run_id}.json", result)
    write_run_markdown(runs_dir / f"{run_id}.md", result)
    return result


def load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return read_json(path, {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs" / "train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_file(args.config_json)
    result = run_experiment(config=config, docs_dir=args.docs_dir)
    print(
        f"run_id={result['run_id']} "
        f"fit_status={result['analysis']['fit_status']} "
        f"best_val_acc={result['analysis']['best_val_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
