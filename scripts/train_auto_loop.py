#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic experiment loop for the mini GPT sentiment classifier."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sentiment_experiment import DEFAULT_CONFIG, make_run_id, run_experiment, utc_now

DOCS_DIR = REPO_ROOT / "docs" / "train"
RUNS_DIR = DOCS_DIR / "runs"
LOCK_PATH = DOCS_DIR / "train.lock"
STATUS_PATH = DOCS_DIR / "status.md"
QUEUE_PATH = DOCS_DIR / "experiment_queue.json"
HISTORY_PATH = DOCS_DIR / "experiment_history.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def ensure_docs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not QUEUE_PATH.exists():
        write_json(QUEUE_PATH, [])
    if not HISTORY_PATH.exists():
        write_json(HISTORY_PATH, [])
    if not STATUS_PATH.exists():
        write_status("idle", "No experiment has been started yet.")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_lock() -> dict[str, Any] | None:
    if not LOCK_PATH.exists():
        return None

    lock = read_json(LOCK_PATH, {})
    pid = lock.get("pid")
    if isinstance(pid, int) and is_pid_alive(pid):
        return lock

    LOCK_PATH.unlink(missing_ok=True)
    return None


def write_lock(run_id: str, config: dict[str, Any]) -> None:
    write_json(
        LOCK_PATH,
        {
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": utc_now(),
            "config": config,
        },
    )


def clear_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def write_status(state: str, message: str, extra: dict[str, Any] | None = None) -> None:
    extra = extra or {}
    extra_block = json.dumps(extra, ensure_ascii=False, indent=2)
    content = f"""# Training Automation Status

- updated_at: {utc_now()}
- state: {state}
- message: {message}

```json
{extra_block}
```
"""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(content, encoding="utf-8")


def append_history(result: dict[str, Any]) -> None:
    history = read_json(HISTORY_PATH, [])
    history.append(
        {
            "run_id": result["run_id"],
            "parent_run_id": result.get("parent_run_id"),
            "started_at": result["started_at"],
            "finished_at": result["finished_at"],
            "fit_status": result["analysis"]["fit_status"],
            "overfit_score": result["analysis"]["overfit_score"],
            "best_val_loss": result["analysis"]["best_val_loss"],
            "best_val_acc": result["analysis"]["best_val_acc"],
            "final_train_loss": result["analysis"]["final_train_loss"],
            "final_val_loss": result["analysis"]["final_val_loss"],
            "generalization_gap": result["analysis"]["generalization_gap"],
            "acc_gap": result["analysis"]["acc_gap"],
            "config": result["config"],
            "tokenizer_path": result["tokenizer_path"],
            "latest_checkpoint_path": result["latest_checkpoint_path"],
            "best_checkpoint_path": result["best_checkpoint_path"],
            "next_hypothesis": result["next_hypothesis"],
            "metric_graph": result.get("metric_graph"),
            "branch_graph": result.get("branch_graph"),
            "branch_events": result.get("branch_events", []),
            "selected_branch_config": result.get("selected_branch_config"),
        }
    )
    write_json(HISTORY_PATH, history)


def drop_rate_candidate(base: dict[str, Any]) -> dict[str, Any] | None:
    current_drop = float(base["drop_rate"])
    next_drop = min(current_drop + 0.1, 0.5)
    if next_drop == current_drop:
        return None
    candidate = dict(base)
    candidate["drop_rate"] = round(next_drop, 3)
    candidate["candidate_type"] = "drop_rate_only"
    candidate["hypothesis"] = "Overfit: increase dropout while keeping depth fixed."
    return candidate


def n_layers_candidate(base: dict[str, Any]) -> dict[str, Any] | None:
    current_layers = int(base["n_layers"])
    if current_layers <= 1:
        return None
    candidate = dict(base)
    candidate["n_layers"] = current_layers - 1
    candidate["candidate_type"] = "n_layers_only"
    candidate["hypothesis"] = "Overfit: reduce transformer depth while keeping dropout fixed."
    return candidate


def combined_candidate(base: dict[str, Any]) -> dict[str, Any] | None:
    current_drop = float(base["drop_rate"])
    current_layers = int(base["n_layers"])
    next_drop = min(current_drop + 0.1, 0.5)
    if current_layers <= 1 or next_drop == current_drop:
        return None
    candidate = dict(base)
    candidate["drop_rate"] = round(next_drop, 3)
    candidate["n_layers"] = current_layers - 1
    candidate["candidate_type"] = "drop_rate_and_n_layers"
    candidate["hypothesis"] = "Overfit: increase dropout and reduce depth from the save point."
    return candidate


def generate_overfit_candidates(last_run: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(last_run["config"])
    base.pop("run_id", None)
    base["parent_run_id"] = last_run["run_id"]
    base["parent_checkpoint_path"] = last_run.get("best_checkpoint_path")
    base["parent_tokenizer_path"] = last_run.get("tokenizer_path")

    candidates = []
    for builder in (drop_rate_candidate, n_layers_candidate, combined_candidate):
        candidate = builder(base)
        if candidate is not None:
            candidates.append(candidate)
        if len(candidates) >= 3:
            break
    return candidates


def load_initial_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_CONFIG)
    config = dict(DEFAULT_CONFIG)
    config.update(read_json(path, {}))
    return config


def pick_next_experiment(initial_config_path: Path | None) -> dict[str, Any] | None:
    queue = read_json(QUEUE_PATH, [])
    history = read_json(HISTORY_PATH, [])

    if queue:
        next_config = queue.pop(0)
        write_json(QUEUE_PATH, queue)
        return next_config

    if not history:
        return load_initial_config(initial_config_path)

    last_run = history[-1]
    if last_run.get("fit_status") != "overfit":
        write_status(
            "idle",
            "Last run is not overfit, so no automatic candidate was generated.",
            {"last_run": last_run},
        )
        return None

    candidates = generate_overfit_candidates(last_run)
    write_json(QUEUE_PATH, candidates)
    if not candidates:
        write_status(
            "idle",
            "Last run was overfit, but no valid drop_rate/n_layers candidate remains.",
            {"last_run": last_run},
        )
        return None

    next_config = candidates.pop(0)
    write_json(QUEUE_PATH, candidates)
    return next_config


def run_one_cycle(initial_config_path: Path | None) -> dict[str, Any] | None:
    ensure_docs()
    lock = live_lock()
    if lock is not None:
        write_status(
            "running",
            "Another training process is still running; status only.",
            lock,
        )
        return None

    config = pick_next_experiment(initial_config_path)
    if config is None:
        return None

    run_id = config.get("run_id") or make_run_id()
    config["run_id"] = run_id
    write_lock(run_id, config)
    write_status("running", "Started an experiment.", {"run_id": run_id, "config": config})

    try:
        result = run_experiment(config=config, docs_dir=DOCS_DIR)
        append_history(result)
        selected_branch_config = result.get("selected_branch_config")
        if selected_branch_config is not None:
            queue = read_json(QUEUE_PATH, [])
            queue.insert(0, selected_branch_config)
            write_json(QUEUE_PATH, queue)
        write_status(
            "finished",
            "Experiment completed.",
            {
                "run_id": result["run_id"],
                "fit_status": result["analysis"]["fit_status"],
                "overfit_score": result["analysis"]["overfit_score"],
                "selected_branch_config": selected_branch_config,
                "branch_graph": result.get("branch_graph"),
                "next_hypothesis": result["next_hypothesis"],
            },
        )
        return result
    except Exception as exc:  # noqa: BLE001
        write_status(
            "failed",
            "Experiment failed.",
            {"run_id": run_id, "error": str(exc)},
        )
        raise
    finally:
        clear_lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--interval-minutes", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        result = run_one_cycle(args.config_json)
        if args.once:
            break
        if result is None:
            time.sleep(max(args.interval_minutes, 0.1) * 60)
        else:
            time.sleep(max(args.interval_minutes, 0.1) * 60)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
