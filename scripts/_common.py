"""
_common.py — small shared helpers for the bootcamp scripts.

Intentionally dependency-light: only pyyaml + (optionally) huggingface_hub.
Loaded by 01_8b_qlora.py, 02_70b_sft.py, and later-day scripts.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def setup_env_dirs(scratch: str | None = None) -> dict[str, str]:
    scratch = scratch or os.environ.get("SCRATCH", "/scratch/finetune")
    cache = f"{scratch}/cache"
    logs = f"{scratch}/logs"
    out = f"{scratch}/outputs"
    for d in (cache, logs, out):
        Path(d).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache)
    os.environ.setdefault("WANDB_DIR", logs)
    return {"scratch": scratch, "cache": cache, "logs": logs, "out": out}


@dataclass
class WallClockGuard:
    """TrainerCallback-friendly stopper: kills a run after N minutes."""

    max_minutes: float
    started_at: float = 0.0

    def __post_init__(self) -> None:
        self.started_at = time.time()

    def should_stop(self) -> bool:
        return (time.time() - self.started_at) / 60.0 >= self.max_minutes


def make_wallclock_callback(max_minutes: float | None):
    """Return a transformers TrainerCallback that stops training past max_minutes.

    Returns None if max_minutes is falsy so callers can `[cb] if cb else []`.
    """
    if not max_minutes or max_minutes <= 0:
        return None
    from transformers import TrainerCallback

    guard = WallClockGuard(max_minutes=float(max_minutes))

    class _Cb(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if guard.should_stop():
                print(
                    f"[wallclock] hit {max_minutes:.1f} min cap "
                    f"at step {state.global_step}; requesting stop."
                )
                control.should_training_stop = True
            return control

    return _Cb()


def push_to_hub(
    *,
    local_dir: str | Path,
    repo_id: str,
    private: bool = False,
    repo_type: str = "model",
    commit_message: str = "upload",
) -> None:
    """Thin wrapper around huggingface_hub upload_folder with create-if-missing."""
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )


def write_model_card(
    output_dir: str | Path,
    *,
    title: str,
    base_model: str,
    summary: str,
    extras: dict[str, Any] | None = None,
) -> None:
    """Drop a minimal README.md model card so HF repos aren't empty on push."""
    extras = extras or {}
    body = [f"# {title}", "", summary, "", "## Details", ""]
    body.append(f"- Base model: `{base_model}`")
    for k, v in extras.items():
        body.append(f"- {k}: {v}")
    body.append("")
    body.append("Trained as part of the MI300X 50-hour fine-tuning bootcamp.")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "README.md").write_text("\n".join(body))
