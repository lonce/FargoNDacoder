from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _to_serializable(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_run_config(
    output_dir: str | Path,
    params: dict,
    model_config,
    data_config,
    filename_stem: str = "run_config",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "rnndac_run_config_v1",
        "params": _to_serializable(params),
        "model_config": _to_serializable(model_config),
        "data_config": _to_serializable(data_config),
    }

    pt_path = output_dir / f"{filename_stem}.pt"
    json_path = output_dir / f"{filename_stem}.json"

    torch.save(payload, pt_path)

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return pt_path, json_path


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model,
    optimizer=None,
    params: dict | None = None,
    model_config=None,
    data_config=None,
    extra: dict | None = None,
):
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "rnndac_checkpoint_v1",
        "step": step,
        "model_state_dict": model.state_dict(),
        "model_config": _to_serializable(model_config if model_config is not None else getattr(model, "config", None)),
        "data_config": _to_serializable(data_config),
        "params": _to_serializable(params),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        payload["extra"] = _to_serializable(extra)

    ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_run_config(config_path: str | Path):
    return torch.load(config_path, weights_only=False, map_location="cpu")


def load_checkpoint(checkpoint_path: str | Path, map_location="cpu"):
    return torch.load(checkpoint_path, weights_only=False, map_location=map_location)