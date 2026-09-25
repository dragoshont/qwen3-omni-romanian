"""Shared deterministic and resumable runtime helpers for research training.

The legacy experiment scripts were written as one-shot programs.  This module
keeps the repaired runs small enough to audit while making interruption safe:
adapter parameters, optimizer/scheduler state, sampler position, and every RNG
state are saved together at each checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
try:
    from deterministic_sampler import EpochShuffleSampler
except ModuleNotFoundError:  # Support `import scripts.research_runtime` in audits.
    from .deterministic_sampler import EpochShuffleSampler


def seed_everything(seed: int) -> None:
    """Seed every RNG used by these scripts and request deterministic kernels."""

    if not 0 <= seed < 2**63:
        raise ValueError("training seed must be in [0, 2**63)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> tuple[str, dict[str, str]]:
    root = Path(path)
    members: dict[str, str] = {}
    digest = hashlib.sha256()
    for member in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = member.relative_to(root).as_posix()
        member_hash = sha256_file(member)
        members[relative] = member_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(member_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), members


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "accelerate": package_version("accelerate"),
        "datasets": package_version("datasets"),
        "numpy": np.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True,
        "allow_tf32": False,
    }


def trainable_parameter_manifest(model: torch.nn.Module) -> dict[str, Any]:
    entries = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "tensor_count": len(entries),
        "parameter_count": sum(entry["numel"] for entry in entries),
        "parameters": entries,
    }


def assert_adapter_isolation(
    model: torch.nn.Module,
    *,
    require_mtp: bool,
    allowed_talker_targets: Iterable[str] = ("q_proj", "v_proj"),
    allowed_mtp_targets: Iterable[str] = (
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], dict[str, Any]]:
    """Fail closed if PEFT recursively attaches an unintended trainable tensor."""

    talker_targets = tuple(allowed_talker_targets)
    mtp_targets = tuple(allowed_mtp_targets)
    talker_params: list[torch.nn.Parameter] = []
    mtp_params: list[torch.nn.Parameter] = []
    violations: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" not in name:
            violations.append(f"non-LoRA trainable tensor: {name}")
            continue
        if "code_predictor" in name:
            mtp_params.append(parameter)
            if not any(f".{target}." in name for target in mtp_targets):
                violations.append(f"unexpected MTP target: {name}")
        else:
            talker_params.append(parameter)
            if not any(f".{target}." in name for target in talker_targets):
                violations.append(f"unexpected Talker target: {name}")
    if not talker_params:
        violations.append("no Talker adapter parameters are trainable")
    if require_mtp and not mtp_params:
        violations.append("joint condition has no trainable MTP adapter parameters")
    if not require_mtp and mtp_params:
        violations.append(
            f"Talker-only condition leaked {len(mtp_params)} trainable MTP tensors"
        )
    if violations:
        raise RuntimeError("adapter isolation failed:\n- " + "\n- ".join(violations))
    manifest = trainable_parameter_manifest(model)
    manifest["talker_parameter_count"] = sum(item.numel() for item in talker_params)
    manifest["mtp_parameter_count"] = sum(item.numel() for item in mtp_params)
    return talker_params, mtp_params, manifest


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_resume_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: EpochShuffleSampler,
    completed_step: int,
    run_config: dict[str, Any],
    loss_history: list[dict[str, Any]],
    step_losses: list[float],
    extra_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "format_version": 1,
        "completed_step": completed_step,
        "run_config": run_config,
        "trainable_parameters": trainable,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": _rng_state(),
        "loss_history": loss_history,
        "step_losses": step_losses,
        "extra_state": extra_state or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: EpochShuffleSampler,
    expected_config: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], list[float], dict[str, Any]]:
    """Restore a trusted local checkpoint created by :func:`save_resume_state`."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported resume-state format: {path}")
    if payload.get("run_config") != expected_config:
        raise ValueError("resume-state run configuration does not match this invocation")
    named = dict(model.named_parameters())
    saved = payload["trainable_parameters"]
    current_names = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(saved) != current_names:
        raise ValueError("resume-state trainable parameter names do not match the model")
    with torch.no_grad():
        for name, value in saved.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    _restore_rng_state(payload["rng"])
    return (
        int(payload["completed_step"]),
        list(payload.get("loss_history", [])),
        [float(value) for value in payload.get("step_losses", [])],
        dict(payload.get("extra_state", {})),
    )


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
