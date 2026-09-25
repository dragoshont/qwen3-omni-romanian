"""Run or resume every preregistered confirmatory training condition."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "confirmatory_matrix.json"
STATUS = ROOT / "reports" / "confirmatory_training_status.json"
LOG_DIR = ROOT / "logs" / "confirmatory_training"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def complete(condition: dict, seed: int) -> bool:
    report_path = ROOT / "reports" / f"{condition['name']}_seed{seed}_training.json"
    adapter_path = ROOT / "models" / "controlled" / condition["name"] / f"seed_{seed}"
    if not report_path.exists() or not (adapter_path / "adapter_model.safetensors").exists():
        return False
    try:
        report = load_json(report_path)
        return (
            report.get("run_name") == condition["name"]
            and report.get("training_seed") == seed
            and report.get("steps") == condition["steps"]
            and report.get("run_config", {}).get("fixed_endpoint", True)
        )
    except (OSError, json.JSONDecodeError):
        return False


def write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run_and_tee(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    matrix = load_json(CONFIG)
    jobs = [
        (condition, seed)
        for condition in matrix["conditions"]
        for seed in matrix["training_seeds"]
    ]
    state = {
        "protocol_version": matrix["protocol_version"],
        "started_at_utc": now(),
        "updated_at_utc": now(),
        "status": "running",
        "total_jobs": len(jobs),
        "completed_jobs": [],
        "current_job": None,
    }
    write_status(state)
    for condition, seed in jobs:
        job_id = f"{condition['name']}:seed{seed}"
        if complete(condition, seed):
            print(f"SKIP complete {job_id}", flush=True)
            state["completed_jobs"].append(job_id)
            state["updated_at_utc"] = now()
            write_status(state)
            continue
        state["current_job"] = job_id
        state["updated_at_utc"] = now()
        write_status(state)
        command = [
            sys.executable,
            str(ROOT / condition["script"]),
            "--seed",
            str(seed),
            "--steps",
            str(condition["steps"]),
            "--warmup-steps",
            str(condition["warmup_steps"]),
            "--run-name",
            condition["name"],
            "--data-file",
            condition["data_file"],
        ]
        log_path = LOG_DIR / f"{condition['name']}_seed{seed}.log"
        try:
            run_and_tee(command, log_path)
        except Exception as error:
            state["status"] = "failed"
            state["failed_job"] = job_id
            state["error"] = repr(error)
            state["updated_at_utc"] = now()
            write_status(state)
            raise
        state["completed_jobs"].append(job_id)
        state["updated_at_utc"] = now()
        write_status(state)
    state["status"] = "complete"
    state["current_job"] = None
    state["completed_at_utc"] = now()
    state["updated_at_utc"] = now()
    write_status(state)
    print("Confirmatory training matrix complete.", flush=True)


if __name__ == "__main__":
    main()
