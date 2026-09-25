"""Run the corrective T4 Full-200 matrix for GitHub issue #1.

The runner is resumable: a report is skipped only when it contains 200 samples
and records the expected seed. Generated audio remains ignored by Git.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 314, 2718)
CHECKPOINTS = {
    1000: "models/T4_data_scale_5h/final/checkpoint_step_1000",
    2500: "models/T4_data_scale_5h/final/checkpoint_step_2500",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_is_complete(path: Path, expected_seed: int, expected_checkpoint: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        return (
            report.get("total_samples") == 200
            and len(report.get("detailed_results", [])) == 200
            and report.get("evaluation_config", {}).get("seed") == expected_seed
            and report.get("evaluation_config", {}).get("per_prompt_seed_scheme")
            and report.get("champion_model") == expected_checkpoint
            and report.get("holdout_sha256") == sha256(ROOT / "eval" / "ro_holdout_200.jsonl")
            and all("generation_seed" in row for row in report.get("detailed_results", []))
        )
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    started = time.time()
    for seed in SEEDS:
        for step, checkpoint in CHECKPOINTS.items():
            report = ROOT / "reports" / f"t4_seeded_step{step}_seed{seed}_full200.json"
            output = ROOT / "outputs" / f"t4_seeded_step{step}_seed{seed}_full200"
            if report_is_complete(report, seed, checkpoint):
                print(f"SKIP complete: step={step} seed={seed}", flush=True)
                continue

            print("=" * 72, flush=True)
            print(f"RUN step={step} seed={seed}", flush=True)
            print("=" * 72, flush=True)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "eval_full_200_champion.py"),
                checkpoint,
                str(report.relative_to(ROOT)),
                str(output.relative_to(ROOT)),
                str(seed),
            ]
            subprocess.run(command, cwd=ROOT, check=True)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_t4_seeded_full200.py")],
        cwd=ROOT,
        check=True,
    )
    print(f"Corrective matrix completed in {(time.time() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
