"""Aggregate the controlled-seed T4 Full-200 correction for issue #1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HOLDOUT = ROOT / "eval" / "ro_holdout_200.jsonl"
SEEDS = (42, 314, 2718)
STEPS = (1000, 2500)
BOOTSTRAP_SEED = 1
BOOTSTRAP_SAMPLES = 20_000


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_metrics(report: dict) -> dict:
    rows = report["detailed_results"]
    cers = np.asarray([row["cer"] for row in rows], dtype=float)
    wers = np.asarray([row["wer"] for row in rows], dtype=float)
    return {
        "mean_cer": float(np.mean(cers)),
        "median_cer": float(np.median(cers)),
        "p90_cer": float(np.percentile(cers, 90)),
        "mean_wer": float(np.mean(wers)),
        "median_wer": float(np.median(wers)),
        "p90_wer": float(np.percentile(wers, 90)),
        "repetition_rate": float(report["repetition_rate"]),
        "eos_rate": float(report["eos_rate"]),
        "catastrophic_cer_count": int(np.sum(cers > 1.0)),
    }


def summarize_runs(runs: list[dict]) -> dict:
    keys = list(runs[0])
    summary = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def paired_bootstrap(step_rows: dict[int, dict[tuple[int, str], dict]], metric: str) -> dict:
    keys = sorted(step_rows[1000])
    before = np.asarray([step_rows[1000][key][metric] for key in keys], dtype=float)
    after = np.asarray([step_rows[2500][key][metric] for key in keys], dtype=float)
    delta = after - before
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        draw = rng.integers(0, len(keys), len(keys))
        sampled[index] = float(np.mean(delta[draw]))
    return {
        "step2500_minus_step1000_mean": float(np.mean(delta)),
        "bootstrap_95pct_ci": [float(x) for x in np.percentile(sampled, [2.5, 97.5])],
        "step2500_wins": int(np.sum(after < before)),
        "ties": int(np.sum(after == before)),
        "step2500_losses": int(np.sum(after > before)),
    }


def main() -> None:
    holdout_rows = [json.loads(line) for line in HOLDOUT.read_text(encoding="utf-8").splitlines() if line]
    holdout = {row["id"]: row for row in holdout_rows}
    raw_reports = {}
    report_hashes = {}
    per_run = {step: [] for step in STEPS}
    paired_rows = {step: {} for step in STEPS}

    for step in STEPS:
        for seed in SEEDS:
            path = REPORTS / f"t4_seeded_step{step}_seed{seed}_full200.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing required report: {path}")
            report = load(path)
            config = report.get("evaluation_config", {})
            if config.get("seed") != seed or report.get("total_samples") != 200:
                raise ValueError(f"Incomplete or wrong-seed report: {path}")
            rows = report["detailed_results"]
            if len(rows) != 200 or len({row['id'] for row in rows}) != 200:
                raise ValueError(f"Invalid row IDs: {path}")
            for row in rows:
                expected = holdout[row["id"]]
                if row["category"] != expected["category"] or row["reference_text"] != expected["text"]:
                    raise ValueError(f"Holdout mismatch in {path}: {row['id']}")
                paired_rows[step][(seed, row["id"])] = row
            raw_reports[(step, seed)] = report
            report_hashes[f"step{step}_seed{seed}"] = sha256(path)
            per_run[step].append(run_metrics(report))

    across_seed = {f"step{step}": summarize_runs(per_run[step]) for step in STEPS}
    scores = {}
    for step in STEPS:
        metrics = across_seed[f"step{step}"]
        repetition = metrics["repetition_rate"]["mean"]
        scores[f"step{step}"] = (
            0.5 * metrics["mean_cer"]["mean"]
            + 0.5 * metrics["median_cer"]["mean"]
            + (0.1 if repetition > 1.0 else 0.0)
        )
    winner = min(scores, key=scores.get)

    output = {
        "issue": 1,
        "evaluation": {
            "steps": list(STEPS),
            "seeds": list(SEEDS),
            "samples_per_run": 200,
            "total_generations": len(STEPS) * len(SEEDS) * 200,
            "holdout_sha256": sha256(HOLDOUT),
            "report_sha256": report_hashes,
        },
        "per_run": {
            f"step{step}": {str(seed): result for seed, result in zip(SEEDS, per_run[step])}
            for step in STEPS
        },
        "across_seed": across_seed,
        "paired_step2500_vs_step1000": {
            metric: paired_bootstrap(paired_rows, metric) for metric in ("cer", "wer")
        },
        "selection": {
            "formula": "0.5 * across-seed mean of run mean CER + 0.5 * across-seed mean of run median CER + repetition penalty",
            "scores": scores,
            "asr_selected_checkpoint": winner,
            "native_listener_review_pending": True,
        },
    }
    json_path = REPORTS / "t4_seeded_full200_comparison.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# T4 controlled-seed Full-200 comparison",
        "",
        "Corrective evaluation for GitHub issue #1.",
        "",
        "| Checkpoint | Mean CER (mean ± SD) | Median CER (mean ± SD) | Mean WER (mean ± SD) | Median WER (mean ± SD) | Repetition |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in STEPS:
        values = across_seed[f"step{step}"]
        md.append(
            f"| Step {step:,} | "
            f"{values['mean_cer']['mean']*100:.2f}% ± {values['mean_cer']['std']*100:.2f} | "
            f"{values['median_cer']['mean']*100:.2f}% ± {values['median_cer']['std']*100:.2f} | "
            f"{values['mean_wer']['mean']*100:.2f}% ± {values['mean_wer']['std']*100:.2f} | "
            f"{values['median_wer']['mean']*100:.2f}% ± {values['median_wer']['std']*100:.2f} | "
            f"{values['repetition_rate']['mean']:.2f}% |"
        )
    md.extend(
        [
            "",
            f"**ASR-metric selection:** `{winner}`.",
            "",
            "This resolves the uncontrolled-randomness defect for checkpoint comparison. "
            "Blinded native-listener review remains a separate release gate.",
            "",
            "See `t4_seeded_full200_comparison.json` for per-run metrics, hashes, "
            "paired bootstrap intervals and the exact selection score.",
        ]
    )
    md_path = REPORTS / "t4_seeded_full200_comparison.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"ASR-selected checkpoint: {winner}")


if __name__ == "__main__":
    main()

