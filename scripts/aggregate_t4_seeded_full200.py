"""Aggregate the preregistered T4 Full-200 correction for GitHub issue #1."""

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
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_SAMPLES = 20_000


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_prompt_seed(base_seed: int, prompt_id: str) -> int:
    payload = f"t4-seeded-v2:{base_seed}:{prompt_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def run_metrics(report: dict) -> dict:
    rows = report["detailed_results"]
    arrays = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in ("cer", "wer", "strict_cer", "strict_wer")
    }
    result = {}
    for key, values in arrays.items():
        result[f"mean_{key}"] = float(np.mean(values))
        result[f"median_{key}"] = float(np.median(values))
        result[f"p90_{key}"] = float(np.percentile(values, 90))
    result.update(
        repetition_rate=float(report["repetition_rate"]),
        eos_rate=float(report["eos_rate"]),
        catastrophic_cer_count=int(np.sum(arrays["cer"] > 1.0)),
    )
    return result


def summarize_runs(runs: list[dict]) -> dict:
    summary = {}
    for key in runs[0]:
        values = np.asarray([run[key] for run in runs], dtype=float)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def prompt_clustered_comparison(values: dict[int, np.ndarray], metric: str) -> dict:
    """Resample prompts; retain all fixed-seed observations inside each cluster."""
    before = values[1000]  # [seed, prompt]
    after = values[2500]
    delta = after - before
    prompt_delta = np.mean(delta, axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mean_samples = np.empty(BOOTSTRAP_SAMPLES)
    median_samples = np.empty(BOOTSTRAP_SAMPLES)
    prompt_count = before.shape[1]
    for index in range(BOOTSTRAP_SAMPLES):
        draw = rng.integers(0, prompt_count, prompt_count)
        mean_samples[index] = float(np.mean(prompt_delta[draw]))
        median_samples[index] = float(
            np.median(after[:, draw]) - np.median(before[:, draw])
        )
    per_seed = {
        str(seed): float(np.mean(delta[seed_index]))
        for seed_index, seed in enumerate(SEEDS)
    }
    return {
        "metric": metric,
        "unit_of_resampling": "prompt (all three fixed seeds retained within prompt)",
        "step2500_minus_step1000_mean": float(np.mean(delta)),
        "mean_difference_prompt_clustered_95pct_ci": [
            float(x) for x in np.percentile(mean_samples, [2.5, 97.5])
        ],
        "step2500_minus_step1000_median": float(
            np.median(after) - np.median(before)
        ),
        "median_difference_prompt_clustered_95pct_ci": [
            float(x) for x in np.percentile(median_samples, [2.5, 97.5])
        ],
        "per_seed_mean_differences": per_seed,
        "prompt_level_step2500_wins": int(np.sum(prompt_delta < 0)),
        "prompt_level_ties": int(np.sum(prompt_delta == 0)),
        "prompt_level_step2500_losses": int(np.sum(prompt_delta > 0)),
    }


def main() -> None:
    holdout_rows = [
        json.loads(line)
        for line in HOLDOUT.read_text(encoding="utf-8").splitlines()
        if line
    ]
    holdout = {row["id"]: row for row in holdout_rows}
    prompt_ids = [row["id"] for row in holdout_rows]
    report_hashes = {}
    checkpoint_hashes = {step: set() for step in STEPS}
    per_run = {step: [] for step in STEPS}
    matrices = {
        metric: {step: np.empty((len(SEEDS), len(prompt_ids))) for step in STEPS}
        for metric in ("cer", "wer", "strict_cer", "strict_wer")
    }

    for step in STEPS:
        for seed_index, seed in enumerate(SEEDS):
            path = REPORTS / f"t4_seeded_step{step}_seed{seed}_full200.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing required report: {path}")
            report = load(path)
            config = report.get("evaluation_config", {})
            if config.get("seed") != seed or report.get("total_samples") != 200:
                raise ValueError(f"Incomplete or wrong-seed report: {path}")
            rows = report["detailed_results"]
            by_id = {row["id"]: row for row in rows}
            if len(rows) != 200 or set(by_id) != set(holdout):
                raise ValueError(f"Invalid row IDs: {path}")
            for prompt_index, prompt_id in enumerate(prompt_ids):
                row = by_id[prompt_id]
                expected = holdout[prompt_id]
                if row["category"] != expected["category"] or row["reference_text"] != expected["text"]:
                    raise ValueError(f"Holdout mismatch in {path}: {prompt_id}")
                if row.get("generation_seed") != expected_prompt_seed(seed, prompt_id):
                    raise ValueError(f"Prompt-seed mismatch in {path}: {prompt_id}")
                for metric in matrices:
                    matrices[metric][step][seed_index, prompt_index] = row[metric]
            if report.get("holdout_sha256") != sha256(HOLDOUT):
                raise ValueError(f"Holdout hash mismatch: {path}")
            checkpoint_hashes[step].add(report.get("checkpoint_sha256"))
            report_hashes[f"step{step}_seed{seed}"] = sha256(path)
            per_run[step].append(run_metrics(report))

    if any(len(hashes) != 1 or None in hashes for hashes in checkpoint_hashes.values()):
        raise ValueError(f"Checkpoint hashes changed across seeds: {checkpoint_hashes}")

    across_seed = {f"step{step}": summarize_runs(per_run[step]) for step in STEPS}
    comparisons = {
        metric: prompt_clustered_comparison(matrices[metric], metric)
        for metric in matrices
    }
    primary = comparisons["cer"]
    primary_ci = primary["mean_difference_prompt_clustered_95pct_ci"]
    means = {
        f"step{step}": across_seed[f"step{step}"]["mean_cer"]["mean"]
        for step in STEPS
    }
    operational_candidate = min(means, key=means.get)
    conclusive = primary_ci[1] < 0 or primary_ci[0] > 0
    selection = {
        "preregistered_rule": "Lower normalized mean CER is an automated-metric winner only when the prompt-clustered 95% CI for the difference excludes zero.",
        "primary_metric": "normalized CER",
        "mean_normalized_cer": means,
        "prompt_clustered_95pct_ci_step2500_minus_step1000": primary_ci,
        "status": "automated_metric_winner" if conclusive else "statistically_inconclusive",
        "automated_metric_winner": operational_candidate if conclusive else None,
        "operational_candidate": operational_candidate,
        "native_listener_review_pending": True,
        "untouched_external_test_pending": True,
    }

    output = {
        "issue": 1,
        "protocol": "eval/T4_SEEDED_PROTOCOL.md",
        "evaluation": {
            "steps": list(STEPS),
            "seeds": list(SEEDS),
            "samples_per_run": 200,
            "total_generations": len(STEPS) * len(SEEDS) * 200,
            "holdout_sha256": sha256(HOLDOUT),
            "checkpoint_sha256": {
                f"step{step}": next(iter(checkpoint_hashes[step])) for step in STEPS
            },
            "report_sha256": report_hashes,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "per_run": {
            f"step{step}": {
                str(seed): result for seed, result in zip(SEEDS, per_run[step])
            }
            for step in STEPS
        },
        "across_seed": across_seed,
        "prompt_clustered_comparisons": comparisons,
        "selection": selection,
    }
    json_path = REPORTS / "t4_seeded_full200_comparison.json"
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = [
        "# T4 hardened seeded Full-200 comparison",
        "",
        "Corrective evaluation for GitHub issue #1 under the protocol in "
        "`eval/T4_SEEDED_PROTOCOL.md`.",
        "",
        "Primary CER is normalized as specified in the protocol. Values are the "
        "mean and sample SD across three fixed generation seeds.",
        "",
        "| Checkpoint | Mean CER | Median CER | Mean WER | Strict mean CER | EOS | ASR repetition proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in STEPS:
        values = across_seed[f"step{step}"]
        md.append(
            f"| Step {step:,} | "
            f"{values['mean_cer']['mean']*100:.2f}% ± {values['mean_cer']['std']*100:.2f} | "
            f"{values['median_cer']['mean']*100:.2f}% ± {values['median_cer']['std']*100:.2f} | "
            f"{values['mean_wer']['mean']*100:.2f}% ± {values['mean_wer']['std']*100:.2f} | "
            f"{values['mean_strict_cer']['mean']*100:.2f}% ± {values['mean_strict_cer']['std']*100:.2f} | "
            f"{values['eos_rate']['mean']:.2f}% | "
            f"{values['repetition_rate']['mean']:.2f}% |"
        )
    ci_pct = [value * 100 for value in primary_ci]
    md.extend(
        [
            "",
            f"Step 2,500 minus step 1,000 mean normalized CER: "
            f"{primary['step2500_minus_step1000_mean']*100:.2f} percentage points "
            f"(prompt-clustered 95% interval {ci_pct[0]:.2f} to {ci_pct[1]:.2f}).",
            "",
            f"**Decision status:** `{selection['status']}`. "
            f"**Operational candidate:** `{operational_candidate}`.",
            "",
            "The three seeds quantify inference-time sampling variation only. They do "
            "not quantify training-seed variation. Blinded native-listener review and "
            "an untouched external test remain release gates.",
        ]
    )
    md_path = REPORTS / "t4_seeded_full200_comparison.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"Decision status: {selection['status']}")
    print(f"Operational candidate: {operational_candidate}")


if __name__ == "__main__":
    main()
