"""Independently verify the published T4 Full-200 reports.

This script reads only committed benchmark data and regenerates a compact,
machine-readable audit. It does not run model inference or ASR.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
EVAL_FILE = ROOT / "eval" / "ro_holdout_200.jsonl"
OUTPUT_FILE = REPORT_DIR / "t4_verification_summary.json"
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_SAMPLES = 20_000

REPORTS = {
    "t3": REPORT_DIR / "champion_200_benchmark.json",
    "t4_step1000": REPORT_DIR / "t4_step1000_full200_benchmark.json",
    "t4_step2000": REPORT_DIR / "t4_step2000_full200_benchmark.json",
    "t4_step2500": REPORT_DIR / "t4_step2500_full200_benchmark.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_holdout() -> dict[str, dict]:
    rows = []
    with EVAL_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return {row["id"]: row for row in rows}


def trimmed_mean(values: np.ndarray, proportion: float = 0.05) -> float:
    ordered = np.sort(values)
    trim = int(len(ordered) * proportion)
    return float(np.mean(ordered[trim:-trim])) if trim else float(np.mean(ordered))


def metric_summary(rows: list[dict], report: dict, t3_audit: dict) -> dict:
    cers = np.asarray([row["cer"] for row in rows], dtype=float)
    wers = np.asarray([row["wer"] for row in rows], dtype=float)
    repetition_rate = report.get("repetition_rate")
    eos_rate = report.get("eos_rate")
    max_token_hit_rate = report.get("max_token_hit_rate")
    if repetition_rate is None:
        repetition_rate = t3_audit.get("repetition_rate")
    if eos_rate is None:
        eos_rate = t3_audit.get("eos_rate")
    if max_token_hit_rate is None:
        max_token_hit_rate = t3_audit.get("max_token_hit_rate")

    return {
        "count": len(rows),
        "mean_cer": float(np.mean(cers)),
        "median_cer": float(np.median(cers)),
        "p90_cer": float(np.percentile(cers, 90)),
        "trimmed_mean_cer_5pct": trimmed_mean(cers),
        "mean_wer": float(np.mean(wers)),
        "median_wer": float(np.median(wers)),
        "p90_wer": float(np.percentile(wers, 90)),
        "trimmed_mean_wer_5pct": trimmed_mean(wers),
        "cer_over_100pct_count": int(np.sum(cers > 1.0)),
        "wer_over_200pct_count": int(np.sum(wers > 2.0)),
        "repetition_rate": repetition_rate,
        "eos_rate": eos_rate,
        "max_token_hit_rate": max_token_hit_rate,
    }


def paired_bootstrap(
    baseline: dict[str, dict], candidate: dict[str, dict], metric: str
) -> dict:
    ids = list(baseline)
    before = np.asarray([baseline[item_id][metric] for item_id in ids], dtype=float)
    after = np.asarray([candidate[item_id][metric] for item_id in ids], dtype=float)
    delta = after - before
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mean_deltas = np.empty(BOOTSTRAP_SAMPLES)
    median_deltas = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(ids), len(ids))
        mean_deltas[index] = float(np.mean(delta[sample]))
        median_deltas[index] = float(np.median(after[sample]) - np.median(before[sample]))
    return {
        "candidate_minus_baseline_mean": float(np.mean(delta)),
        "mean_delta_bootstrap_95pct_ci": [
            float(x) for x in np.percentile(mean_deltas, [2.5, 97.5])
        ],
        "candidate_minus_baseline_median": float(np.median(after) - np.median(before)),
        "median_delta_bootstrap_95pct_ci": [
            float(x) for x in np.percentile(median_deltas, [2.5, 97.5])
        ],
        "candidate_wins": int(np.sum(after < before)),
        "ties": int(np.sum(after == before)),
        "candidate_losses": int(np.sum(after > before)),
    }


def main() -> None:
    holdout = load_holdout()
    t3_audit = load_json(REPORT_DIR / "champion_200_comprehensive_audit.json")["overall"]
    report_data = {name: load_json(path) for name, path in REPORTS.items()}
    indexed = {
        name: {row["id"]: row for row in report["detailed_results"]}
        for name, report in report_data.items()
    }

    integrity = {}
    metrics = {}
    for name, rows_by_id in indexed.items():
        mismatches = []
        for item_id, row in rows_by_id.items():
            expected = holdout.get(item_id)
            if (
                expected is None
                or row["category"] != expected["category"]
                or row["reference_text"] != expected["text"]
            ):
                mismatches.append(item_id)
        category_counts = {}
        for row in rows_by_id.values():
            category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
        integrity[name] = {
            "row_count": len(rows_by_id),
            "unique_id_count": len(set(rows_by_id)),
            "holdout_mismatch_count": len(mismatches),
            "holdout_mismatch_ids": mismatches,
            "category_counts": category_counts,
        }
        metrics[name] = metric_summary(
            list(rows_by_id.values()), report_data[name], t3_audit if name == "t3" else {}
        )

    scores = {}
    for name in ("t4_step1000", "t4_step2000", "t4_step2500"):
        values = metrics[name]
        penalty = 0.1 if values["repetition_rate"] > 1.0 else 0.0
        scores[name] = 0.5 * values["mean_cer"] + 0.5 * values["median_cer"] + penalty

    output = {
        "verification": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "holdout_sha256": sha256(EVAL_FILE),
            "report_sha256": {name: sha256(path) for name, path in REPORTS.items()},
        },
        "integrity": integrity,
        "recomputed_metrics": metrics,
        "paired_comparisons": {
            "t4_step2500_vs_t3": {
                metric: paired_bootstrap(indexed["t3"], indexed["t4_step2500"], metric)
                for metric in ("cer", "wer")
            },
            "t4_step2500_vs_t4_step1000": {
                metric: paired_bootstrap(
                    indexed["t4_step1000"], indexed["t4_step2500"], metric
                )
                for metric in ("cer", "wer")
            },
        },
        "selection": {
            "formula": "0.5 * mean_cer + 0.5 * median_cer + (0.1 if repetition_rate > 1% else 0)",
            "scores": scores,
            "lowest_score": min(scores, key=scores.get),
            "status": "provisional",
            "caveats": [
                "Each checkpoint has one stochastic decoding run.",
                "The evaluator used do_sample=True but did not apply the configured seed.",
                "Final selection requires seeded repeated evaluation and native-listener review.",
            ],
        },
    }
    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
