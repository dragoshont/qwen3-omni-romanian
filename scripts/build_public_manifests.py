"""Rebuild public, code-free manifests from the private local target files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research_runtime import git_commit, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = "81231a262c34d30bacbcda4a0ba7ddc13ab88bc0"
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_sample(row: dict) -> dict:
    audio_path = ROOT / Path(row["wav_path"])
    return {
        "sample_id": row["sample_id"],
        "dataset_row_id": row.get("dataset_row_id"),
        "transcript": row["transcript"],
        "transcript_sha256": hashlib.sha256(row["transcript"].encode("utf-8")).hexdigest(),
        "audio_duration_s": row.get("audio_duration_s"),
        "num_frames": row["num_frames"],
        "phonetic_score": row.get("phonetic_score"),
        "source_audio_sha256": sha256_file(audio_path) if audio_path.exists() else None,
        "codes_16_sha256": canonical_hash(row["codes_16"]),
    }


def manifest(name: str, rows: list[dict], private_path: Path, selection: dict) -> dict:
    durations = [float(row.get("audio_duration_s", 0.0)) for row in rows]
    return {
        "schema_version": 2,
        "name": name,
        "evidence_status": "local research data; audio and codec targets are not redistributed here",
        "source": {
            "dataset_id": "eduardem/romanian-tts-single-speaker",
            "url": "https://huggingface.co/datasets/eduardem/romanian-tts-single-speaker",
            "revision": DATASET_REVISION,
            "license_reported_by_pinned_card": "CC-BY-SA-4.0",
            "speaker": "Sanda (female, as reported by the dataset card)",
            "sampling_rate_hz": 24000,
        },
        "codec": {
            "model_id": "kyutai/mimi",
            "revision": MIMI_REVISION,
            "retained_codebooks": 16,
            "nominal_frame_rate_hz": 12.5,
        },
        "selection": selection,
        "sample_count": len(rows),
        "total_duration_s": round(sum(durations), 2),
        "total_duration_hours": round(sum(durations) / 3600, 6),
        "private_target_file_sha256": sha256_file(private_path),
        "generated_by_git_commit": git_commit(ROOT),
        "samples": [public_sample(row) for row in rows],
    }


def main() -> None:
    five_hour_path = ROOT / "reports" / "mimi_codes_ro_5h.jsonl"
    one_hour_path = ROOT / "reports" / "mimi_codes_ro1000.jsonl"
    five_hour_rows = jsonl(five_hour_path)
    one_hour_rows = jsonl(one_hour_path)
    if len(five_hour_rows) != 4311:
        raise ValueError(f"expected 4311 five-hour rows, found {len(five_hour_rows)}")
    if len(one_hour_rows) != 1000:
        raise ValueError(f"expected 1000 one-hour rows, found {len(one_hour_rows)}")

    by_target_hash = {canonical_hash(row["codes_16"]): row for row in five_hour_rows}
    linked_one_hour: list[dict] = []
    unmatched = 0
    for row in one_hour_rows:
        match = by_target_hash.get(canonical_hash(row["codes_16"]))
        linked = dict(row)
        if match is not None and match["transcript"] == row["transcript"]:
            linked["dataset_row_id"] = match["dataset_row_id"]
            linked["audio_duration_s"] = match["audio_duration_s"]
            linked["num_frames"] = match["num_frames"]
            linked["phonetic_score"] = match["phonetic_score"]
            linked["wav_path"] = match["wav_path"]
        else:
            # The legacy one-hour curriculum predates source-ID capture.  Do
            # not invent provenance for rows absent from the later 5 h slice.
            unmatched += 1
            linked["dataset_row_id"] = None
            linked["num_frames"] = len(row["codes_16"][0])
            linked["audio_duration_s"] = round(linked["num_frames"] / 12.5, 2)
            linked["phonetic_score"] = None
        linked_one_hour.append(linked)

    write_json(
        ROOT / "manifests" / "t4_5h_manifest.json",
        manifest(
            "t4_5h_training_targets",
            five_hour_rows,
            five_hour_path,
            {
                "algorithm": "stream order; length 20-220 characters; duration 1.8-11.5 seconds; phonetic score >=3; exact normalized Full-200 exclusions",
                "random_sampling": False,
                "selection_seed": None,
                "target_duration_s": 18000,
                "known_limitation": "stream-order threshold selection is not coverage-aware",
            },
        ),
    )
    write_json(
        ROOT / "manifests" / "ro1000_manifest.json",
        manifest(
            "ro1000_training_targets",
            linked_one_hour,
            one_hour_path,
            {
                "algorithm": "legacy first-stage selection; source rows recovered where possible by exact transcript and codec-target hash",
                "random_sampling": False,
                "selection_seed": None,
                "source_row_ids_recovered": len(one_hour_rows) - unmatched,
                "source_row_ids_unavailable": unmatched,
                "known_limitation": "legacy selection was not preregistered; some original source row IDs were not recorded",
            },
        ),
    )
    print("Rebuilt manifests/t4_5h_manifest.json and manifests/ro1000_manifest.json")


if __name__ == "__main__":
    main()
