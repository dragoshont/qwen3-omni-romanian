"""Freeze a source-pinned Romanian FLEURS test before confirmatory training."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from datasets import Audio, load_dataset

from research_runtime import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "google/fleurs"
CONFIG = "ro_ro"
SPLIT = "test"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SELECTION_NAMESPACE = "ro-omni-fleurs-external-v1-20260924"
TARGET_COUNT = 200


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("ş", "ș").replace("ţ", "ț")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def selection_key(row: dict) -> str:
    payload = f"{SELECTION_NAMESPACE}:{row['audio']['path']}:{row['raw_transcription']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    training_texts: set[str] = set()
    for manifest_path in (
        ROOT / "manifests" / "ro1000_manifest.json",
        ROOT / "manifests" / "t4_5h_manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_texts.update(normalize(row["transcript"]) for row in manifest["samples"])

    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        split=SPLIT,
        streaming=True,
        revision=REVISION,
    ).cast_column("audio", Audio(decode=False))
    candidates = []
    for row in dataset:
        text = row["raw_transcription"].strip()
        if normalize(text) in training_texts:
            continue
        candidates.append(row)
    candidates.sort(key=selection_key)
    selected = candidates[:TARGET_COUNT]
    if len(selected) != TARGET_COUNT:
        raise ValueError(f"expected {TARGET_COUNT} eligible rows, found {len(selected)}")

    audio_dir = ROOT / "data" / "external_fleurs_ro_v1"
    audio_dir.mkdir(parents=True, exist_ok=True)
    output = ROOT / "eval" / "ro_external_fleurs_v1.jsonl"
    lines = []
    for row in selected:
        source_stem = Path(row["audio"]["path"]).stem
        audio_path = audio_dir / f"{source_stem}.wav"
        audio_bytes = row["audio"]["bytes"]
        if audio_bytes is None:
            raise ValueError(f"FLEURS audio {row['audio']['path']} has no bytes")
        if not audio_path.exists() or audio_path.read_bytes() != audio_bytes:
            audio_path.write_bytes(audio_bytes)
        record = {
            "id": f"fleurs_ro_{source_stem}",
            "category": "external_fleurs",
            "text": row["raw_transcription"].strip(),
            "source": {
                "dataset_id": DATASET_ID,
                "config": CONFIG,
                "split": SPLIT,
                "revision": REVISION,
                "row_id": row["id"],
                "original_path": row["audio"]["path"],
                "gender_class": int(row["gender"]),
                "natural_audio_num_samples": int(row["num_samples"]),
                "natural_audio_sampling_rate_hz": 16000,
                "natural_audio_sha256": sha256_file(audio_path),
                "local_audio_path": audio_path.relative_to(ROOT).as_posix(),
            },
            "selection_sha256": selection_key(row),
        }
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)} ({len(lines)} frozen prompts)")
    print(f"SHA-256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
