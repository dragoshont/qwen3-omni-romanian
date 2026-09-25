from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import normalize_romanian_asr, prompt_seed  # noqa: E402
from deterministic_sampler import EpochShuffleSampler  # noqa: E402


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class EvaluationProtocolTests(unittest.TestCase):
    def test_prompt_seed_matches_frozen_formula(self) -> None:
        prompt_id = "ro_0001"
        expected = int.from_bytes(
            hashlib.sha256(f"t4-seeded-v2:42:{prompt_id}".encode()).digest()[:8],
            "big",
        ) % (2**63 - 1)
        self.assertEqual(prompt_seed(42, prompt_id), expected)

    def test_romanian_normalization(self) -> None:
        self.assertEqual(normalize_romanian_asr("ŞTIINŢĂ,  în România!"), "știință în românia")

    def test_eval_sets_are_unique_and_external_is_disjoint(self) -> None:
        full = jsonl(ROOT / "eval" / "ro_holdout_200.jsonl")
        quick = jsonl(ROOT / "eval" / "ro_holdout_quick_40.jsonl")
        external = jsonl(ROOT / "eval" / "ro_external_fleurs_v1.jsonl")
        self.assertEqual(len(full), 200)
        self.assertEqual(len(quick), 40)
        self.assertEqual(len(external), 200)
        for rows in (full, quick, external):
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
        full_ids = {row["id"] for row in full}
        self.assertTrue({row["id"] for row in quick}.issubset(full_ids))
        training = set()
        for name in ("ro1000_manifest.json", "t4_5h_manifest.json"):
            payload = json.loads((ROOT / "manifests" / name).read_text(encoding="utf-8"))
            training.update(normalize_romanian_asr(row["transcript"]) for row in payload["samples"])
        self.assertFalse(training.intersection(normalize_romanian_asr(row["text"]) for row in external))


class SamplerTests(unittest.TestCase):
    def test_sampler_resume_is_exact(self) -> None:
        uninterrupted = EpochShuffleSampler(11, 314)
        expected = [uninterrupted.next() for _ in range(47)]
        first = EpochShuffleSampler(11, 314)
        observed = [first.next() for _ in range(19)]
        state = first.state_dict()
        resumed = EpochShuffleSampler(11, 314)
        resumed.load_state_dict(state)
        observed.extend(resumed.next() for _ in range(28))
        self.assertEqual(observed, expected)

    def test_sampler_visits_every_row_once_per_epoch(self) -> None:
        sampler = EpochShuffleSampler(23, 2718)
        first_epoch = [sampler.next() for _ in range(23)]
        second_epoch = [sampler.next() for _ in range(23)]
        self.assertEqual(sorted(first_epoch), list(range(23)))
        self.assertEqual(sorted(second_epoch), list(range(23)))
        self.assertNotEqual(first_epoch, second_epoch)


class ManifestTests(unittest.TestCase):
    def test_training_manifests_have_pinned_provenance(self) -> None:
        expected = {"ro1000_manifest.json": 1000, "t4_5h_manifest.json": 4311}
        for name, count in expected.items():
            payload = json.loads((ROOT / "manifests" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["sample_count"], count)
            self.assertEqual(len(payload["samples"]), count)
            self.assertEqual(payload["source"]["revision"], "81231a262c34d30bacbcda4a0ba7ddc13ab88bc0")
            self.assertEqual(payload["codec"]["revision"], "89091b3e466eb6a9d11e537bf26b144f194978f7")
            self.assertIsNone(payload["selection"]["selection_seed"])
            self.assertTrue(all(row["codes_16_sha256"] for row in payload["samples"]))

    def test_confirmatory_matrix_is_complete(self) -> None:
        matrix = json.loads((ROOT / "configs" / "confirmatory_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(matrix["training_seeds"], [42, 314, 2718])
        names = {condition["name"] for condition in matrix["conditions"]}
        self.assertEqual(
            names,
            {
                "component_talker_only_1h_1000",
                "component_joint_1h_1000",
                "scaling_joint_1h_matched_epochs_580",
                "scaling_joint_1h_matched_updates_2500",
                "scaling_joint_5h_primary_2500",
            },
        )


if __name__ == "__main__":
    unittest.main()
