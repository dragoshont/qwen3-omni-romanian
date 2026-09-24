# Evidence and Claim Discipline

This repository separates **observed facts**, **implementation-derived evidence**, and **hypotheses**.

## E1 — Codec bridge

Thirty Romanian clips were encoded with Mimi and the first 16 streams were fed to Qwen3-Omni Code2Wav.

Independent ASR:
- originals: ~5.8% WER
- reconstructed: ~6.4% WER

Interpretation: sufficiently compatible to justify using these codes as targets.

This does **not** establish that Qwen officially specifies Mimi as its canonical target encoder.

## E2 — Memory feasibility

Actual training:
- T2 ~12.13 GB
- T3 ~12.16 GB
- T4 live ~12.57 GB

An earlier ~5.88 GB dry run did not exercise the full multi-level MTP training graph and is not used as evidence for full training memory.

## E3 — Leakage control

Earlier audit found exact normalized transcript overlaps:
- Quick-40 vs training: 0/40
- Full-200 vs training: 0/200

Future manifests should record this check every time.

## E4 — T3 Full-200

- mean CER 38.53%
- median CER 29.30%
- P90 CER 45.13%
- mean WER 87.85%
- median WER 75.00%
- P90 WER 110.82%
- EOS 100%
- max-token 0%
- repetition 1.5%

This is evidence of adaptation, not production readiness.

## E5 — T4 intermediate

- step 1000 mean CER 26.03%, median 24.68%, loops 0%
- step 1500 mean CER 26.50%, median 25.44%, loops 0%
- step 2000 median CER 22.74% but one catastrophic loop inflates mean CER to 59.87%

Final claims wait for Full-200.

## Open falsification targets

Useful external challenges include:
1. evidence that Mimi-token semantics are mismatched despite decoder reconstruction;
2. label-alignment errors;
3. hidden train/eval leakage;
4. decoding artifacts mistaken for training improvement;
5. better equal-quality low-VRAM recipes.
