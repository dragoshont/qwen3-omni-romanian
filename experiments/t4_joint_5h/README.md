# T4 — Joint Talker + MTP, ~5 Hours

> **In progress. Snapshot: 2026-09-24 18:28 EEST.**

## Purpose

Measure the effect of a larger/broader Romanian data condition while preserving the T3 architecture.

T4 starts from:
- the same stock Qwen base
- fresh LoRA adapters
- same LoRA ranks/targets
- same quantization strategy
- same evaluation decoding

It is **not initialized from T3**.

## Dataset target

- ~18,000 seconds / ~5 h
- ~3,350 utterances
- held-out Full-200 exact-normalized exclusion
- coverage-aware deterministic selection

## Training target

- 2,500 optimizer steps
- gradient accumulation 4
- ~10,000 sample presentations
- ~3 effective epochs
- checkpoints: 500 / 1000 / 1500 / 2000 / 2500

## Live Quick-40

| Step | Mean CER | Median CER | Mean WER | Median WER | EOS | Loops |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 65.43% | 30.60% | 165.71% | 80.00% | 100% | 2.5% |
| 1000 | 26.03% | 24.68% | 70.04% | 69.62% | 100% | 0% |
| 1500 | 26.50% | 25.44% | 71.36% | 72.73% | 100% | 0% |
| 2000 | 59.87% | **22.74%** | 106.95% | **66.67%** | 100% | 2.5% |
| 2500 | pending | pending | pending | pending | pending | pending |

At step 2,000 a single catastrophic long-sentence loop inflated mean error while the median improved.

## Champion selection

Do not automatically choose the lowest mean or final step.

Preserve:
1. best stable-mean candidate;
2. best robust/median candidate.

If they differ, run Full-200 on both.
