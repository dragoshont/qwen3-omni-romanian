# T4 — Joint Talker + MTP, ~5 Hours

> **Completed 24 September 2026. Full-200 champion: step 2,500.**

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

## Quick-40 checkpoint screen

| Step | Mean CER | Median CER | Mean WER | Median WER | EOS | Loops |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 65.43% | 30.60% | 165.71% | 80.00% | 100% | 2.5% |
| 1000 | 26.03% | 24.68% | 70.04% | 69.62% | 100% | 0% |
| 1500 | 26.50% | 25.44% | 71.36% | 72.73% | 100% | 0% |
| 2000 | 59.87% | **22.74%** | 106.95% | **66.67%** | 100% | 2.5% |
| 2500 | **24.24%** | 23.17% | **68.42%** | **66.67%** | 100% | 0% |

At step 2,000 a single catastrophic long-sentence loop inflated mean error while the median improved.

## Full-200 champion selection

Steps 1,000, 2,000 and 2,500 were evaluated on the same Full-200 set. Step 2,500 was selected by the declared multi-metric rule:

- mean CER: 30.73%
- median CER: 22.85%
- mean WER: 75.81%
- median WER: 63.64%
- EOS success: 100%
- repetition: 1.0%

See [`../../reports/t4_final_champion_declaration.md`](../../reports/t4_final_champion_declaration.md).
