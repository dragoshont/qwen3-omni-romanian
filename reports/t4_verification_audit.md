# T4 verification audit

Date: 24 September 2026

## Bottom line

The completed T4 reports use the intended Full-200 benchmark, and the published
aggregate CER/WER values can be recomputed from their 200 per-sample records.
There is no exact normalized overlap between the 4,311 T4 training transcripts
and the 200 held-out prompts.

Step 2,500 remains the **provisional** champion under the existing composite
score. It is not yet a final release choice because each checkpoint has only one
stochastic decoding run and the configured seed was not applied by the evaluator.

## What looked like a data mismatch

The derived file `reports/t4_step2500_full200_benchmark.json` contained mojibaked
Romanian text in 198 of its 200 `reference_text` fields. This was an encoding
error introduced when the canonical step-2,500 report was copied/enriched; it was
not a different set of prompts:

- the IDs and categories matched the canonical report;
- all 200 metric values matched the canonical report;
- `reports/t4_best_200_benchmark.json` retained correct UTF-8 Romanian text;
- the repaired step-2,500 report now matches `eval/ro_holdout_200.jsonl` exactly.

The corrupted file is preserved in Git history at commit `37b2167` with SHA-256
`d17e52a3cf58bf177df98dd1bc6b71d513affa79d7694a55a89ba089e4759d66`.
The repaired file has SHA-256
`f84249bb3d43cbb2bd80979fff1b7b960298893855ceb58eff4aa36fb4635d6e`.

## Benchmark alignment checks

For T3 and T4 steps 1,000, 2,000 and 2,500:

- 200 rows are present;
- all 200 IDs are unique;
- each of the ten categories contains exactly 20 prompts;
- every ID, category and reference text matches `eval/ro_holdout_200.jsonl`;
- mean, median and P90 CER/WER recompute from the detailed rows.

The machine-readable results are in `t4_verification_summary.json` and can be
regenerated with:

```powershell
python scripts/verify_t4_results.py
```

## Training/evaluation separation

An independent normalized-text comparison found:

- T4 training manifest: 4,311 unique transcripts;
- held-out benchmark: 200 unique prompts;
- exact normalized overlaps: **0**;
- highest fuzzy text similarity: **72.83%**, between two semantically different
  sentences that share a Romanian date phrase.

The manifest previously reported 201 excluded strings because the preparation
script inserted an empty value for a missing alternate field in addition to the
200 real prompts. The filter still excluded all 200 prompts correctly; the count
and script have been corrected to ignore empty values.

## Recomputed Full-200 metrics

| Run | Mean CER | Median CER | P90 CER | 5% trimmed mean CER | Mean WER | Median WER | Repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T3 | 38.53% | 29.30% | 45.13% | 29.52% | 87.85% | 75.00% | 1.5% |
| T4 step 1,000 | 30.09% | 23.92% | 44.28% | 25.73% | 76.10% | 70.00% | 0.5% |
| T4 step 2,000 | 36.00% | 23.46% | 39.24% | 24.05% | 77.62% | 65.99% | 1.0% |
| T4 step 2,500 | 30.73% | 22.85% | 41.47% | 24.00% | 75.81% | 63.64% | 1.0% |

The generated declaration previously displayed T3 repetition as 0% because the
base T3 report lacks per-row failure flags and the compiler defaulted missing
values to zero. The verified T3 comprehensive audit reports 1.5%; the compiler
now imports that value.

## What the evidence supports

Compared with T3, step 2,500 improves the median CER by 6.45 percentage points
and wins on CER for 131 of 200 prompts (12 ties). Its 5% trimmed mean CER improves
from 29.52% to 24.00%. These are strong signs that five-hour training improved
typical behavior and reduced the outlier-sensitive error distribution.

The raw 20.3% relative mean-CER reduction is descriptive, not yet a stable effect
estimate: a paired bootstrap 95% interval for the mean difference crosses zero
because both runs contain extreme failures. The median-CER interval does not
cross zero in this audit.

Step 2,500 only narrowly beats step 1,000 under the existing composite score
(0.26788 versus 0.27002). Step 1,000 has a slightly better raw mean CER and lower
repetition; step 2,500 has better median CER and mean/median WER. Their paired
bootstrap intervals overlap on most aggregate differences.

## Methodological misalignment that remains

This is not data leakage, but it prevents a final checkpoint claim:

1. evaluation used `do_sample=True`;
2. the YAML recorded seed 42, but the completed evaluator did not set Python,
   NumPy or PyTorch RNG seeds;
3. only one stochastic generation was evaluated per checkpoint;
4. the YAML recorded a fixed 350-token limit while the evaluator used
   `min(380, max(80, word_count * 20))`;
5. no blinded native-Romanian listening study has been completed.

The evaluator and configuration now expose and record the actual seed and token
policy for future runs. Existing audio and metrics remain labeled as the original
unseeded single-run evidence rather than being retroactively reclassified.

## Decision

- Preserve steps 1,000, 2,000 and 2,500.
- Treat step 2,500 as the provisional release candidate.
- Treat step 1,000 as the stable-mean challenger.
- Do not publish a final model-card champion claim until seeded repeated
  Full-200 evaluation and native-listener comparison are complete.

