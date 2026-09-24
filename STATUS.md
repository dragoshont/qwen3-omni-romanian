# Status

_Last updated: 2026-09-24 21:25 EEST_

## Completed experiment

**T4 — controlled ~5-hour data-scaling experiment**

- completed: 2,500 optimizer steps
- gradient accumulation: 4
- peak training VRAM: ~12.57 GB
- Full-200 candidates evaluated: steps 1,000, 2,000 and 2,500
- provisional champion: step 2,500
- provisional-candidate mean / median CER: 30.73% / 22.85%
- provisional-candidate mean / median WER: 75.81% / 63.64%
- EOS success: 100%
- repetition rate: 1.0%

## Interpretation

Relative to the T3 one-hour control, T4 step 2,500 reduces mean CER by 20.3% and median CER by 22.0% on the same Full-200 prompt set. Step 2,500 narrowly wins the existing composite score, while step 1,000 retains a slightly better raw mean CER and lower repetition. Because each checkpoint has only one unseeded stochastic run, step 2,500 remains provisional.

See [`reports/t4_verification_audit.md`](reports/t4_verification_audit.md) for the integrity audit and [`reports/t4_final_champion_declaration.md`](reports/t4_final_champion_declaration.md) for the provisional comparison.

The immediate gate is seeded repeated Full-200 evaluation of steps 1,000 and 2,500, followed by native-listener evaluation. Clean-environment reproduction and release-rights review follow after checkpoint selection is frozen.
