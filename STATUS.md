# Status

_Last updated: 2026-09-24 21:25 EEST_

## Completed experiment

**T4 — controlled ~5-hour data-scaling experiment**

- completed: 2,500 optimizer steps
- gradient accumulation: 4
- peak training VRAM: ~12.57 GB
- Full-200 candidates evaluated: steps 1,000, 2,000 and 2,500
- declared champion: step 2,500
- champion mean / median CER: 30.73% / 22.85%
- champion mean / median WER: 75.81% / 63.64%
- EOS success: 100%
- repetition rate: 1.0%

## Interpretation

Relative to the T3 one-hour control, T4 step 2,500 reduces mean CER by 20.3% and median CER by 22.0% on the same Full-200 protocol. Step 2,000 has a slightly better median than step 1,000 but a worse mean and more category instability. Step 2,500 is therefore the declared multi-metric champion.

See [`reports/t4_final_champion_declaration.md`](reports/t4_final_champion_declaration.md) for the complete comparison and category breakdown.

The next gates are native-listener evaluation, clean-environment reproduction and release-rights review for adapters and audio artifacts.
