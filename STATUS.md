# Status

_Last updated: 2026-09-24 18:28 EEST_

## Active experiment

**T4 — controlled ~5-hour data-scaling experiment**

- step at snapshot: ~2,140 / 2,500
- gradient accumulation: 4
- pace: ~5.4 s / optimizer step
- VRAM: ~12.57 GB
- joint loss: ~5.64
- Talker loss: ~2.37
- MTP loss: ~3.08
- final Quick-40: pending
- Full-200: pending

## Interpretation

T4 checkpoints 1,000 and 1,500 improve cleanly over the T3 1-hour Quick-40 control under identical decoding.

Checkpoint 2,000 reaches the lowest median CER/WER so far but contains one catastrophic loop, so it is not automatically the champion.

Planned completion:

1. finish step 2,500;
2. evaluate Quick-40;
3. retain the best stable-mean candidate and best robust/median candidate;
4. if they differ, run Full-200 on both;
5. declare a T4 champion only after Full-200.
