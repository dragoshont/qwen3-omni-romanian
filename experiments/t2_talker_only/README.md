# T2 — Talker-only

Talker LoRA r8 / alpha16 on `q_proj,v_proj`, with explicit codec EOS supervision.

Training:
- ~1,000 steps
- loss ~8.79 → 3.29
- peak VRAM ~12.13 GB

Quick-40:
- mean WER 287%
- median WER 109.09%
- mean CER 221.49%
- median CER 55.55%
- EOS 100%
- max-token 0%
- repetition 27.5%

Interpretation: termination behavior was fixed, but Romanian quality remained poor.
