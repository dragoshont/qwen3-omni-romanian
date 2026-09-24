# T3 — Joint Talker + MTP, ~1 Hour

## Configuration

Talker:
- LoRA rank 8
- alpha 16
- q/v projections

MTP:
- LoRA rank 8
- alpha 16
- q/v/o + gate/up/down projections

Frozen:
- Thinker
- Code2Wav
- Mimi encoder

Peak training VRAM: ~12.16 GB.

## Quick-40

- mean CER 30.86%
- median CER 28.49%
- mean WER 78.97%
- median WER 78.36%
- EOS 100%
- max-token 0%
- repetition 0%

Compared with T2:
- CER win: 35/40
- WER win: 33/40

## Full-200

- mean CER 38.53%
- median CER 29.30%
- P90 CER 45.13%
- mean WER 87.85%
- median WER 75.00%
- P90 WER 110.82%
- EOS 100%
- max-token 0%
- repetition 1.5% (3/200)
- mean duration 4.47 s
- total generated audio 894.08 s
- inference peak 7.27 GB

Interpretation: T3 establishes a clear autonomous Romanian learning signal but is not a release-quality voice.
