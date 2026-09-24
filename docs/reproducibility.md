# Reproducibility

## Recorded upstream revisions

- QwenLM/Qwen3-Omni: `e4235853125589c789f06a2dd83e9f4126df5e9d`
- QwenLM/Qwen3-TTS: `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- QwenLM/Qwen3-ASR: `7c6daf77a2421100f5fb066495372c00129d39ff`
- Hert4/Qwen-Omni-Training-Talker: `d9ad37e150a8e93d16169449d33b48a3141372bf`
- DuplexOmni: `33bfba1a821b09c5aa66790944f9098584979d34`
- kyutai-labs/moshi-finetune: `2acc879fe7c48f885a18f6cc9548bccb2674d87b`
- nu-dialogue/moshi-finetune: `3879d293585786677e6d6f4530e67dbebbe133f2`

## Environment separation

Research used separate environments because the relevant projects currently require different Transformers generations:

- Omni: Python ~3.12, Transformers >=5.2
- Qwen3-TTS: around Transformers 4.57.3
- Qwen3-ASR: around Transformers 4.57.6
- Mimi/evaluation: isolated as required

Exact lockfiles will replace these notes before a reproducibility release.

## A publishable experiment should include

- exact environment lock;
- upstream revisions;
- dataset manifest and source IDs/hashes;
- held-out exclusion proof;
- training config;
- random seeds;
- checkpoint metadata;
- generation config;
- per-sample metrics;
- aggregate metrics;
- timing and peak memory;
- redistributable audio examples where permitted.

A generated benchmark sample must be produced from prompt text only.
