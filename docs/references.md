# Technical References

Main upstream projects that informed the current design:

## Qwen
- `QwenLM/Qwen3-Omni` — base multimodal/omni architecture.
- `QwenLM/Qwen3-TTS` — official TTS stack and fine-tuning reference.
- `QwenLM/Qwen3-ASR` — secondary Romanian-capable ASR evaluator.

## Talker training / codec evidence
- `Hert4/Qwen-Omni-Training-Talker` — community Talker training using Mimi, 24 kHz audio, first 16 code streams and frozen Code2Wav.
- `DuplexOmni` — Qwen3-Omni-derived speech/duplex research implementation and practical Mimi evidence.

## Duplex reference
- `kyutai-labs/moshi-finetune`
- `nu-dialogue/moshi-finetune`

## Evaluation
- Whisper large-v3 / turbo — independent primary ASR evaluator.
- Qwen3-ASR — secondary opinion, not sole evaluator for a Qwen-family generator.

Formal bibliographic metadata will be added before preprint publication. Exact commit pins are in `reproducibility.md`.
