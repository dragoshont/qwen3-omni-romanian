# Methodology

## Principle

Change one major variable at a time and preserve enough artifacts to explain every result.

## T0 — stock baseline

Quick-40:
- mean WER 218.65%
- median WER 100%
- mean CER 209.57%
- median CER 55.61%
- EOS 85%
- max-token 15%
- repetition 22.5%

## T1 — MTP only

Quick-40:
- mean WER 240.65%
- median WER 100%
- mean CER 244.99%
- median CER 73.91%
- EOS 65%
- max-token 35%
- repetition 22.5%

Interpretation: MTP-only is insufficient.

## T2 — Talker only

Talker LoRA r8 / alpha16 on q/v with explicit `codec_eos_token_id` supervision.

- training loss ~8.79 → 3.29 over 1,000 steps
- peak VRAM ~12.13 GB

Quick-40:
- mean WER 287%
- median WER 109.09%
- mean CER 221.49%
- median CER 55.55%
- EOS 100%
- max-token 0%
- repetition 27.5%

Interpretation: termination was learned; Romanian quality remained poor.

## T3 — joint Talker + MTP, ~1 h

- Talker LoRA r8/a16, q/v
- MTP LoRA r8/a16, q/v/o + gate/up/down
- Thinker frozen
- Code2Wav frozen
- peak VRAM ~12.16 GB

Quick-40:
- mean CER 30.86%
- median CER 28.49%
- mean WER 78.97%
- median WER 78.36%
- EOS 100%
- repetition 0%

T3 beat T2 on CER for 35/40 prompts and WER for 33/40.

## T4 — ~5 h, same architecture

T4 uses the same stock base and **fresh adapters**. It does not continue from T3.

Target:
- ~18,000 seconds / ~5 h
- ~3,350 utterances
- 2,500 optimizer steps
- gradient accumulation 4
- ~3 effective passes
- checkpoints at 500/1000/1500/2000/2500

Quick-40 screens checkpoints. Full-200 confirms the final candidate.

## Controlled decoding

- seed 42
- temperature 0.8
- top-k 50
- top-p 0.9
- repetition penalty 1.15
- max tokens 350
- codec EOS token id 2148

Ground-truth codec tokens are forbidden during benchmark generation.
