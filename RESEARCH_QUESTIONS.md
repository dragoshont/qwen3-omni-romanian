# Research Questions

## RQ1 — Target codec construction
Can Romanian waveform targets be encoded into a representation compatible with the released Qwen3-Omni Talker/MTP + Code2Wav path?

**Current evidence:** strong practical support for a Mimi-derived 16-stream path.  
**Caveat:** this is empirical/community evidence, not an official Qwen statement.

## RQ2 — Consumer GPU feasibility
Can meaningful Qwen3-Omni speech adaptation be trained on a 16 GB RTX-class GPU?

**Current evidence:** yes. Real T3/T4 training is approximately 12–13 GB VRAM.

## RQ3 — Component responsibility
Is Romanian adaptation mainly obtained by adapting MTP, Talker, or both?

**Current evidence:** MTP-only did not improve overall quality; Talker-only fixed termination but remained linguistically poor; joint Talker + MTP was decisively better.

## RQ4 — Data scaling
With architecture and training recipe fixed, how does held-out Romanian quality change from ~1 h → ~5 h → larger corpora?

**Current evidence:** T4 intermediate checkpoints improve over T3 on Quick-40. Full-200 is pending.

## RQ5 — Data quality vs quantity
At equal duration, does professionally narrated Romanian outperform a short-utterance TTS corpus for intelligibility, prosody and long-form language?

**Status:** planned secondary experiment.

## RQ6 — Full-model conditioning
Can Talker/MTP be trained from cached frozen Thinker features and reproduce full end-to-end behavior?

**Status:** planned.

## RQ7 — Duplex behavior
Can the adapted speech path eventually support interruption, overlap and backchannels while preserving Qwen's broader semantic architecture?

**Status:** future work.
