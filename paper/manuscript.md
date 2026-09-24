# Adapting Qwen3-Omni for Romanian Speech Generation on a 16 GB Consumer GPU

**Living manuscript / preprint draft — T4 evaluation complete**

## Abstract

We investigate whether the speech-generation path of Qwen3-Omni can be adapted to Romanian without training the full multimodal model and without datacenter-class GPU memory. We first test a practical waveform-to-target-code bridge using Mimi-derived 16-stream acoustic tokens and the released Qwen Code2Wav decoder. We then conduct controlled ablations of MTP-only, Talker-only and joint Talker+MTP LoRA adaptation on an RTX 5080 16 GB. MTP-only does not improve autonomous Romanian generation, while Talker-only corrects termination behavior but remains linguistically poor. Joint adaptation on approximately one hour of Romanian produces a large held-out improvement and fits in approximately 12.2 GB VRAM. A controlled five-hour scaling experiment, initialized from the same stock base with fresh adapters, selects step 2,500 as the multi-metric champion. On Full-200 it reaches 30.73% mean CER and 22.85% median CER, relative reductions of 20.3% and 22.0% versus the one-hour T3 control.

The work is maintained as an evidence-first research log. We distinguish practical/community evidence for Mimi compatibility from official model documentation and separate private-data experiments from release-cleared data.

## 1. Introduction

Romanian has strong multilingual text and ASR support, but high-quality conversational speech generation as part of a multimodal, potentially duplex model remains less accessible.

Rather than replacing Qwen3-Omni's speech path with an independent TTS stack, we ask whether its released Talker/MTP/Code2Wav path can be adapted directly under practical constraints:

- primary gradient hardware: RTX 5080 16 GB;
- resumable/preemptible training;
- prompt-text-only held-out generation;
- independent ASR plus native listening;
- strict experiment separation.

## 2. Research Questions

See `../RESEARCH_QUESTIONS.md`.

## 3. Target Representation

Two independent third-party Qwen3-Omni-derived training implementations use Mimi at 24 kHz and the first 16 codebooks. We therefore test this assumption empirically rather than treating it as official documentation.

A 30-sample bridge test finds a small WER increase from original audio (~5.8%) to Mimi-code → Qwen Code2Wav reconstruction (~6.4%), supporting use of these codes as training targets.

## 4. Experiments

### T0 — Stock
Poor Romanian generation and frequent termination failures.

### T1 — MTP-only
No overall improvement.

### T2 — Talker-only
Explicit EOS supervision corrects termination but is insufficient for Romanian quality.

### T3 — Joint Talker + MTP, ~1 h
Peak training VRAM ~12.16 GB.

Full-200:
- mean CER 38.53%
- median CER 29.30%
- mean WER 87.85%
- median WER 75.00%
- EOS 100%
- repetition 1.5%

### T4 — ~5 h controlled scaling
Same architecture as T3, but fresh adapters from the stock base.

Intermediate Quick-40:
- step 1000: mean CER 26.03%, median 24.68%
- step 1500: mean CER 26.50%, median 25.44%
- step 2000: median CER 22.74%, with one catastrophic loop

Full-200 evaluation selects step 2,500 as the multi-metric champion: mean CER 30.73%, median CER 22.85%, mean WER 75.81%, median WER 63.64%, 100% EOS success and 1.0% repetition.

## 5. Evaluation

The main held-out benchmark contains 200 Romanian prompts across ten phonetic/linguistic categories. We report mean/median/tail ASR error together with EOS and repetition failures because rare collapses can dominate arithmetic means.

Ground-truth acoustic tokens are prohibited at benchmark inference.

## 6. Current Conclusions

Current evidence supports:

1. a practical 16-stream target-token bridge;
2. meaningful Talker/MTP adaptation on a 16 GB RTX 5080;
3. joint adaptation clearly outperforming single-component ablations;
4. a larger data condition improving Full-200 mean and median CER over the one-hour control.

Claims about production quality, scaling beyond five hours, full-model parity and duplex behavior remain open.

## 7. Limitations

- ASR error is an incomplete measure of naturalness.
- T3 absolute WER remains high.
- current principal training data is narrow/single-speaker.
- full frozen-Thinker conditioning parity is not yet demonstrated.
- community Mimi compatibility is not the same as an official Qwen specification.
- public release rights must be evaluated separately from private research feasibility.

## 8. Future Work

- conduct native-listener evaluation of the T4 champion;
- add larger controlled scaling point if useful;
- compare equal-duration natural narration vs clean TTS-style speech;
- add native-human preference;
- test full BF16 Thinker conditioning;
- test preservation across other languages;
- investigate interruption, overlap and backchannels.

## 9. Reproducibility

See `../docs/reproducibility.md`.
