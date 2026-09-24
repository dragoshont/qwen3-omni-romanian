# Qwen3-Omni Romanian Speech Adaptation

> **Research repository — completed T4 results, 24 September 2026.**
>
> Ongoing evidence-first work on adapting the speech-generation path of **Qwen3-Omni** to Romanian on consumer hardware. The goal is reproducibility and falsifiable experiments, not a polished demo.

## Target

The long-term target is a Qwen3-Omni system that can hold **natural, intelligent spoken conversations in Romanian**: understand Romanian speech, reason over the conversation and respond directly with fluent Romanian speech, while preserving the broader multimodal architecture.

This repository currently addresses the speech-generation part of that goal. It tests whether the released Talker/MTP path can be adapted to Romanian reproducibly on a 16 GB consumer GPU. It does **not** yet claim a complete low-latency duplex assistant; interruption, backchannels and simultaneous listening/speaking remain later research stages.

Code, model adapters, datasets and archival research snapshots have different publication requirements. See the [artifact publication plan](docs/artifact-publication-plan.md) for what belongs on GitHub, Hugging Face and Zenodo.

## Current status

The core hypothesis is working:

- a practical 16-stream Romanian target-code bridge has been validated through frozen Qwen Code2Wav;
- real Talker + MTP adaptation fits on an RTX 5080 16 GB at roughly 12–13 GB VRAM;
- a joint ~1-hour Romanian run (T3) produces autonomous Romanian speech;
- the controlled ~5-hour run (T4) improves both mean and median Full-200 CER over T3.

**T4 is complete. Step 2,500 is the declared multi-metric champion:** 30.73% mean CER, 22.85% median CER, 100% EOS success and 1.0% repetition on the Full-200 benchmark.

## Why this exists

Qwen3-Omni offers a useful starting point for a Romanian conversational speech model: multimodal semantic reasoning, a speech-generation Talker, MTP prediction for additional acoustic streams, and a waveform decoder.

The main research questions are:

1. Can valid Romanian target speech tokens be constructed for the released Talker/MTP path?
2. Can meaningful adaptation be trained on a **16 GB consumer GPU**?
3. Does Romanian quality improve as clean data scales?
4. Can this be done while preserving the broader Omni architecture rather than replacing it with standalone TTS?
5. Later: can the same system support low-latency duplex interaction, interruption and backchannels?

## Headline evidence

### 1. Codec bridge

In an initial 30-clip test, 24 kHz Romanian speech was encoded with Mimi, the first 16 codebooks were retained, and those codes were decoded with Qwen3-Omni Code2Wav.

Independent ASR measured approximately:

- original recordings: **5.8% WER**
- Mimi → Qwen Code2Wav reconstruction: **6.4% WER**

This strongly supports the practical target-token strategy.

**Important:** this repository does not claim that Qwen officially documents Mimi as the canonical Qwen3-Omni target encoder. The evidence is practical/community-derived plus our own bridge test.

### 2. Consumer-GPU training

Real Talker/MTP training, not just a toy dry run, occupies approximately:

- T2: **12.13 GB**
- T3: **12.16 GB**
- T4: **12.57 GB**

on an RTX 5080 16 GB.

### 3. Joint adaptation works

The controlled experiment sequence is:

```text
T0  stock speech path
 |
 +-- T1  MTP-only
 |
 +-- T2  Talker-only + explicit codec EOS
 |
 +-- T3  joint Talker + MTP, ~1 h Romanian
 |
 +-- T4  same architecture, fresh adapters, ~5 h Romanian
```

The principal T3/T4 architecture is:

```text
Qwen3-Omni base
├─ Thinker                         frozen
├─ Talker
│  └─ LoRA r=8, alpha=16
│     q_proj, v_proj
├─ MTP
│  └─ LoRA r=8, alpha=16
│     q_proj, v_proj, o_proj,
│     gate_proj, up_proj, down_proj
└─ Code2Wav                       frozen

Talker base: NF4 / 4-bit
Targets: 16 codec streams
```

## T3 result

### Quick-40

| Metric | T3 |
|---|---:|
| Mean CER | 30.86% |
| Median CER | 28.49% |
| Mean WER | 78.97% |
| Median WER | 78.36% |
| EOS success | 100% |
| Max-token termination | 0% |
| Repetition loops | 0% |

### Full-200

| Metric | T3 |
|---|---:|
| Mean CER | 38.53% |
| Median CER | 29.30% |
| P90 CER | 45.13% |
| Mean WER | 87.85% |
| Median WER | 75.00% |
| P90 WER | 110.82% |
| EOS success | 100% |
| Max-token termination | 0% |
| Repetition loops | 1.5% (3/200) |
| Mean generated duration | 4.47 s |
| Peak inference VRAM | 7.27 GB |

This is evidence that adaptation works. It is **not** yet a production-quality Romanian voice.

## T4 result

T4 starts from the same stock Qwen base with **fresh adapters**. It is not a continuation from T3. Step 2,500 was selected by the declared composite rule after Full-200 evaluation of steps 1,000, 2,000 and 2,500.

| Full-200 run | Mean CER | Median CER | P90 CER | Mean WER | Median WER | EOS | Repetition |
|---|---:|---:|---:|---:|---:|---:|---:|
| T3 control | 38.53% | 29.30% | 45.13% | 87.85% | 75.00% | 100% | 0.0% |
| T4 step 1000 | **30.09%** | 23.91% | 44.28% | 76.10% | 70.00% | 100% | 0.5% |
| T4 step 2000 | 36.00% | 23.46% | **39.24%** | 77.62% | 65.99% | 100% | 1.0% |
| T4 step 2500 | 30.73% | **22.85%** | 41.47% | **75.81%** | **63.64%** | 100% | 1.0% |

Compared with T3, the selected step-2,500 checkpoint reduces mean CER by 20.3% and median CER by 22.0%. The full comparison, including category breakdowns and tail metrics, is in [`reports/t4_final_champion_declaration.md`](reports/t4_final_champion_declaration.md).

This is a research result, not production-quality Romanian speech. ASR error remains high, category performance is uneven and native-listener evaluation is still required.

## Evaluation

A held-out generation only counts when inference receives **prompt text only**. Ground-truth codec tokens are forbidden at inference.

The Romanian Full-200 benchmark has 20 unseen prompts in each of ten categories:

1. conversational Romanian;
2. ă/â/î;
3. ș/ț;
4. ce/ci/ge/gi/che/chi/ghe/ghi;
5. consonant clusters;
6. numbers, dates and currency;
7. Romanian names and places;
8. English technical loanwords / code-switching;
9. questions and exclamations;
10. long natural sentences.

Metrics include independent Whisper WER/CER, mean/median/tail errors, EOS success, max-token termination, repetition/collapse rate, latency and memory. Human native-speaker evaluation is planned because ASR is not a complete measure of speech quality.

## Repository map

```text
.
├── README.md
├── STATUS.md
├── ROADMAP.md
├── RESEARCH_QUESTIONS.md
├── DATA_PROVENANCE.md
├── LEGAL.md
├── CITATION.cff
├── configs/
├── data/
├── docs/
├── eval/
├── evidence/
├── experiments/
├── models/
├── paper/
└── scripts/
```

## Data and weights

This public scaffold intentionally contains **no copyrighted audiobook audio or book text**, no redistributed Qwen base weights, and no rights-unclear voice checkpoint.

A private audiobook-alignment research lane is documented separately, but any public release of data or weights gets a separate rights review.

## What feedback would be most useful?

Please open an issue if you can reproduce, falsify or improve any of these:

- Mimi/Qwen codec compatibility;
- Talker/MTP target alignment;
- EOS supervision;
- rare catastrophic loops during later checkpoints;
- better Romanian corpora with clear provenance;
- low-VRAM training;
- frozen-Thinker conditioning parity;
- paths toward duplex Romanian conversation.

## Status

**Research preview / pre-paper.**

Publishing early is deliberate: assumptions are cheapest to correct before the experiment grows.
