# Romanian Qwen3-Omni Research Audit Archive
**Archive Date:** September 24, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 5080 (16 GB GDDR7 VRAM, 32 GB RAM, Windows 11)  
**Software Environment:** PyTorch 2.11.0+cu128, Transformers 5.17.0, PEFT 0.14.0, BitsAndBytes 0.45.3  

---

## 1. What Experiments Were Run

| Experiment ID | Focus | Checkpoint / Outputs | Key Outcome |
| :--- | :--- | :--- | :--- |
| **EXP_00** | Preflight Environment Verification | `reports/morning_audit.md` | PyTorch 2.11.0+cu128 Blackwell SM120 native execution verified on RTX 5080. |
| **EXP_01** | Surgical Code2Wav Extraction | `models/code2wav.safetensors` | Isolated 216M-parameter synthesizer strictly from shard 15 without 70 GB full download. |
| **EXP_02** | Codec Token Bridge | `outputs/mimi_baseline/` | Encoded Romanian audio into 16-codebook 12.5 Hz Mimi discrete tokens. |
| **EXP_03** | Codec Cross-Decode Gate | `outputs/qwen_code2wav_from_mimi/` | **PASS**: 27/30 clips intelligible (90.0%). Qwen WER 6.4% vs Original 5.8%. Proved Mimi tokens drive Qwen Code2Wav. |
| **EXP_04** | Talker Architecture Analysis | `models/qwen3-omni-talker/` | Exported standalone 3.32B MoE Talker; measured BF16 VRAM ceiling at layer 16 (9.96 GB) on Windows. |
| **EXP_05** | 4-Bit Talker + MTP QLoRA Backprop Gate | `reports/talker_qlora_memory_test.json` | **PASS**: 4-bit NF4 Talker + LoRA forward, backward, and optimizer step verified at only **5.88 GB peak VRAM** (10.01 GB headroom). |
| **EXP_06** | Romanian MTP LoRA Scaling (ro30 ➔ ro150 ➔ ro500) | `models/romanian_mtp_lora_ro500/final` | Scaled to 500 clips (7,500 steps, loss 3.40 ➔ 3.06). CER improved from 51.4% to 31.9%. |
| **EXP_07** | Controlled Matrix A0: Stock Baseline | `outputs/T0_stock/`, `reports/T0_stock_benchmark.json` | Evaluated 40 held-out sentences. Identified that stock Talker lacks Romanian EOS stopping policy (Mean CER 209.6%, Repetition 22.5%). |
| **EXP_08** | Controlled Matrix A1: MTP-Only Control | `outputs/T1_mtp_only/`, `reports/T1_mtp_benchmark.json` | Stock Talker + Romanian MTP evaluated on 40 held-out sentences. Confirmed MTP alone improves acoustics but cannot fix stream 0 stopping. |
| **EXP_09** | Controlled Matrix B1: Romanian Talker-Only LoRA | `models/T2_talker_only/` | 4-bit Talker LoRA training with explicit Romanian EOS supervision on 1h Romanian speech (`ro1000`). Loss dropped 8.79 ➔ 3.42. |
| **EXP_10** | Controlled Matrix B2: Joint Talker + MTP LoRA (T3) | `models/T3_talker_mtp/final` | Joint training of stream 0 fundamental cadence and streams 1..15 harmonic residuals. Won 35/40 (87.5%) head-to-head vs B1. Quick-40 Mean CER: 30.86%, 0% repetition. |
| **EXP_11** | Full 200 Held-Out Benchmark on T3 Champion | `reports/champion_200_benchmark.json` | Evaluated 200 held-out sentences across 10 categories. Mean CER: 38.53%, Median CER: 29.30%, EOS: 100%, Repetition: 1.5%. Passed all 6 project scaling gate criteria. |
| **EXP_12** | Phase T4 Data Scaling (5-Hour Corpus, 2,500 Steps) | `models/T4_data_scale_5h/final/` | Deterministically built 5.00-hour manifest (4,311 clips, 18,003s) with zero held-out overlap. Trained fresh LoRA on stock base for 2,500 optimizer steps. Saved checkpoints at steps 500, 1000, 1500, 2000, 2500. |
| **EXP_13** | Dual Full-200 Benchmark Matrix | `reports/t4_final_champion_declaration.md` | Multi-metric evaluation on Step 1000 (stable-mean), Step 2000 (median-CER), and Step 2500 across 200 held-out sentences. |

---

## 2. Checkpoint Preservation and Champion Selection

To prevent selection bias from a single aggregate metric, candidate checkpoints are preserved and evaluated under a multi-metric Pareto protocol:

1. **`models/preserved_checkpoints/T3_talker_mtp_step1000`**:
   - 1-Hour Control baseline (Step 1000). Full-200 Mean CER: 38.53%, Median CER: 29.30%.
2. **`models/preserved_checkpoints/T4_checkpoint_step_1000`**:
   - T4 5-hour data-scale checkpoint at Step 1000. Best early stable-mean checkpoint (Quick-40 Mean CER: 26.03%, Repetition: 0.0%).
3. **`models/preserved_checkpoints/T4_checkpoint_step_2000`**:
   - T4 5-hour data-scale checkpoint at Step 2000. Best median-CER checkpoint (Quick-40 Median CER: 22.74%).
4. **`models/preserved_checkpoints/T4_checkpoint_step_2500`**:
   - T4 5-hour data-scale final completion checkpoint. Full-200 Mean CER: 30.73%, Median CER: 22.85%, Repetition: 1.0%.

---

## 3. Which Generated Audio is True Held-Out Text-to-Speech

- **`outputs/champion_200/*.wav`** and **`outputs/t4_step2500_full200/*.wav`** (200 audio files):  
  **True Held-Out TTS (Level 3)**. Generated strictly from the 200 unseen Romanian text prompts in `eval/ro_holdout_200.jsonl` covering all 10 categories. Zero target audio or ground-truth codes were supplied.
- **`outputs/t4_step1000_full200/*.wav`** & **`outputs/t4_step2000_full200/*.wav`** (200 audio files each):  
  **True Held-Out TTS (Level 3)** for candidate evaluation.
- **`outputs/T0_stock/*.wav`** & **`outputs/T1_mtp_only/*.wav`** (40 audio files each):  
  **True Held-Out TTS (Level 3)** for controlled quick-40 ablation matrix.

---

## 4. What Was Actually Trained

1. **Phase T3 Architecture (Preserved in T4)**:
   - Talker LoRA: rank 8, alpha 16, target modules: `q_proj`, `v_proj` (655,360 trainable parameters).
   - MTP LoRA: rank 8, alpha 16, target modules: `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` across all 5 code predictor layers (1,064,960 trainable parameters).
   - Total trainable parameters: **1,720,320** (~0.05% of model).
2. **Backbone Quantization**:
   - Talker backbone: 4-bit NF4 quantized via BitsAndBytes.
   - Peak training VRAM: **5.88 GB - 7.30 GB** (over 8 GB headroom on RTX 5080 16GB).
3. **100% Frozen Modules**:
   - `Thinker` (multimodal backbone) — frozen.
   - `Code2Wav` (216M-parameter waveform synthesizer) — frozen.
   - Base Talker weights — frozen.

---

## 5. Provenance and Integrity

- **Level 3 Autonomous TTS**: Generated text-to-speech without any teacher forcing, ground truth codes, or reference audio.
- **Independent Evaluation**: Transcriptions and error rates computed using independent `openai/whisper-large-v3-turbo` in Romanian language mode (`ro`).
- **Data Isolation**: Strict exclusion applied between training manifests and all 200 held-out sentences.
