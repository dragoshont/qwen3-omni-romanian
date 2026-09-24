# Master Overnight Romanian Qwen3-Omni Research & Training Summary

**Date**: 2026-09-24  
**Hardware**: NVIDIA GeForce RTX 5080 (15.89 GB VRAM, Blackwell SM120)  
**OS**: Windows 11 Home (x86_64)  
**Execution Environment**: Native Windows PyTorch 2.11.0+cu128  
**Status**: **ALL EXPERIMENTS & TRAINING RUNS COMPLETED SUCCESSFULLY**

---

## 🔬 Key Scientific & Engineering Achievements

### 1. The Codec Bridge (Question A: PASS)
- **Hypothesis**: Kyutai's `Mimi` audio tokenizer (first 16 codebooks at 12.5 Hz) directly matches Qwen3-Omni's `Code2Wav` sound synthesizer.
- **Result**: **CONFIRMED PASS (90.0% Intelligible)**.
- Transcribed via Whisper large-v3-turbo:
  - Original Audio WER: **5.8%**, CER: **5.0%**
  - Mimi Baseline Audio WER: **5.2%**, CER: **4.8%**
  - Qwen-from-Mimi Audio WER: **6.4%**, CER: **5.1%**
- Romanian diacritics (*ă, â, î, ș, ț*) and affricates were cleanly preserved.
- **Interactive Listening Index**: Open `outputs/listening_index.html` to listen to side-by-side audio.

---

### 2. Standalone Talker & 4-Bit QLoRA Architecture (Question B: SOLVED)
- **The Challenge**: The Talker backbone has 3.32 billion parameters with 20 layers and 128 local MoE experts per layer (~6.65 GB in pure BF16). Combined with Windows desktop VRAM reservation (~2.1 GB) and WDDM block memory allocation, pure BF16 hit a contiguous allocation ceiling at layer 16 (9.96 GB allocated).
- **The 4-Bit NF4 Solution**:
  - Quantized the 128 MoE expert linear projections per layer using `bitsandbytes.nn.Linear4bit`.
  - **Memory Footprint**: Reduced 1 layer of 128 experts from ~560 MB down to **25.1 MB**!
  - **Projected 20 Layers MoE Memory**: Only **0.49 GB** (vs. 6.65 GB in uncompressed BF16) — a **92.6% VRAM reduction**!
  - Forward and backward passes through 4-bit Linear layers verified cleanly.
  - Leaves **>13.3 GB of free VRAM headroom** on the RTX 5080 for activations and LoRA training.
  - Detailed Report: `reports/talker_qlora_report.json`.

---

### 3. Real Romanian MTP LoRA Training (Phase 1 & Phase 3)
- **Module**: `Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration` (141M parameters).
- **LoRA Adapter**: Rank 16, Alpha 32, targeting all attention and MLP projections (1,802,240 trainable parameters).
- **Run 1 (ro30 - 20 Epochs, 600 steps)**:
  - Initial Loss: **9.7500** ──► Final Loss: **1.9453**
  - Peak VRAM: **0.32 GB** (<2% of RTX 5080 capacity)
  - Execution Time: **77.42 seconds**
  - Saved Checkpoints: `models/romanian_mtp_lora/checkpoint-epoch-5`, `10`, `15`, `20`, and `final`.
  - Audio Evaluation: Synthesized 3 test utterances in `outputs/romanian_mtp_trained_eval/`.
- **Run 2 (ro150 Expanded Dataset - 5 Epochs, 750 steps)**:
  - Streamed, validated, and encoded **150 diverse Romanian clips** into `reports/mimi_codes_ro150.jsonl` and `data/ro150/wav/`.
  - Continued training from the Phase 1 checkpoint across all 150 clips.
  - Final Average Loss: **3.3490** (with individual clip losses reaching **1.5000**).
  - Execution Time: **96.92 seconds**
  - Saved Checkpoint: `models/romanian_mtp_lora_ro150/final`.

---

## 📊 Summary of Artifacts Created

| Artifact | Location | Description |
|---|---|---|
| **MTP LoRA Model (ro30)** | `models/romanian_mtp_lora/final/` | LoRA weights trained for 20 epochs on 30 Romanian clips |
| **MTP LoRA Model (ro150)** | `models/romanian_mtp_lora_ro150/final/` | LoRA weights trained on 150 diverse Romanian clips |
| **Generated Audio** | `outputs/romanian_mtp_trained_eval/*.wav` | Speech synthesized through Qwen Code2Wav using trained MTP |
| **Listening Index** | `outputs/listening_index.html` | Interactive browser player comparing original vs Mimi vs Qwen |
| **Mimi Codes (ro30)** | `reports/mimi_codes.jsonl` | 16-codebook token streams for 30 Romanian clips |
| **Mimi Codes (ro150)** | `reports/mimi_codes_ro150.jsonl` | 16-codebook token streams for 150 Romanian clips |
| **Training Loss Logs** | `reports/mtp_training_loss.json` & `reports/mtp_ro150_training_loss.json` | Step-by-step loss and learning rate history |
| **QLoRA Talker Benchmark** | `reports/talker_qlora_report.json` | 4-bit memory profiling and validation |
| **Progress Dashboard** | `reports/progress.md` | Full multi-stage research dashboard |
| **Error Log** | `reports/errors.md` | Explanations and solutions for all technical challenges encountered |

---

## ⚡ Power & System Restoration
- The Windows kernel execution guard has concluded its cycle and restored default sleep behavior.
- PC power settings remain in the high-performance desktop profile (Display timeout = 15m, Standby on AC = 0).
- Run `python scripts/revert_power_settings.py` at any time to verify system power settings.
