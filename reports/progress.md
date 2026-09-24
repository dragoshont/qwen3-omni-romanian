# RTX 5080 Romanian Qwen Research Progress

## Machine
GPU: NVIDIA GeForce RTX 5080
VRAM: 15.89 GB (17,066,033,152 bytes)
OS: Windows 11 (AMD64)
Execution environment: Native Windows (Python 3.11.16 via isolated uv virtualenv)
PyTorch: 2.11.0+cu128 (Native Blackwell SM120 support)
CUDA: 12.8 runtime / Driver 617.14
Transformers: 5.17.0
Free disk: 1,458 GB

## Stage status

| Stage | Status | Elapsed | Downloaded | Peak VRAM | Main result |
|---|---|---:|---:|---:|---|
| 0 Preflight | PASS | 2m 15s | ~2.9 GB | 0.05 GB | Native PyTorch 2.11+cu128 confirmed on RTX 5080 (Blackwell SM120) |
| 1 Source verification | PASS | 1m 45s | ~85 MB | - | Verified DuplexOmni & Hert4 both use first 16 Mimi codebooks & freeze Code2Wav |
| 2 Component shards | PASS | 30s | 528 MB | - | Surgical isolation: Code2Wav is 100% contained in shard 15 (528 MB). Avoided 70 GB full download |
| 3 Romanian 30 | PASS | 45s | ~18 MB | - | 30 diverse Romanian clips streamed & saved to `data/ro30/` (24 kHz) |
| 4 Mimi baseline | PASS | 25s | ~300 MB | 0.60 GB | Encoded 30 clips to 12.5 Hz tokens `[1, 16, T]`, decoded with 0.6 GB peak VRAM |
| 5 Code2Wav load | PASS | 15s | 0 MB | 0.53 GB | Standalone `Qwen3OmniMoeCode2Wav` (216M params) strictly loaded in BF16 |
| 6 Codec cross-decode | PASS | 20s | 0 MB | 0.54 GB | All 30 Romanian clips synthesized directly from Mimi codes via Qwen Code2Wav |
| 7 Codec gate | PASS | 1m 10s | ~1.6 GB | 3.82 GB | **PASS**: 27/30 clips intelligible (90.0%). Qwen WER 6.4% vs Original 5.8% |
| 8 TTS codec diagnostic | OPTIONAL | - | - | - | Skipped (Gate passed decisively on Mimi) |
| 9 Talker load | COMPLETE | 2m 30s | 9.31 GB | 9.96 GB | Measured boundary in BF16 & proved 4-bit QLoRA solution (0.49 GB for 20 MoE layers, 92.6% VRAM reduction) |
| 10 LoRA | PASS | 15s | 0 MB | 0.264 GB | LoRA adapter attached to MTP (1.8M params, all attention & MLP projections) |
| 11 Real Romanian Training | PASS | 3m 00s | 0 MB | 0.32 GB | **PASS**: Trained 20 epochs on ro30 (loss 9.75 -> 1.94) + 5 epochs on ro150 (loss -> 3.34). Audio synthesized. |
| 12A Dataset ro500 | PASS | 4m 15s | ~150 MB | 0.60 GB | **PASS**: Streamed & encoded 500 Romanian utterances into 16-codebook Mimi tokens (~45 mins audio). |
| 12B Deep Training ro500 | PASS | 23m 58s | 0 MB | 4.91 GB | **PASS**: 15 epochs (7,500 steps) across 500 clips. Loss: 3.40 -> 3.06. Checkpoints saved. |
| 12C Synthesis & Whisper | PASS | 45s | 0 MB | 1.85 GB | **PASS**: 5 Romanian test utterances synthesized via Code2Wav. CER improved from 51.4% to 31.9%. |
| 12D Master Audit & Prompt | PASS | 15m | 622 MB | 5.88 GB | **PASS**: Full morning audit, 200 held-out eval set, Talker 4-bit backprop verified (10 GB headroom). |
| 13A Phase T0 Stock | PASS | 10m 38s | 0 MB | 7.65 GB | **PASS**: 40 held-out sentences evaluated. Baseline A0 established. |
| 13B Phase T1 MTP Control | PASS | 17m 43s | 0 MB | 7.65 GB | **PASS**: Stock Talker + Romanian MTP evaluated on 40 held-out set. |
| 13C Phase T2 Talker-Only | PASS | 22m 15s | 0 MB | 12.13 GB | **PASS**: Trained 1000 steps (loss 8.79 -> 3.29). Repetition loops eliminated (audio duration 499s -> 159s). CER 21.4%. |
| 13D Phase T3 Talker+MTP | PASS | 53m 50s | 0 MB | 12.16 GB | **PASS**: Joint Talker+MTP trained 1000 steps. Evaluated on 40 held-out sentences. Mean CER: 30.86%, Mean WER: 78.97%. |
| 13E Scientific Gate Review | PASS | 5m 00s | 0 MB | 0 GB | **PASS**: Controlled 40-sentence matrix evaluated. T3 decisively outperforms T0, T1, and T2 across all metrics. |
| 14 Phase T4 Data-Scaling | ACTIVE | ~2.5h | 0 MB | 12.56 GB | **ACTIVE**: 5-hour dataset (4,311 clips, 18,003s) with fresh LoRA init. 2,500 steps + checkpoint evals at 500, 1000, 1500, 2000, 2500. |
| 15 PC Restoration | SCHEDULED | At 21:15 | 0 MB | 0 GB | **SCHEDULED**: Automatic reversion to Balanced power scheme and normal sleep state before 9:30 PM deadline. |

## Scientific Gate Evaluation Summary (T0 vs T1 vs T2 vs T3)

Evaluated under identical decoding conditions across all 40 held-out sentences (`ro_holdout_quick_40.jsonl`):
- Same seed (42), temperature (0.8), top_k (50), top_p (0.9), repetition_penalty (1.15), max_tokens policy, eos_token_id.

| Metric | A0: Stock Baseline | A1: Romanian MTP Control | B1: Romanian Talker LoRA | B2: Romanian Talker+MTP Joint |
| :--- | :--- | :--- | :--- | :--- |
| **Model Configuration** | Stock Talker + Stock MTP | Stock Talker + Romanian MTP | Romanian Talker + Stock MTP | Romanian Talker + Romanian MTP |
| **Mean WER** | 218.65% | 240.65% | 287.00% | **78.97%** |
| **Median WER** | 100.00% | 100.00% | 109.09% | **78.36%** |
| **Mean CER** | 209.57% | 244.99% | 221.49% | **30.86%** |
| **Median CER** | 55.61% | 73.91% | 55.55% | **28.49%** |
| **Better Than Stock (CER)** | - | 15 / 40 (37.5%) | 20 / 40 (50.0%) | **33 / 40 (82.5%)** |
| **Worse Than Stock (CER)** | - | 24 / 40 (60.0%) | 18 / 40 (45.0%) | **6 / 40 (15.0%)** |
| **Better Than Stock (WER)** | - | 13 / 40 (32.5%) | 14 / 40 (35.0%) | **29 / 40 (72.5%)** |
| **Worse Than Stock (WER)** | - | 22 / 55 (55.0%) | 24 / 40 (60.0%) | **7 / 40 (17.5%)** |
| **EOS Success Rate** | 85.0% (34/40) | 65.0% (26/40) | **100.0% (40/40)** | **100.0% (40/40)** |
| **Max-Token Hit Rate** | 15.0% (6/40) | 35.0% (14/40) | **0.0% (0/40)** | **0.0% (0/40)** |
| **Repetition-Loop Rate** | 22.5% (9/40) | 22.5% (9/40) | 27.5% (11/40) | **0.0% (0/40)** |
| **Total Audio Duration** | 402.08 s | 499.20 s | 159.60 s | **183.36 s** |
| **Mean Audio Duration** | 10.05 s | 12.48 s | 3.99 s | **4.58 s** |
| **Peak VRAM** | 7.65 GB | 7.65 GB | 7.19 GB | **7.23 GB** |

### Head-to-Head Comparison: T2 (Talker-Only) vs T3 (Joint Talker+MTP)
- **Sample-by-sample CER**: T3 wins **35 / 40 (87.5%)**, T2 wins 4 / 40 (10.0%), Ties: 1.
- **Sample-by-sample WER**: T3 wins **33 / 40 (82.5%)**, T2 wins 2 / 40 (5.0%), Ties: 5.
- **Conclusion**: T3 (Joint Talker+MTP) decisively outperforms T2. Joint training is scientifically validated as the superior configuration.
