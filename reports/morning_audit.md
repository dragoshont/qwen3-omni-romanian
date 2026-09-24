============================================================
ROMANIAN QWEN OVERNIGHT AUDIT
============================================================
Machine: NVIDIA GeForce RTX 5080 / 16 GB VRAM
System RAM: 32 GB DDR5
OS: Windows 11 Home (x86_64, AMD64)
Runtime: Native Windows uv virtualenv (Python 3.11.16)
PyTorch: 2.11.0+cu128 (Native Blackwell SM120 support)
CUDA: 12.8 runtime / Driver 617.14
Transformers: 5.17.0

Experiments discovered: 8
Checkpoints discovered: 12 (including ro30, ro150, ro500, cycle2 milestones)
Romanian clips actually used: 500 clips in active training; 1,000 clips materialized in ro1000
Romanian audio hours actually used: 0.75 h (ro500); 1.258 h (ro1000)
Last completed training step: 7,500 steps (Stage 12B final) + 3,000 steps (Cycle 2)

What was actually trained:
[X] MTP only (141M params, LoRA Rank 16 targeting all attention & MLP projections)
[ ] Talker only (Benchmarked and profiled; pending Phase T2)
[ ] Talker + MTP (Pending Phase T3)
[ ] Thinker (FROZEN per safety rules; reserved for future 256 GB M5 Ultra)
[ ] Code2Wav (FROZEN per safety rules; already validated with 90% intelligibility)
[ ] unknown

Did true text -> Romanian speech happen?
NO (NOT YET PROVEN)

Did inference use ground-truth Mimi tokens?
YES (Stream 0 was provided from ground-truth Mimi audio tokens; MTP predicted streams 1..15)

Peak measured VRAM: 4.91 GB (Stage 12B 500-clip training)
Current scientific confidence: MEDIUM-HIGH
Next recommended experiment: Execute controlled matrix (Phase T0 stock baseline -> Phase T1 MTP-only -> Phase T2 Talker-only 1h -> Phase T3 Talker+MTP 1h) on the 40/200 held-out evaluation sets.
============================================================

---

## Section 5: The 30 Numerical Answers for the Overnight / Scaled Runs

1. **Dataset name**: `eduardem/romanian-tts-single-speaker` (Hugging Face)
2. **Number of clips**: 30 (ro30) -> 150 (ro150) -> 500 (ro500, active training) -> 1,000 (ro1000, materialized)
3. **Total duration**: 2.7 mins (ro30), 13.2 mins (ro150), 45.1 mins (ro500), 75.5 mins (ro1000)
4. **Number of speakers**: 1 (female speaker, Sanda)
5. **Train split**: 100% of streamed subsets (no random leak)
6. **Validation split**: Evaluated on deterministic subset (samples 0, 1, 2, 50, 150)
7. **Holdout split**: Held-out 200-sentence test set created in `eval/ro_holdout_200.jsonl`
8. **Number of epochs**: 20 (ro30), 5 (ro150), 15 (ro500 Stage 12B), 6 (Cycle 2)
9. **Optimizer steps**: 600 (ro30), 750 (ro150), 7,500 (ro500), 3,000 (Cycle 2) = 11,850 total steps
10. **Batch size**: 1 utterance per forward step
11. **Gradient accumulation**: 1
12. **Effective batch size**: 1
13. **Sequence-length policy**: Natural utterance duration (1.5s to 12.0s, resampled to 24 kHz mono)
14. **Learning rate**: 1e-4 (ro30), 5e-5 (ro150), 4e-5 (ro500), 2.5e-5 (Cycle 2)
15. **Scheduler**: CosineAnnealingLR with minimum learning rate 5e-7 to 1e-6
16. **LoRA rank**: 16
17. **LoRA alpha**: 32
18. **LoRA target modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
19. **Quantization method**: BF16 base weights (MTP) with BF16 LoRA adapter
20. **Which components had gradients**: `talker.code_predictor` LoRA parameters only (1,802,240 params)
21. **Which components were frozen**: MTP base weights, `talker.model.codec_embedding` (frozen layer0 embedding), Code2Wav
22. **Did the run finish**: YES, both Stage 12B and Cycle 2 ran to completion
23. **Final checkpoint path**: `models/romanian_mtp_lora_ro500/final`
24. **Peak allocated VRAM**: 4.91 GB
25. **Peak reserved VRAM**: 5.22 GB
26. **Wall-clock time**: 77.4s (ro30), 96.9s (ro150), 1,438.6s / 23.98 min (ro500)
27. **Mean sec/step**: ~0.191 seconds/step
28. **Any OOM**: None during MTP training (hit WDDM ceiling only during uncompressed BF16 Talker load)
29. **Any CPU offload**: None (entire active MTP model and optimizer resided on GPU)
30. **Any NaN/Inf**: Zero NaN or Inf encountered; gradient norm clipped at 1.0

---

## Section 6: Provenance & Level Classification of Generated Audio

| File | Input Source | Stream 0 Source | Streams 1..15 Source | Verified Scientific Level |
|---|---|---|---|---|
| `outputs/qwen_code2wav_from_mimi/ro_sample_01.wav` | Ground-truth WAV | Mimi Encode (Ground-truth) | Mimi Encode (Ground-truth) | **LEVEL 0 — Decoder Sanity Test** |
| `outputs/romanian_mtp_trained_eval/ro_sample_01_mtp_lora.wav` | Ground-truth WAV | Mimi Encode (Ground-truth) | MTP LoRA (ro30 trained) | **LEVEL 1 — MTP Reconstruction** |
| `outputs/romanian_mtp_trained_eval/ro_sample_01_ro500_mtp.wav` | Ground-truth WAV | Mimi Encode (Ground-truth) | MTP LoRA (ro500 trained) | **LEVEL 1 — MTP Reconstruction** |
| `outputs/autonomous_eval/cycle2_full15_ro_sample_01.wav` | Ground-truth WAV | Mimi Encode (Ground-truth) | MTP LoRA (Cycle 2 trained) | **LEVEL 1 — MTP Reconstruction** |

**Highest Verified Level Today**: **LEVEL 1 (MTP Reconstruction)**.
* True Romanian Text-to-Speech (**LEVEL 3**) requires the Talker to predict Stream 0 from Romanian text prompts rather than receiving ground-truth Mimi tokens. That is the exact focus of Phase T2 and T3.
