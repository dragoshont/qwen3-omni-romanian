# 4-Bit QLoRA Claim Audit & Technical Verification

**Date**: 2026-09-24  
**Hardware**: NVIDIA GeForce RTX 5080 (15.89 GB VRAM, Blackwell SM120)  
**Evaluator**: Autonomous AI Research Engineer  

---

## 1. Executive Summary: Measured vs. Extrapolated

During previous sessions, several claims were made regarding the feasibility of 4-bit NF4 base quantization for the 3.32-billion parameter Qwen3-Omni Talker backbone. This audit separates **direct physical measurements** from **mathematical extrapolations**.

| Parameter / Claim | Actually Measured? | Measured Value | Extrapolated Value | Status |
|---|---|---|---|---|
| **Talker MoE Parameter Count** | **YES** | 3,324,596,480 params (3.32B) | - | **VERIFIED FACT** |
| **Talker MoE Layer Structure** | **YES** | 20 layers, 128 local experts/layer, top-6 routing | - | **VERIFIED FACT** |
| **Pure BF16 Talker Memory Footprint** | **YES** | 6.20 GB allocated in static eval state | 6.65 GB tensor memory | **VERIFIED FACT** |
| **Pure BF16 Talker OOM at Layer 16** | **YES** | Crashed at 9.96 GB allocated | - | **VERIFIED FACT** (under WDDM chunk allocation) |
| **1-Layer 4-Bit Expert Memory Reduction** | **YES** | 25.1 MB allocated for 128 Linear4bit layers | vs ~560 MB in BF16 | **VERIFIED FACT** |
| **20-Layer 4-Bit MoE Weight Storage** | **NO** | Extrapolated from 1-layer measurement | **0.49 GB** (20 × 25.1 MB) | **MATHEMATICAL EXTRAPOLATION** |
| **Forward & Backward Through Linear4bit** | **YES** | Backward pass succeeded cleanly through `Linear4bit` on CUDA | - | **VERIFIED FACT** |
| **Full Talker 20-Layer Training Step in 4-Bit** | **NO** | Not yet executed end-to-end with PEFT | Extrapolated <8.5 GB VRAM | **PENDING FULL GRAPH VERIFICATION** |

---

## 2. Detailed Technical Breakdown

### A. The Linear4bit MoE Weight Mechanism
* In `run_overnight_research.py` (lines 401–424), 128 linear layers of dimensions `[1024, 384]` were instantiated using `bitsandbytes.nn.Linear4bit(compute_dtype=torch.bfloat16, quant_type='nf4')`.
* Measured memory before allocation: **0.00 GB**.
* Measured memory after 128 experts: **25.1 MB** (0.0245 GB).
* A single synthetic forward pass `out = expert[0](x)` and backward pass `loss.backward()` completed without CUDA errors.

### B. The Extrapolation Boundary
* While the individual layer memory is mathematically accurate (20 × 25.1 MB ≈ 502 MB), the **entire Qwen3-Omni Talker graph contains additional modules**:
  1. Self-Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) = ~0.65 GB.
  2. Input/Output embeddings (`codec_embedding`, `text_projection`, `codec_head`) = ~0.50 GB.
  3. Shared experts (`shared_expert`, `shared_expert_gate`) = ~0.35 GB.
  4. MTP predictor (`code_predictor`) = ~0.29 GB.
* **Realistic Projected 4-Bit Talker Base Footprint**:
  `0.50 GB (MoE) + 0.65 GB (Attn) + 0.50 GB (Embeds) + 0.35 GB (Shared) + 0.29 GB (MTP) ≈ 2.29 GB`.
* Combined with AdamW optimizer states for LoRA adapters (~0.15 GB) and activation caching with gradient checkpointing (~1.5–2.5 GB), the total training footprint is estimated at **4.5–5.5 GB VRAM**, well under the 15.0 GB safety threshold.

---

## 3. Scientific Verdict
* **Is 4-bit QLoRA mathematically feasible on RTX 5080?**: **YES**.
* **Has a full 20-layer Qwen3-Omni Talker forward/backward step with LoRA been run end-to-end?**: **NO, pending Phase T2/T3 execution**.
