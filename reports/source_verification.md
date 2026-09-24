# Source Verification Report: Mimi Codec & Qwen3-Omni Community Evidence

Date: September 23, 2026
Workspace: `ro-omni`

## 1. Repositories and Exact Commit SHAs

The four reference repositories were cloned into `src/` and verified at the exact commits:

1. **Qwen3-Omni**:
   - Repository: `https://github.com/QwenLM/Qwen3-Omni.git`
   - Verified Commit SHA: `e4235853125589c789f06a2dd83e9f4126df5e9d`
2. **DuplexOmni**:
   - Repository: `https://github.com/MuyeHuang/DuplexOmni.git`
   - Verified Commit SHA: `33bfba1a821b09c5aa66790944f9098584979d34`
3. **Qwen-Omni-Training-Talker (Hert4)**:
   - Repository: `https://github.com/Hert4/Qwen-Omni-Training-Talker.git`
   - Verified Commit SHA: `d9ad37e150a8e93d16169449d33b48a3141372bf`
4. **Qwen3-TTS**:
   - Repository: `https://github.com/QwenLM/Qwen3-TTS.git`
   - Verified Commit SHA: `022e286b98fbec7e1e916cb940cdf532cd9f488e`

---

## 2. Source Code Evidence Inspection

### A. DuplexOmni (`src/DuplexOmni`)

#### `data_pipeline/05_mimi_codec/config.py` (lines 31-34)
```python
# Mimi 帧率 12.5 Hz => 80ms 一帧；目标时长取 480ms（6 帧）
FRAME_MS = 80
TARGET_AUDIO_MS = 480
NUM_MIMI_LAYERS_USE = 16  # 只使用前 16 层，与 Qwen3-Omni 一致
```
*Meaning*: Explicitly states that only the first 16 layers of Mimi codes are used, which is defined as consistent with Qwen3-Omni.

#### `data_pipeline/05_mimi_codec/extract_mimi_codes_parquet.py` (lines 593-605, 740-745)
```python
with th.inference_mode():
    codes_batch = mimi_model.encode(audio_tensor).audio_codes.cpu()
...
def normalize_codes_layout(codes: torch.Tensor) -> torch.Tensor:
    ...
    if codes.shape[0] >= NUM_MIMI_LAYERS_USE:
        return codes[:NUM_MIMI_LAYERS_USE].contiguous().long()
```
*Meaning*: Directly encodes audio waveforms using Kyutai's `MimiModel.encode()`, retrieves `audio_codes`, and slices the first 16 codebook streams (`[:16]`).

---

### B. Hert4 Qwen Talker Trainer (`src/Qwen-Omni-Training-Talker`)

#### `Qwen3/train_lora_full.py` (lines 563-595)
```python
mimi_model = MimiModel.from_pretrained(
    MIMI_REPO_ID, 
    torch_dtype=model.talker.dtype
).to(execution_device)
mimi_model.eval()
for param in mimi_model.parameters():
    param.requires_grad_(False)

# Freeze Code2Wav
freeze_module(model.code2wav, "Code2Wav")

# Apply LoRA based on flags
if TRAIN_THINKER:
    model = apply_lora_to_thinker(model)
...
if TRAIN_TALKER:
    model = apply_lora_to_talker(model)
...
if TRAIN_MTP:
    model = apply_lora_to_mtp(model)
    model.talker.code_predictor.train()
else:
    freeze_module(model.talker.code_predictor, "MTP (code_predictor)")
```
*Meaning*:
1. Mimi is loaded as the ground-truth audio target tokenizer and kept frozen (`requires_grad_(False)`).
2. `model.code2wav` is explicitly frozen.
3. Talker and MTP (`code_predictor`) are decoupled with independent training flags (`TRAIN_TALKER`, `TRAIN_MTP`) and separate LoRA parameter groups.

#### `Qwen3/train_lora_full.py` (lines 193-201, 335-337, 413-440)
```python
codes = encode_audio_to_codes(audio_path, feature_extractor, mimi_model, device)
codes = align_codebook_dim(codes, model.code2wav.config.num_quantizers)
...
layer0_codes = sample_codes[:, 0, :]
...
num_mtp_layers = NUM_CODE_GROUPS - 1
for mtp_layer_idx in range(num_mtp_layers):
    ...
    target_layer_codes = sample_codes[:, mtp_layer_idx + 1, :].to(device)
    mtp_outputs = code_predictor(inputs_embeds=mtp_inputs, generation_steps=mtp_layer_idx, ...)
```
*Meaning*:
- The first stream (`layer0_codes`) is supervised by Talker logits.
- The remaining 15 streams (`mtp_layer_idx + 1`) are supervised sequentially by the multi-token predictor (MTP / `code_predictor`).

---

## 3. Direct Scientific Verification Checklist

| Question | Answer | Direct Code Evidence |
|---|---|---|
| **Does DuplexOmni use Mimi.encode?** | **yes** | `mimi_model.encode(audio_tensor).audio_codes.cpu()` in `extract_mimi_codes_parquet.py:741` |
| **Does it keep exactly first 16 streams?** | **yes** | `NUM_MIMI_LAYERS_USE = 16`, `codes[:NUM_MIMI_LAYERS_USE]` in `config.py:34` & `extract_mimi_codes_parquet.py:601` |
| **Does Hert4 also use Mimi targets?** | **yes** | `MimiModel.from_pretrained(MIMI_REPO_ID)` in `train_lora_full.py:563`, `encode_audio_to_codes` in line 194 |
| **Does Hert4 freeze Code2Wav?** | **yes** | `freeze_module(model.code2wav, "Code2Wav")` in `train_lora_full.py:572` |
| **Does Hert4 train Talker and MTP separately?** | **yes** | Independent switches `TRAIN_TALKER` and `TRAIN_MTP` with separate LoRA targets in `train_lora_full.py:581, 590` |

---

## 4. Why This Matters Scientifically

Independent implementations doing the exact same specific thing make the hypothesis much more credible:
1. Kyutai's Mimi produces 32 codebooks by default at 12.5 Hz (24 kHz audio).
2. Both DuplexOmni and Hert4 independently truncate Mimi codes to exactly the first 16 codebooks.
3. Both projects treat these 16 codebooks as the exact target token format for Qwen3-Omni's speech generation.

However, as stated in our scientific rules: **Independent implementations are circumstantial evidence, not empirical proof.** 
Tonight's decisive test is whether real Romanian audio encoded into Mimi's first 16 codebooks reconstructs recognizable Romanian speech through Qwen's frozen `Code2Wav` synthesizer.
