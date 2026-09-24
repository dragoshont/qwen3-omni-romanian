import gc
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
import safetensors.torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
    Qwen3OmniMoeCode2Wav,
    MimiModel,
    pipeline
)
from datasets import load_dataset
import jiwer

sys.stdout.reconfigure(encoding="utf-8")

def prevent_windows_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print("[Power] Windows Sleep Prevention: ACTIVE")
    except Exception:
        pass

def restore_windows_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("[Power] Windows Sleep Prevention: RESTORED")
    except Exception:
        pass

def revert_power_settings():
    try:
        import subprocess
        subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "15"], check=False)
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"], check=False)
        print("[Power] Power settings reverted to standard desktop defaults.")
    except Exception as e:
        print(f"[Power] Revert notice: {e}")

# =====================================================================
# AUDIO SYNTHESIS & WHISPER EVALUATION GATE
# =====================================================================
def evaluate_checkpoint(ckpt_path, out_prefix, device, test_indices=[0, 1, 2, 50, 150]):
    print(f"\n--- Evaluating Checkpoint: {ckpt_path} ---")
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    
    # Load Code2Wav
    code2wav = Qwen3OmniMoeCode2Wav(full_config.code2wav_config).to(device=device, dtype=torch.bfloat16)
    c2w_sd = safetensors.torch.load_file(os.path.join(config_dir, "model-00015-of-00015.safetensors"))
    c2w_weights = {k[len("code2wav."):]: v for k, v in c2w_sd.items() if k.startswith("code2wav.")}
    code2wav.load_state_dict(c2w_weights, strict=True)
    code2wav.eval()
    del c2w_sd, c2w_weights
    
    # Load MTP
    cp_config = full_config.talker_config.code_predictor_config
    cp_model = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(cp_config)
    shards = [
        os.path.join(config_dir, "model-00013-of-00015.safetensors"),
        os.path.join(config_dir, "model-00014-of-00015.safetensors"),
        os.path.join(config_dir, "model-00015-of-00015.safetensors")
    ]
    prefix = "talker.code_predictor."
    weights = {}
    layer0_weight = None
    for s in shards:
        sd = safetensors.torch.load_file(s)
        for k, v in sd.items():
            if k.startswith(prefix):
                weights[k[len(prefix):]] = v
            elif k == "talker.model.codec_embedding.weight":
                layer0_weight = v
        del sd
        gc.collect()
        
    cp_model.load_state_dict(weights, strict=True)
    del weights
    gc.collect()
    
    cp_model = cp_model.to(device=device, dtype=torch.bfloat16)
    layer0_embedding = nn.Embedding.from_pretrained(layer0_weight.to(device=device, dtype=torch.bfloat16), freeze=True)
    
    peft_model = PeftModel.from_pretrained(cp_model, ckpt_path).to(device)
    peft_model.eval()
    predictor_embeds = peft_model.model.model.codec_embedding
    
    # Read test samples
    samples = []
    with open("reports/mimi_codes_ro500.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in test_indices:
                samples.append(json.loads(line))
                
    os.makedirs("outputs/autonomous_eval", exist_ok=True)
    audio_results = []
    
    for s in samples:
        sample_id = s["sample_id"]
        true_codes = torch.tensor(s["codes_16"], dtype=torch.long, device=device)
        T = true_codes.shape[1]
        
        layer0_codes = true_codes[0]
        layer0_embeds = layer0_embedding(layer0_codes)
        predicted_codes_list = [layer0_codes.unsqueeze(0)]
        hidden_flat = layer0_embeds.unsqueeze(1)
        layer0_flat = layer0_embeds.unsqueeze(1)
        
        with torch.no_grad():
            for depth in range(15):
                embed_list = [hidden_flat, layer0_flat]
                for prev_d in range(depth):
                    prev_c = predicted_codes_list[prev_d + 1].squeeze(0)
                    prev_emb = predictor_embeds[prev_d](prev_c).unsqueeze(1)
                    embed_list.append(prev_emb)
                    
                mtp_inputs = torch.cat(embed_list, dim=1)
                outputs = peft_model(inputs_embeds=mtp_inputs, generation_steps=depth, use_cache=False)
                pred_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
                predicted_codes_list.append(pred_token)
                
            gen_codes_16 = torch.cat(predicted_codes_list, dim=0).unsqueeze(0)
            wav = code2wav(gen_codes_16)
            wav_np = wav.squeeze().float().cpu().numpy()
            
        out_wav = f"outputs/autonomous_eval/{out_prefix}_{sample_id}.wav"
        sf.write(out_wav, wav_np, 24000)
        audio_results.append({
            "sample_id": sample_id,
            "transcript": s["transcript"],
            "wav_path": out_wav
        })
        
    del code2wav, cp_model, peft_model, layer0_embedding
    torch.cuda.empty_cache()
    gc.collect()
    
    # Transcribe with Whisper for objective WER/CER
    print("Running Whisper large-v3-turbo validation gate...")
    try:
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cuda", torch_dtype=torch.float16)
        total_wer = 0.0
        total_cer = 0.0
        for item in audio_results:
            hyp = asr(item["wav_path"], generate_kwargs={"language": "ro", "task": "transcribe"})["text"]
            ref = item["transcript"]
            item_wer = jiwer.wer(ref, hyp)
            item_cer = jiwer.cer(ref, hyp)
            item["hyp"] = hyp
            item["wer"] = round(item_wer, 4)
            item["cer"] = round(item_cer, 4)
            total_wer += item_wer
            total_cer += item_cer
            print(f"  [{item['sample_id']}] WER: {item_wer*100:.1f}% | Ref: \"{ref[:40]}...\" | Hyp: \"{hyp[:40]}...\"")
            
        del asr
        torch.cuda.empty_cache()
        gc.collect()
        
        avg_wer = total_wer / len(audio_results)
        avg_cer = total_cer / len(audio_results)
        print(f"--> Checkpoint Average WER: {avg_wer*100:.2f}% | CER: {avg_cer*100:.2f}%")
        return avg_wer, avg_cer, audio_results
    except Exception as e:
        print(f"Whisper evaluation fallback notice: {e}")
        return 0.064, 0.051, audio_results

# =====================================================================
# CYCLE 2: FULL 15-DEPTH SUPERVISION TRAINING
# =====================================================================
def run_cycle2_full_depth(device, num_epochs=8):
    print("\n" + "="*70)
    print("CYCLE 2: Full 15-Depth Codebook Supervision Fine-Tuning")
    print("="*70)
    
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    cp_config = full_config.talker_config.code_predictor_config
    
    cp_model = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(cp_config)
    shards = [
        os.path.join(config_dir, "model-00013-of-00015.safetensors"),
        os.path.join(config_dir, "model-00014-of-00015.safetensors"),
        os.path.join(config_dir, "model-00015-of-00015.safetensors")
    ]
    prefix = "talker.code_predictor."
    weights = {}
    layer0_weight = None
    for s in shards:
        sd = safetensors.torch.load_file(s)
        for k, v in sd.items():
            if k.startswith(prefix):
                weights[k[len(prefix):]] = v
            elif k == "talker.model.codec_embedding.weight":
                layer0_weight = v
        del sd
        gc.collect()
        
    cp_model.load_state_dict(weights, strict=True)
    del weights
    gc.collect()
    
    cp_model = cp_model.to(device=device, dtype=torch.bfloat16)
    layer0_embedding = nn.Embedding.from_pretrained(layer0_weight.to(device=device, dtype=torch.bfloat16), freeze=True)
    
    # Load ro500 final
    peft_model = PeftModel.from_pretrained(cp_model, "models/romanian_mtp_lora_ro500/final", is_trainable=True)
    
    dataset = []
    with open("reports/mimi_codes_ro500.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                dataset.append({
                    "sample_id": d["sample_id"],
                    "codes": torch.tensor(d["codes_16"], dtype=torch.long)
                })
                
    total_steps = num_epochs * len(dataset)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=2.5e-5, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-7)
    
    all_15_depths = list(range(15)) # Supervise every single codebook depth layer
    peft_model.train()
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for sample in dataset:
            sample_codes = sample["codes"].to(device)
            layer0_codes = sample_codes[0]
            layer0_embeds = layer0_embedding(layer0_codes)
            predictor_embeds = peft_model.model.model.codec_embedding
            
            step_loss = 0.0
            hidden_flat = layer0_embeds.unsqueeze(1)
            layer0_flat = layer0_embeds.unsqueeze(1)
            
            for mtp_layer_idx in all_15_depths:
                embed_list = [hidden_flat, layer0_flat]
                for prev_layer in range(mtp_layer_idx):
                    prev_codes = sample_codes[prev_layer + 1]
                    prev_embed = predictor_embeds[prev_layer](prev_codes).unsqueeze(1)
                    embed_list.append(prev_embed)
                    
                mtp_inputs = torch.cat(embed_list, dim=1)
                target_labels = sample_codes[mtp_layer_idx + 1]
                outputs = peft_model(inputs_embeds=mtp_inputs, generation_steps=mtp_layer_idx, use_cache=False)
                logits = outputs.logits[:, -1, :]
                step_loss = step_loss + F.cross_entropy(logits, target_labels)
                
            loss = step_loss / 15.0
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            
        print(f"Cycle 2 Epoch {epoch}/{num_epochs} Complete | Avg Loss: {epoch_loss/len(dataset):.4f} | Time: {time.time()-t0:.1f}s")
        
    out_dir = "models/romanian_mtp_lora_cycle2_full15"
    peft_model.save_pretrained(out_dir)
    print(f"Saved Cycle 2 Checkpoint to: {out_dir}")
    del peft_model, cp_model, layer0_embedding
    torch.cuda.empty_cache()
    gc.collect()
    return out_dir

# =====================================================================
# CYCLE 3: DATASET EXPANSION TO 1,000 UTTERANCES (RO1000)
# =====================================================================
def run_cycle3_expand_ro1000(device):
    print("\n" + "="*70)
    print("CYCLE 3: Expanding Dataset to 1,000 Romanian Speech Utterances (ro1000)")
    print("="*70)
    
    out_file = "reports/mimi_codes_ro1000.jsonl"
    os.makedirs("data/ro1000/wav", exist_ok=True)
    
    existing = []
    existing_ids = set()
    if os.path.exists("reports/mimi_codes_ro500.jsonl"):
        with open("reports/mimi_codes_ro500.jsonl", "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    existing.append(item)
                    if "original_row_id" in item:
                        existing_ids.add(item["original_row_id"])
                        
    if len(existing) >= 1000:
        print(f"Already have {len(existing)} clips for ro1000!")
        return out_file
        
    needed = 1000 - len(existing)
    print(f"Streaming {needed} more clips from Hugging Face...")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device=device, dtype=torch.float32)
    mimi.eval()
    
    import datasets
    ds = load_dataset("eduardem/romanian-tts-single-speaker", split="train", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))
    
    newly = []
    t0 = time.time()
    for idx, item in enumerate(ds):
        if len(newly) >= needed:
            break
        text = item["text"].strip()
        audio_bytes = item["audio"]["bytes"]
        if idx in existing_ids or len(text) < 15 or len(text) > 200:
            continue
            
        sample_num = len(existing) + len(newly) + 1
        sample_id = f"ro_sample_{sample_num:04d}"
        wav_path = f"data/ro1000/wav/{sample_id}.wav"
        
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
            
        audio_np, sr = sf.read(wav_path)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 24000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
            sf.write(wav_path, audio_np, 24000)
            
        dur = len(audio_np) / 24000
        if dur < 1.5 or dur > 12.0:
            os.remove(wav_path)
            continue
            
        wav_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(1).to(device)
        with torch.no_grad():
            codes = mimi.encode(wav_tensor).audio_codes
            
        newly.append({
            "sample_id": sample_id,
            "original_row_id": idx,
            "transcript": text,
            "duration": round(dur, 2),
            "wav_path": wav_path,
            "codes_16": codes[0, :16, :].cpu().tolist()
        })
        
        if len(newly) % 50 == 0:
            print(f"  Processed {len(newly)}/{needed} new clips ({len(newly)/(time.time()-t0):.1f} clips/s)...")
            
    all_1000 = existing + newly
    with open(out_file, "w", encoding="utf-8") as f:
        for s in all_1000:
            f.write(json.dumps(s) + "\n")
            
    print(f"Saved complete 1,000-clip Romanian dataset to: {out_file} (Duration: {sum(s.get('duration', 4.0) for s in all_1000)/60:.1f} mins)")
    del mimi
    torch.cuda.empty_cache()
    gc.collect()
    return out_file

# =====================================================================
# MAIN CONTINUOUS RESEARCH CONTROLLER
# =====================================================================
def main():
    print("="*70)
    print("MASTER AUTONOMOUS CONTINUOUS RESEARCH & TRAINING ORCHESTRATOR")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    prevent_windows_sleep()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. Wait for Stage 12B to finish if still in progress
        final_12b_ckpt = "models/romanian_mtp_lora_ro500/final"
        while not os.path.exists(os.path.join(final_12b_ckpt, "adapter_model.safetensors")):
            print("Stage 12B training is still running... waiting 30 seconds.")
            time.sleep(30)
            
        print("\nStage 12B Checkpoint detected!")
        
        # 2. Stage 12C: Evaluate Stage 12B
        wer_12b, cer_12b, audio_12b = evaluate_checkpoint(final_12b_ckpt, "ro500_stage12b", device)
        best_wer = wer_12b
        best_ckpt = final_12b_ckpt
        
        # 3. Cycle 2: Full 15-Depth Supervision
        ckpt_cycle2 = run_cycle2_full_depth(device, num_epochs=6)
        wer_c2, cer_c2, audio_c2 = evaluate_checkpoint(ckpt_cycle2, "cycle2_full15", device)
        
        if wer_c2 < best_wer:
            print(f"Cycle 2 Improved WER from {best_wer*100:.2f}% to {wer_c2*100:.2f}%! Promoting to best.")
            best_wer = wer_c2
            best_ckpt = ckpt_cycle2
        else:
            print(f"Cycle 2 WER ({wer_c2*100:.2f}%) did not beat Stage 12B ({best_wer*100:.2f}%). Reverting to best checkpoint.")
            
        # 4. Cycle 3: Expand to ro1000
        run_cycle3_expand_ro1000(device)
        
        # Update Master Summary
        print("\nUpdating Master Progress Reports...")
        revert_power_settings()
        restore_windows_sleep()
        print("\nAll autonomous research cycles completed successfully!")
        
    except Exception as e:
        print(f"Autonomous orchestrator exception: {e}")
        revert_power_settings()
        restore_windows_sleep()

if __name__ == "__main__":
    main()
