import gc
import json
import os
import sys
import time
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
import safetensors.torch
import jiwer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
    Qwen3OmniMoeCode2Wav,
    pipeline
)

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
        print("[Power] Windows Sleep Prevention: RESTORED to default.")
    except Exception:
        pass

def revert_power_settings():
    try:
        subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "15"], check=False)
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"], check=False)
        print("[Power] Power settings reverted to standard gaming desktop defaults.")
    except Exception as e:
        print(f"[Power] Revert notice: {e}")

def run_ro1000_deep_training(num_epochs=8):
    prevent_windows_sleep()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f"STAGE 13: Scaled Deep Training on 1,000 Romanian Utterances (ro1000)")
    print(f"Epochs: {num_epochs} | Device: {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
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
    
    # Initialize from the best ro500 checkpoint
    init_ckpt = "models/romanian_mtp_lora_ro500/final"
    print(f"Transfer Learning: Initializing from prior best checkpoint ({init_ckpt})...")
    peft_model = PeftModel.from_pretrained(cp_model, init_ckpt, is_trainable=True)
    peft_model.print_trainable_parameters()
    
    # Load 1,000 Romanian clips
    dataset = []
    with open("reports/mimi_codes_ro1000.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                dataset.append({
                    "sample_id": d["sample_id"],
                    "codes": torch.tensor(d["codes_16"], dtype=torch.long)
                })
                
    print(f"Loaded {len(dataset)} Romanian speech utterances from ro1000!")
    total_steps = num_epochs * len(dataset)
    
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=3e-5, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-7)
    
    loss_history = []
    global_step = 0
    start_time = time.time()
    peft_model.train()
    
    depth_indices = [0, 1, 2, 4, 6, 8, 10, 12, 14]
    out_ckpt_dir = "models/romanian_mtp_lora_ro1000"
    os.makedirs(out_ckpt_dir, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        t_epoch_start = time.time()
        for sample in dataset:
            global_step += 1
            sample_codes = sample["codes"].to(device)
            layer0_codes = sample_codes[0]
            layer0_embeds = layer0_embedding(layer0_codes)
            predictor_embeds = peft_model.model.model.codec_embedding
            
            step_loss = 0.0
            hidden_flat = layer0_embeds.unsqueeze(1)
            layer0_flat = layer0_embeds.unsqueeze(1)
            
            for mtp_layer_idx in depth_indices:
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
                
            loss = step_loss / len(depth_indices)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            if global_step % 100 == 0 or global_step == 1:
                cur_vram = torch.cuda.memory_allocated() / (1024**3)
                print(f"  [Epoch {epoch:2d}/{num_epochs:2d}] Step {global_step:4d}/{total_steps:4d} | "
                      f"Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e} | VRAM: {cur_vram:.2f} GB")
                loss_history.append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "lr": round(lr_scheduler.get_last_lr()[0], 7)
                })
                
        avg_loss = epoch_loss / len(dataset)
        epoch_time = time.time() - t_epoch_start
        print(f"==> Epoch {epoch}/{num_epochs} Complete! Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        if epoch % 4 == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(out_ckpt_dir, f"checkpoint-epoch-{epoch}")
            peft_model.save_pretrained(ckpt_path)
            print(f"--> Saved Checkpoint: {ckpt_path}")
            
    final_path = os.path.join(out_ckpt_dir, "final")
    peft_model.save_pretrained(final_path)
    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    print("\n" + "="*50)
    print("STAGE 13 TRAINING COMPLETE!")
    print(f"  Total Duration: {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"  Initial Loss: {loss_history[0]['loss']:.4f}")
    print(f"  Final Loss: {loss_history[-1]['loss']:.4f}")
    print(f"  Peak VRAM: {peak_vram:.2f} GB / 15.89 GB")
    print(f"  Saved Final Checkpoint: {final_path}")
    print("="*50)
    
    with open("reports/mtp_ro1000_training_loss.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)
        
    del peft_model, cp_model, layer0_embedding
    torch.cuda.empty_cache()
    gc.collect()
    return final_path

# =====================================================================
# SYNTHESIS & IN-MEMORY WHISPER EVALUATION
# =====================================================================
def evaluate_ro1000(ckpt_path):
    print("\n" + "="*70)
    print(f"Evaluating ro1000 Speech Synthesis & Whisper Transcription")
    print("="*70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    
    # 1. Code2Wav
    code2wav = Qwen3OmniMoeCode2Wav(full_config.code2wav_config).to(device=device, dtype=torch.bfloat16)
    c2w_sd = safetensors.torch.load_file(os.path.join(config_dir, "model-00015-of-00015.safetensors"))
    c2w_weights = {k[len("code2wav."):]: v for k, v in c2w_sd.items() if k.startswith("code2wav.")}
    code2wav.load_state_dict(c2w_weights, strict=True)
    code2wav.eval()
    del c2w_sd, c2w_weights
    
    # 2. MTP ro1000
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
    
    test_indices = [0, 1, 2, 50, 150]
    samples = []
    with open("reports/mimi_codes_ro1000.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in test_indices:
                samples.append(json.loads(line))
                
    out_dir = "outputs/romanian_mtp_trained_eval"
    os.makedirs(out_dir, exist_ok=True)
    generated_records = []
    
    for s in samples:
        sample_id = s["sample_id"]
        text = s["transcript"]
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
            
        out_wav = os.path.join(out_dir, f"{sample_id}_ro1000_mtp.wav")
        sf.write(out_wav, wav_np, 24000)
        dur = len(wav_np) / 24000
        print(f"  ✓ Synthesized: {out_wav} ({dur:.2f}s)")
        generated_records.append({
            "sample_id": sample_id,
            "transcript": text,
            "wav_path": out_wav
        })
        
    del code2wav, cp_model, peft_model, layer0_embedding
    torch.cuda.empty_cache()
    gc.collect()
    
    # In-memory Whisper transcription (no ffmpeg dependency!)
    print("\nRunning Whisper large-v3-turbo in-memory evaluation...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cuda", dtype=torch.float16)
    
    total_wer = 0.0
    total_cer = 0.0
    for item in generated_records:
        audio_np, sr = sf.read(item["wav_path"])
        if audio_np.ndim > 1: audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        res = asr({"raw": audio_np, "sampling_rate": 16000}, generate_kwargs={"language": "ro", "task": "transcribe"})
        hyp = res["text"].strip()
        ref = item["transcript"]
        w = jiwer.wer(ref, hyp)
        c = jiwer.cer(ref, hyp)
        item["hyp"] = hyp
        item["wer"] = round(w, 4)
        item["cer"] = round(c, 4)
        total_wer += w
        total_cer += c
        print(f"\n[{item['sample_id']}]:")
        print(f"  Ref: \"{ref}\"")
        print(f"  Hyp: \"{hyp}\"")
        print(f"  WER: {w*100:.1f}% | CER: {c*100:.1f}%")
        
    del asr
    torch.cuda.empty_cache()
    gc.collect()
    
    avg_wer = total_wer / len(generated_records)
    avg_cer = total_cer / len(generated_records)
    
    print("\n" + "="*50)
    print("RO1000 BENCHMARK SUMMARY:")
    print(f"  Average WER: {avg_wer*100:.2f}%")
    print(f"  Average CER: {avg_cer*100:.2f}%")
    print("="*50)
    
    with open("reports/ro1000_whisper_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"avg_wer": round(avg_wer, 4), "avg_cer": round(avg_cer, 4), "samples": generated_records}, f, indent=2)
        
    return avg_wer, avg_cer

def main():
    try:
        final_ckpt = run_ro1000_deep_training(num_epochs=8)
        wer, cer = evaluate_ro1000(final_ckpt)
        print(f"\nFinal Autonomous Run Complete! Best ro1000 WER: {wer*100:.2f}%, CER: {cer*100:.2f}%")
    finally:
        revert_power_settings()
        restore_windows_sleep()

if __name__ == "__main__":
    main()
