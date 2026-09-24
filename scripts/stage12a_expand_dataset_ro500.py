import gc
import json
import os
import sys
import time
import torch
import soundfile as sf
import librosa
from datasets import load_dataset
from transformers import MimiModel, AutoFeatureExtractor

sys.stdout.reconfigure(encoding="utf-8")

def prevent_windows_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass

def restore_windows_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass

def expand_to_ro500(target_count=500):
    prevent_windows_sleep()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f"STAGE 12A: Expanding Romanian Speech Dataset to {target_count} Utterances")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")
    
    os.makedirs("data/ro500/wav", exist_ok=True)
    out_codes_file = "reports/mimi_codes_ro500.jsonl"
    
    # 1. Collect already processed clips from ro150
    existing_samples = []
    existing_row_ids = set()
    
    if os.path.exists("reports/mimi_codes_ro150.jsonl"):
        with open("reports/mimi_codes_ro150.jsonl", "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    existing_samples.append(item)
                    if "original_row_id" in item:
                        existing_row_ids.add(item["original_row_id"])
                        
    print(f"Found {len(existing_samples)} existing encoded Romanian clips.")
    
    if len(existing_samples) >= target_count:
        print(f"Already have {len(existing_samples)} clips! Writing to {out_codes_file}...")
        with open(out_codes_file, "w", encoding="utf-8") as f:
            for s in existing_samples[:target_count]:
                f.write(json.dumps(s) + "\n")
        restore_windows_sleep()
        return len(existing_samples[:target_count])
        
    needed = target_count - len(existing_samples)
    print(f"Streaming and encoding {needed} additional Romanian utterances from Hugging Face...")
    
    # 2. Load Mimi Model on GPU for fast batched/streaming RVQ encoding
    print("Loading Kyutai Mimi codec for tokenization...")
    mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(device=device, dtype=torch.float32)
    mimi_model.eval()
    
    import datasets
    ds = load_dataset("eduardem/romanian-tts-single-speaker", split="train", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))
    
    newly_collected = []
    start_time = time.time()
    
    for idx, item in enumerate(ds):
        if len(newly_collected) >= needed:
            break
            
        text = item["text"].strip()
        audio_bytes = item["audio"]["bytes"]
        row_id = idx
        
        if row_id in existing_row_ids or len(text) < 15 or len(text) > 200:
            continue
            
        sample_num = len(existing_samples) + len(newly_collected) + 1
        sample_id = f"ro_sample_{sample_num:04d}"
        wav_path = f"data/ro500/wav/{sample_id}.wav"
        
        # Save audio
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
            
        # Verify and resample to 24 kHz mono if needed
        audio_np, sr = sf.read(wav_path)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 24000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
            sr = 24000
            sf.write(wav_path, audio_np, 24000)
            
        dur = len(audio_np) / 24000
        if dur < 1.5 or dur > 12.0:
            os.remove(wav_path)
            continue
            
        wav_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(1).to(device)
        
        with torch.no_grad():
            codes = mimi_model.encode(wav_tensor).audio_codes # [1, 32, T]
            
        codes_16 = codes[0, :16, :].cpu().tolist()
        
        record = {
            "sample_id": sample_id,
            "original_row_id": row_id,
            "transcript": text,
            "duration": round(dur, 2),
            "wav_path": wav_path,
            "codes_16": codes_16
        }
        newly_collected.append(record)
        
        if len(newly_collected) % 25 == 0 or len(newly_collected) == needed:
            elapsed = time.time() - start_time
            rate = len(newly_collected) / elapsed
            print(f"  Processed {len(newly_collected)}/{needed} new clips ({rate:.1f} clips/s, total: {len(existing_samples) + len(newly_collected)}/{target_count})...")
            
    # Combine existing + new
    all_500 = existing_samples + newly_collected
    
    with open(out_codes_file, "w", encoding="utf-8") as f:
        for s in all_500:
            f.write(json.dumps(s) + "\n")
            
    total_time = time.time() - start_time
    total_audio_mins = sum(s["duration"] for s in all_500) / 60.0
    print(f"\nStage 12A Complete in {total_time:.1f}s!")
    print(f"Total Clips in Dataset: {len(all_500)}")
    print(f"Total Romanian Audio: {total_audio_mins:.2f} minutes")
    print(f"Saved manifest and codes to: {out_codes_file}")
    
    del mimi_model
    torch.cuda.empty_cache()
    gc.collect()
    restore_windows_sleep()
    return len(all_500)

if __name__ == "__main__":
    expand_to_ro500(500)
