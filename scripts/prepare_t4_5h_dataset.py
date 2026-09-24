import gc
import json
import os
import sys
import time
import re
import random
import unicodedata
import torch
import soundfile as sf
import librosa
from datasets import load_dataset
from transformers import MimiModel

sys.stdout.reconfigure(encoding="utf-8")

def normalize_text(text):
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t

def compute_phonetic_richness(text):
    """
    Scores Romanian phonetic richness:
    - Diacritics: ă, â, î, ș, ț
    - Affricates: ce, ci, ge, gi, che, chi, ghe, ghi
    - Consonant clusters: str, spr, zdr, mpl, nct, mpt, stl, tr, pr, br, cr, gr
    - Diphthongs: ea, oa, ia, ie, ua, uă
    """
    t = text.lower()
    score = 0
    # Diacritics
    score += len(re.findall(r'[ăâîșț]', t)) * 3
    # Affricates
    score += len(re.findall(r'che|chi|ghe|ghi|ce|ci|ge|gi', t)) * 2
    # Clusters
    score += len(re.findall(r'str|spr|zdr|mpl|nct|mpt|stl|[bcdfghjklmnprstvz]{3}', t)) * 2
    # Diphthongs
    score += len(re.findall(r'ea|oa|ia|ie|ua|uă', t)) * 1.5
    return score

def prepare_5h_dataset(target_seconds=18000.0, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70, flush=True)
    print("STAGE T4: Deterministic 5-Hour Romanian Dataset Preparation & Mimi Encoding", flush=True)
    print(f"Target Audio Duration: {target_seconds/3600:.2f} hours ({target_seconds:.0f}s)", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load exclusion list from held-out evaluation sets
    exclusion_file = "eval/ro_holdout_200.jsonl"
    excluded_texts = set()
    if os.path.exists(exclusion_file):
        with open(exclusion_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    for field in ("reference_text", "text"):
                        normalized = normalize_text(item.get(field, ""))
                        if normalized:
                            excluded_texts.add(normalized)
    print(f"Loaded {len(excluded_texts)} normalized held-out evaluation sentences to STRICTLY EXCLUDE.", flush=True)
    
    # 2. Output directories
    wav_dir = "data/ro_5h/wav"
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs("manifests", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    out_manifest = "manifests/t4_5h_manifest.json"
    out_codes_file = "reports/mimi_codes_ro_5h.jsonl"
    
    # 3. Check if already prepared
    if os.path.exists(out_codes_file) and os.path.exists(out_manifest):
        with open(out_manifest, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if meta.get("total_duration_s", 0) >= (target_seconds - 300):
                print(f"Dataset already fully prepared and encoded! ({meta['total_duration_s']/3600:.2f} hours, {meta['sample_count']} clips).", flush=True)
                return out_codes_file
                
    # 4. Load Mimi model
    print("Loading Kyutai Mimi codec on GPU for 16-codebook RVQ tokenization...", flush=True)
    mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(device=device, dtype=torch.float32)
    mimi_model.eval()
    
    # 5. Stream dataset with deterministic filtering
    print("Streaming dataset 'eduardem/romanian-tts-single-speaker' from Hugging Face...", flush=True)
    import datasets
    ds = load_dataset("eduardem/romanian-tts-single-speaker", split="train", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))
    
    selected_samples = []
    total_duration = 0.0
    processed_count = 0
    start_time = time.time()
    
    # Open jsonl writer for immediate persistence
    jsonl_f = open(out_codes_file, "w", encoding="utf-8")
    
    print("\nStarting deterministic selection and RVQ encoding loop...", flush=True)
    for idx, item in enumerate(ds):
        if total_duration >= target_seconds:
            break
            
        processed_count += 1
        text = item["text"].strip()
        norm_t = normalize_text(text)
        
        # Length filter: 20 to 220 chars
        if len(text) < 20 or len(text) > 220:
            continue
            
        # Strict exclusion check against held-out benchmark
        if norm_t in excluded_texts:
            print(f"[EXCLUDED] Match with held-out eval: '{text}'", flush=True)
            continue
            
        # Phonetic richness check: require at least some Romanian phonetic elements
        score = compute_phonetic_richness(text)
        if score < 3: # Skip overly trivial non-phonetic snippets
            continue
            
        audio_bytes = item["audio"]["bytes"]
        sample_id = f"ro_5h_{len(selected_samples) + 1:05d}"
        wav_path = os.path.join(wav_dir, f"{sample_id}.wav")
        
        # Write wav
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
            
        # Verify audio duration and 24 kHz
        try:
            audio_np, sr = sf.read(wav_path)
        except Exception:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            continue
            
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
            
        if sr != 24000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
            sr = 24000
            sf.write(wav_path, audio_np, 24000)
            
        dur = len(audio_np) / 24000.0
        # Duration bounds: 1.8s to 11.5s
        if dur < 1.8 or dur > 11.5:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            continue
            
        # Encode via Mimi
        wav_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(1).to(device)
        with torch.no_grad():
            codes = mimi_model.encode(wav_tensor).audio_codes # [1, 32, T]
            
        codes_16 = codes[0, :16, :].cpu().tolist()
        num_frames = len(codes_16[0])
        
        record = {
            "sample_id": sample_id,
            "dataset_row_id": idx,
            "transcript": text,
            "audio_duration_s": round(dur, 2),
            "num_frames": num_frames,
            "phonetic_score": score,
            "wav_path": wav_path,
            "codes_16": codes_16
        }
        
        jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_f.flush()
        
        selected_samples.append({
            "sample_id": sample_id,
            "transcript": text,
            "audio_duration_s": round(dur, 2),
            "num_frames": num_frames,
            "phonetic_score": score,
            "wav_path": wav_path
        })
        
        total_duration += dur
        
        if len(selected_samples) % 50 == 0:
            elapsed = time.time() - start_time
            rate = len(selected_samples) / elapsed
            hours_so_far = total_duration / 3600.0
            eta_m = (target_seconds - total_duration) / (dur * rate) / 60.0 if rate > 0 else 0
            print(f"[{len(selected_samples):04d}] Selected: {hours_so_far:.2f}/5.00h ({total_duration:.1f}s) | Speed: {rate:.1f} clips/s | ETA: {eta_m:.1f}m", flush=True)
            
    jsonl_f.close()
    
    elapsed_total = time.time() - start_time
    total_hours = total_duration / 3600.0
    print("\n" + "=" * 70, flush=True)
    print(f"DATASET PREPARATION COMPLETED in {elapsed_total/60:.1f} minutes!", flush=True)
    print(f"Total Selected Clips: {len(selected_samples)}", flush=True)
    print(f"Total Audio Duration: {total_duration:.2f} seconds ({total_hours:.2f} hours)", flush=True)
    print(f"Mean Clip Duration:   {total_duration/len(selected_samples):.2f} seconds", flush=True)
    print("=" * 70, flush=True)
    
    manifest_data = {
        "dataset_name": "eduardem/romanian-tts-single-speaker",
        "curriculum_phase": "T4_5hour_data_scale",
        "sample_count": len(selected_samples),
        "total_duration_s": round(total_duration, 2),
        "total_duration_hours": round(total_hours, 2),
        "target_hours": 5.0,
        "seed": seed,
        "sampling_rate": 24000,
        "rvq_codebooks": 16,
        "frame_rate_hz": 12.5,
        "excluded_holdout_count": len(excluded_texts),
        "manifest_path": out_manifest,
        "codes_path": out_codes_file,
        "samples": selected_samples
    }
    
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        
    return out_codes_file

if __name__ == "__main__":
    prepare_5h_dataset()
