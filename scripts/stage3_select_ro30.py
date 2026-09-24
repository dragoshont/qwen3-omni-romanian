import io
import json
import os
import re
import sys
import datasets
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")

def select_30_clips():
    seed = 42
    print(f"Loading streaming dataset 'eduardem/romanian-tts-single-speaker' with seed={seed}...")
    
    ds = datasets.load_dataset("eduardem/romanian-tts-single-speaker", split="train", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))
    
    # Target features to cover across the 30 clips
    # ă â î ș ț
    # ce / ci, ge / gi, che / chi, ghe / ghi
    # questions, short, long
    
    os.makedirs("data/ro30/wav", exist_ok=True)
    
    selected = []
    seen_texts = set()
    
    # Feature matchers
    def analyze_text(text):
        t = text.lower()
        feats = []
        if 'ă' in t: feats.append('ă')
        if 'â' in t: feats.append('â')
        if 'î' in t: feats.append('î')
        if 'ș' in t or 'ş' in t: feats.append('ș')
        if 'ț' in t or 'ţ' in t: feats.append('ț')
        if 'ce' in t or 'ci' in t: feats.append('ce/ci')
        if 'ge' in t or 'gi' in t: feats.append('ge/gi')
        if 'che' in t or 'chi' in t: feats.append('che/chi')
        if 'ghe' in t or 'ghi' in t: feats.append('ghe/ghi')
        if '?' in text: feats.append('question')
        if any(c.isdigit() for c in text): feats.append('numbers')
        return feats

    print("Scanning dataset stream for phonetically rich and diverse Romanian clips...")
    
    # Desired buckets to ensure balance
    bucket_counts = {
        'ă': 0, 'â': 0, 'î': 0, 'ș': 0, 'ț': 0,
        'ce/ci': 0, 'ge/gi': 0, 'che/chi': 0, 'ghe/ghi': 0,
        'question': 0, 'short': 0, 'long': 0
    }

    item_idx = 0
    for sample in ds:
        item_idx += 1
        text = sample.get('text', '').strip()
        duration = float(sample.get('duration', 0.0))
        
        if not text or duration < 2.0 or duration > 12.0:
            continue
        if text in seen_texts:
            continue
            
        feats = analyze_text(text)
        
        # Decide if this sample adds value to our 30-clip battery
        is_valuable = False
        reasons = []
        
        if '?' in text and bucket_counts['question'] < 3:
            is_valuable = True
            reasons.append("Question intonation")
            bucket_counts['question'] += 1
        elif 'ghe/ghi' in feats and bucket_counts['ghe/ghi'] < 3:
            is_valuable = True
            reasons.append("Affricate ghe/ghi")
            bucket_counts['ghe/ghi'] += 1
        elif 'che/chi' in feats and bucket_counts['che/chi'] < 3:
            is_valuable = True
            reasons.append("Affricate che/chi")
            bucket_counts['che/chi'] += 1
        elif 'ge/gi' in feats and bucket_counts['ge/gi'] < 3:
            is_valuable = True
            reasons.append("Affricate ge/gi")
            bucket_counts['ge/gi'] += 1
        elif duration <= 3.5 and bucket_counts['short'] < 4:
            is_valuable = True
            reasons.append("Short phrase (<3.5s)")
            bucket_counts['short'] += 1
        elif duration >= 8.0 and bucket_counts['long'] < 4:
            is_valuable = True
            reasons.append("Long complex sentence (>8s)")
            bucket_counts['long'] += 1
        elif len(feats) >= 3 and len(selected) < 30:
            # High diacritic diversity
            missing = [f for f in ['ă', 'â', 'î', 'ș', 'ț', 'ce/ci'] if bucket_counts[f] < 5 and f in feats]
            if missing or len(selected) < 20:
                is_valuable = True
                reasons.append(f"Diacritic cluster ({', '.join(feats)})")
                for f in feats:
                    if f in bucket_counts: bucket_counts[f] += 1
                    
        if is_valuable or (len(selected) < 30 and item_idx % 2 == 0):
            if not reasons:
                reasons.append(f"Phonetic diversity: {', '.join(feats) if feats else 'standard'}")
            seen_texts.add(text)
            
            # Decode audio bytes
            wav_bytes = sample['audio']['bytes']
            audio_data, sr = sf.read(io.BytesIO(wav_bytes))
            
            sample_id = f"ro_sample_{len(selected)+1:02d}"
            wav_filename = f"data/ro30/wav/{sample_id}.wav"
            sf.write(wav_filename, audio_data, sr)
            
            clip_meta = {
                "sample_id": sample_id,
                "original_speaker": sample.get('speaker', 'Sanda'),
                "transcript": text,
                "duration": round(float(duration), 3),
                "sample_rate": sr,
                "num_samples": len(audio_data),
                "wav_path": wav_filename,
                "selection_reason": "; ".join(reasons),
                "linguistic_features": feats
            }
            selected.append(clip_meta)
            print(f"[{len(selected):02d}/30] {sample_id} ({duration:.1f}s): {text[:60]}... -> {clip_meta['selection_reason']}")
            
            if len(selected) == 30:
                break
                
    manifest_path = "data/ro30/manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\nSuccessfully selected and saved 30 Romanian clips to {manifest_path}")
    print("Feature distribution:")
    for k, v in bucket_counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    select_30_clips()
