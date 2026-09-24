import json
import os
import sys
import torch
import soundfile as sf
from transformers import MimiModel, AutoFeatureExtractor

sys.stdout.reconfigure(encoding="utf-8")

def run_mimi_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading kyutai/mimi onto {device}...")
    
    torch.cuda.reset_peak_memory_stats()
    mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(device)
    mimi_model.eval()
    
    os.makedirs("outputs/mimi_decode", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    manifest_path = "data/ro30/manifest.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
        
    print(f"Loaded {len(samples)} samples from {manifest_path}")
    
    mimi_codes_records = []
    
    for i, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        wav_path = sample["wav_path"]
        
        audio, sr = sf.read(wav_path)
        if sr != 24000:
            raise ValueError(f"Sample {sample_id} sample rate is {sr}, expected 24000")
            
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(1).to(device)
        
        with torch.no_grad():
            encoder_outputs = mimi_model.encode(audio_tensor)
            codes = encoder_outputs.audio_codes  # shape: [batch, quantizers, time]
            
            # Extract first 16 codebooks
            codes_16 = codes[:, :16, :]
            
            # Decode using 16 codebooks as baseline control
            decoder_outputs_16 = mimi_model.decode(codes_16)
            recon_audio_16 = decoder_outputs_16.audio_values.squeeze().cpu().numpy()
            
        recon_wav_path = f"outputs/mimi_decode/{sample_id}.wav"
        sf.write(recon_wav_path, recon_audio_16, 24000)
        
        min_token = int(codes_16.min().item())
        max_token = int(codes_16.max().item())
        n_frames = codes_16.shape[2]
        orig_dur = len(audio) / sr
        recon_dur = len(recon_audio_16) / 24000
        
        record = {
            "sample_id": sample_id,
            "transcript": sample["transcript"],
            "original_wav": wav_path,
            "mimi_wav": recon_wav_path,
            "tensor_shape": list(codes.shape),
            "codes_16_shape": list(codes_16.shape),
            "dtype": str(codes.dtype),
            "num_quantizers_total": codes.shape[1],
            "num_quantizers_used": codes_16.shape[1],
            "num_time_frames": n_frames,
            "token_id_min": min_token,
            "token_id_max": max_token,
            "original_duration_s": round(orig_dur, 3),
            "reconstructed_duration_s": round(recon_dur, 3),
            "codes_16": codes_16.squeeze(0).cpu().tolist()  # [16, T] list of ints
        }
        mimi_codes_records.append(record)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1:02d}/30] {sample_id} | Shape: {list(codes_16.shape)} | Tokens: [{min_token}, {max_token}] | Dur: {orig_dur:.2f}s -> {recon_dur:.2f}s")
            
    with open("reports/mimi_codes.jsonl", "w", encoding="utf-8") as f:
        for r in mimi_codes_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"\nAll 30 samples encoded and decoded with Mimi successfully!")
    print(f"Saved token records to reports/mimi_codes.jsonl")
    print(f"Saved reconstructed audio to outputs/mimi_decode/")
    print(f"Peak VRAM during Mimi baseline: {peak_vram_mb:.2f} MB ({peak_vram_mb/1024:.2f} GB)")

if __name__ == "__main__":
    run_mimi_baseline()
