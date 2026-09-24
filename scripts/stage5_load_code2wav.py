import gc
import json
import os
import sys
import torch
import safetensors.torch
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeCode2Wav

sys.stdout.reconfigure(encoding="utf-8")

def run_stage5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 5: Loading Standalone Qwen Code2Wav onto {device} ===")
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    mem_before_load = torch.cuda.memory_allocated() / (1024**2)
    
    # 1. Load config
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    code2wav_config = full_config.code2wav_config
    
    # 2. Instantiate Code2Wav module
    print("Instantiating Qwen3OmniMoeCode2Wav...")
    code2wav = Qwen3OmniMoeCode2Wav(code2wav_config)
    
    # 3. Load checkpoint tensors strictly
    shard_path = os.path.join(config_dir, "model-00015-of-00015.safetensors")
    print(f"Loading weights from {shard_path}...")
    sd = safetensors.torch.load_file(shard_path)
    
    prefix = "code2wav."
    c2w_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    print(f"Filtered {len(c2w_sd)} tensors matching '{prefix}'")
    
    load_res = code2wav.load_state_dict(c2w_sd, strict=True)
    print(f"Strict load result: {load_res}")
    
    # 4. Cast to BF16 and move to CUDA
    code2wav = code2wav.to(device=device, dtype=torch.bfloat16)
    code2wav.eval()
    for param in code2wav.parameters():
        param.requires_grad_(False)
        
    mem_after_load = torch.cuda.memory_allocated() / (1024**2)
    
    # Calculate parameter count and weight memory
    total_params = sum(p.numel() for p in code2wav.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in code2wav.parameters())
    weight_mb = weight_bytes / (1024**2)
    
    # 5. Test dry-run inference with dummy codes
    test_codes = torch.randint(0, 2048, (1, 16, 50), device=device, dtype=torch.long)
    with torch.no_grad():
        test_wav = code2wav(test_codes)
        
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    
    print("\n" + "="*50)
    print("Code2Wav Standalone Module Report:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Weight memory (BF16): {weight_mb:.2f} MB ({weight_mb/1024:.2f} GB)")
    print(f"  GPU memory before load: {mem_before_load:.2f} MB")
    print(f"  GPU memory after load: {mem_after_load:.2f} MB ({mem_after_load/1024:.2f} GB)")
    print(f"  Peak GPU memory (with forward pass): {peak_mem:.2f} MB ({peak_mem/1024:.2f} GB)")
    print(f"  Output audio shape: {tuple(test_wav.shape)}")
    print(f"  Audio min/max: [{test_wav.min().item():.3f}, {test_wav.max().item():.3f}]")
    print("="*50)
    
    report = {
        "status": "PASS",
        "total_parameters": total_params,
        "weight_memory_mb": round(weight_mb, 2),
        "gpu_memory_before_load_mb": round(mem_before_load, 2),
        "gpu_memory_after_load_mb": round(mem_after_load, 2),
        "peak_gpu_memory_mb": round(peak_mem, 2),
        "test_input_shape": list(test_codes.shape),
        "test_output_shape": list(test_wav.shape),
        "strict_load": True
    }
    
    with open("reports/code2wav_load_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("Stage 5 completed successfully! Saved reports/code2wav_load_report.json")

if __name__ == "__main__":
    run_stage5()
