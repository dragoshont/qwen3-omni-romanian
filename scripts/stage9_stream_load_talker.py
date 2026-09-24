import gc
import json
import os
import sys
import torch
import safetensors
import accelerate
from accelerate.utils import set_module_tensor_to_device
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeTalkerForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

def stream_load_talker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 9: Standalone Talker/MTP Stream Load onto {device} ===")
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    mem_before = torch.cuda.memory_allocated() / (1024**3)
    
    # 1. Config & Empty Meta Model
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    talker_config = full_config.talker_config
    
    print("Initializing Talker model on meta device (0 bytes RAM)...")
    with accelerate.init_empty_weights():
        talker = Qwen3OmniMoeTalkerForConditionalGeneration(talker_config)
        
    total_params = sum(p.numel() for p in talker.parameters())
    print(f"Model architecture: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    
    # 2. Build index of shard paths
    shards = [
        os.path.join(config_dir, "model-00013-of-00015.safetensors"),
        os.path.join(config_dir, "model-00014-of-00015.safetensors"),
        os.path.join(config_dir, "model-00015-of-00015.safetensors")
    ]
    
    key_to_shard = {}
    for s in shards:
        with safetensors.safe_open(s, framework="pt") as f:
            for k in f.keys():
                if k.startswith("talker."):
                    key_to_shard[k[len("talker."):]] = s
                    
    print(f"Mapped {len(key_to_shard)} Talker keys across 3 shards.")
    
    handles = {s: safetensors.safe_open(s, framework="pt") for s in shards}
    
    def get_tensor_cpu(key):
        shard = key_to_shard[key]
        return handles[shard].get_tensor("talker." + key).to(dtype=torch.bfloat16)
    
    # 3. Load non-expert parameters directly onto CUDA
    print("Streaming non-expert weights directly to GPU...")
    model_state = talker.state_dict()
    for name in model_state.keys():
        if "mlp.experts.gate_up_proj" in name or "mlp.experts.down_proj" in name:
            continue
        if name in key_to_shard:
            val = get_tensor_cpu(name)
            set_module_tensor_to_device(talker, name, device=device, value=val)
            del val
            
    torch.cuda.empty_cache()
    print(f"Non-expert weights loaded. GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    # 4. Stream and assemble expert weights on CPU first, then transfer single packed tensor to GPU
    num_layers = talker_config.text_config.num_hidden_layers # 20
    num_experts = talker_config.text_config.num_local_experts # 128
    
    print(f"Streaming MoE expert weights for {num_layers} layers x {num_experts} experts (CPU assembly -> GPU)...")
    for l in range(num_layers):
        gate_list = []
        up_list = []
        down_list = []
        for e in range(num_experts):
            g_key = f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"
            u_key = f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"
            d_key = f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"
            
            gate_list.append(get_tensor_cpu(g_key))
            up_list.append(get_tensor_cpu(u_key))
            down_list.append(get_tensor_cpu(d_key))
            
        # Assemble in CPU memory (zero GPU fragmentation)
        gate_stack = torch.stack(gate_list, dim=0)
        up_stack = torch.stack(up_list, dim=0)
        gate_up_proj = torch.cat([gate_stack, up_stack], dim=1).contiguous()
        down_proj = torch.stack(down_list, dim=0).contiguous()
        
        del gate_list, up_list, down_list, gate_stack, up_stack
        
        # Single clean transfer to GPU
        set_module_tensor_to_device(talker, f"model.layers.{l}.mlp.experts.gate_up_proj", device=device, value=gate_up_proj)
        set_module_tensor_to_device(talker, f"model.layers.{l}.mlp.experts.down_proj", device=device, value=down_proj)
        
        del gate_up_proj, down_proj
        
        if (l + 1) % 5 == 0:
            torch.cuda.empty_cache()
            print(f"  Loaded layer {l+1}/{num_layers} experts | VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            
    # Close handles
    for h in handles.values():
        del h
    gc.collect()
    torch.cuda.empty_cache()
    
    talker.eval()
    
    mem_after = torch.cuda.memory_allocated() / (1024**3)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
    print("\n" + "="*50)
    print("STANDALONE TALKER / MTP LOAD METRICS:")
    print(f"  Total Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  GPU VRAM Before: {mem_before:.2f} GB")
    print(f"  GPU VRAM After Load: {mem_after:.2f} GB / 15.89 GB")
    print(f"  Peak VRAM: {peak_mem:.2f} GB")
    print(f"  Free VRAM on RTX 5080: {15.89 - mem_after:.2f} GB")
    print("="*50)
    
    report = {
        "status": "PASS",
        "total_parameters": total_params,
        "vram_before_gb": round(mem_before, 3),
        "vram_after_load_gb": round(mem_after, 3),
        "peak_vram_gb": round(peak_mem, 3),
        "free_vram_gb": round(15.89 - mem_after, 3)
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/talker_load_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    return talker, talker_config

if __name__ == "__main__":
    stream_load_talker()
