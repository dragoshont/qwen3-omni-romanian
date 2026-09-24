import gc
import os
import sys
import torch
import safetensors.torch
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeTalkerForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

def test_load_talker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading standalone Talker onto {device}...")
    
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    talker_config = full_config.talker_config
    
    # 1. Instantiate Talker in BF16 directly on CPU
    print("Instantiating Qwen3OmniMoeTalkerForConditionalGeneration...")
    talker = Qwen3OmniMoeTalkerForConditionalGeneration(talker_config).to(dtype=torch.bfloat16)
    
    # 2. Load tensors from shards 13, 14, 15
    shards = [
        os.path.join(config_dir, "model-00013-of-00015.safetensors"),
        os.path.join(config_dir, "model-00014-of-00015.safetensors"),
        os.path.join(config_dir, "model-00015-of-00015.safetensors")
    ]
    
    raw_weights = {}
    for s in shards:
        print(f"Reading weights from {os.path.basename(s)}...")
        sd = safetensors.torch.load_file(s)
        for k, v in sd.items():
            if k.startswith("talker."):
                raw_weights[k[len("talker."):]] = v
        del sd
        gc.collect()
        
    print(f"Collected {len(raw_weights)} raw Talker tensors.")
    
    # 3. Assemble target state dict for Talker
    assembled = {}
    
    # Non-expert weights
    target_keys = set(talker.state_dict().keys())
    
    for k in target_keys:
        if "mlp.experts.gate_up_proj" in k or "mlp.experts.down_proj" in k:
            continue
        if k in raw_weights:
            assembled[k] = raw_weights[k].to(torch.bfloat16)
        else:
            print(f"Missing non-expert key: {k}")
            
    # Assemble MoE experts for 20 layers
    num_layers = talker_config.text_config.num_hidden_layers # 20
    num_experts = talker_config.text_config.num_local_experts # 128
    
    print(f"Assembling MoE expert matrices for {num_layers} layers x {num_experts} experts...")
    for l in range(num_layers):
        gate_list = []
        up_list = []
        down_list = []
        for e in range(num_experts):
            g_key = f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"
            u_key = f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"
            d_key = f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"
            
            gate_list.append(raw_weights[g_key].to(torch.bfloat16))
            up_list.append(raw_weights[u_key].to(torch.bfloat16))
            down_list.append(raw_weights[d_key].to(torch.bfloat16))
            
        # Stack experts
        # gate_proj: [128, intermediate_dim, hidden_dim]
        # up_proj: [128, intermediate_dim, hidden_dim]
        gate_stack = torch.stack(gate_list, dim=0)
        up_stack = torch.stack(up_list, dim=0)
        # gate_up_proj: [128, 2*intermediate_dim, hidden_dim]
        gate_up_proj = torch.cat([gate_stack, up_stack], dim=1)
        # down_proj: [128, hidden_dim, intermediate_dim]
        down_proj = torch.stack(down_list, dim=0)
        
        assembled[f"model.layers.{l}.mlp.experts.gate_up_proj"] = gate_up_proj
        assembled[f"model.layers.{l}.mlp.experts.down_proj"] = down_proj
        
    del raw_weights
    gc.collect()
    
    # 4. Strict load into talker
    print("Loading assembled state dict into Talker...")
    res = talker.load_state_dict(assembled, strict=True)
    print(f"Talker strict load result: {res}")
    
    del assembled
    gc.collect()
    
    # 5. Move to CUDA and measure VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("Moving Talker to CUDA (BF16)...")
    talker = talker.to(device=device)
    talker.eval()
    
    vram_alloc = torch.cuda.memory_allocated() / (1024**3)
    vram_res = torch.cuda.memory_reserved() / (1024**3)
    total_params = sum(p.numel() for p in talker.parameters())
    
    print("\n" + "="*50)
    print(f"TALKER STANDALONE LOADED SUCCESSFULLY!")
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"VRAM Allocated: {vram_alloc:.2f} GB / 15.89 GB")
    print(f"VRAM Reserved: {vram_res:.2f} GB")
    print(f"Headroom on RTX 5080: {15.89 - vram_alloc:.2f} GB free VRAM!")
    print("="*50)

if __name__ == "__main__":
    test_load_talker()
