import gc
import os
import sys
import json
import torch
import safetensors.torch
from transformers import Qwen3OmniMoeConfig

sys.stdout.reconfigure(encoding="utf-8")

def export_standalone_talker():
    out_dir = "models/qwen3-omni-talker"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Exporting standalone Talker folder to: {out_dir}")
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    talker_config = full_config.talker_config
    
    # 1. Save config.json
    talker_config_dict = talker_config.to_dict()
    # Ensure architecturs is set
    talker_config_dict["architectures"] = ["Qwen3OmniMoeTalkerForConditionalGeneration"]
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(talker_config_dict, f, indent=2)
    print("Saved config.json")
    
    # Check if weights already exported
    safetensor_file = os.path.join(out_dir, "model.safetensors")
    if os.path.exists(safetensor_file):
        print(f"Standalone Talker weights already exist at {safetensor_file} ({os.path.getsize(safetensor_file) / (1024**3):.2f} GB)!")
        return out_dir
        
    # 2. Extract weights from shards 13, 14, 15
    shards = [
        os.path.join(config_dir, "model-00013-of-00015.safetensors"),
        os.path.join(config_dir, "model-00014-of-00015.safetensors"),
        os.path.join(config_dir, "model-00015-of-00015.safetensors")
    ]
    
    raw_weights = {}
    for s in shards:
        print(f"Loading {os.path.basename(s)}...")
        sd = safetensors.torch.load_file(s)
        for k, v in sd.items():
            if k.startswith("talker."):
                raw_weights[k[len("talker."):]] = v
        del sd
        gc.collect()
        
    print(f"Collected {len(raw_weights)} raw Talker tensors. Assembling state dict...")
    assembled = {}
    for k, v in raw_weights.items():
        if "mlp.experts.gate_proj" in k or "mlp.experts.up_proj" in k or "mlp.experts.down_proj" in k:
            continue
        assembled[k] = v.to(torch.bfloat16)
        
    num_layers = talker_config.text_config.num_hidden_layers
    num_experts = talker_config.text_config.num_local_experts
    
    print(f"Stacking MoE experts ({num_layers} layers x {num_experts} experts)...")
    for l in range(num_layers):
        gate_list, up_list, down_list = [], [], []
        for e in range(num_experts):
            g_key = f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"
            u_key = f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"
            d_key = f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"
            gate_list.append(raw_weights[g_key].to(torch.bfloat16))
            up_list.append(raw_weights[u_key].to(torch.bfloat16))
            down_list.append(raw_weights[d_key].to(torch.bfloat16))
            
        gate_stack = torch.stack(gate_list, dim=0)
        up_stack = torch.stack(up_list, dim=0)
        gate_up_proj = torch.cat([gate_stack, up_stack], dim=1)
        down_proj = torch.stack(down_list, dim=0)
        
        assembled[f"model.layers.{l}.mlp.experts.gate_up_proj"] = gate_up_proj
        assembled[f"model.layers.{l}.mlp.experts.down_proj"] = down_proj
        
    del raw_weights
    gc.collect()
    
    print(f"Saving assembled safetensors to {safetensor_file}...")
    safetensors.torch.save_file(assembled, safetensor_file)
    print(f"Export complete! File size: {os.path.getsize(safetensor_file) / (1024**3):.2f} GB")
    return out_dir

if __name__ == "__main__":
    export_standalone_talker()
