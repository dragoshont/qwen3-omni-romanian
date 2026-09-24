import gc
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeTalkerForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

def test_talker_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TESTING STANDALONE TALKER LORA TRAINING STEP ON {device} ===")
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Config
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    talker_config = full_config.talker_config
    
    # 2. Instantiate on CPU
    print("Instantiating Talker on CPU in BF16...")
    talker = Qwen3OmniMoeTalkerForConditionalGeneration(talker_config).to(dtype=torch.bfloat16)
    
    # 3. Load tensors from shards 13, 14, 15
    shards = [
        os.path.join(config_dir, "model-00013-of-00015.safetensors"),
        os.path.join(config_dir, "model-00014-of-00015.safetensors"),
        os.path.join(config_dir, "model-00015-of-00015.safetensors")
    ]
    raw_weights = {}
    for s in shards:
        sd = safetensors.torch.load_file(s)
        for k, v in sd.items():
            if k.startswith("talker."):
                raw_weights[k[len("talker."):]] = v
        del sd
        gc.collect()
        
    print(f"Loaded {len(raw_weights)} raw Talker tensors from local shards.")
    
    # Assemble state dict
    assembled = {}
    target_keys = set(talker.state_dict().keys())
    for k in target_keys:
        if "mlp.experts.gate_up_proj" in k or "mlp.experts.down_proj" in k:
            continue
        if k in raw_weights:
            assembled[k] = raw_weights[k].to(torch.bfloat16)
            
    num_layers = talker_config.text_config.num_hidden_layers
    num_experts = talker_config.text_config.num_local_experts
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
    
    talker.load_state_dict(assembled, strict=True)
    del assembled
    gc.collect()
    
    print("Moving Talker to CUDA (BF16)...")
    talker = talker.to(device=device)
    
    # 4. Apply LoRA adapter to Talker attention projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    peft_talker = get_peft_model(talker, lora_config)
    peft_talker.print_trainable_parameters()
    
    # Enable gradient checkpointing to save VRAM
    peft_talker.model.model.gradient_checkpointing_enable()
    peft_talker.train()
    
    # 5. Read one real Romanian sample
    with open("reports/mimi_codes.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        
    sample_codes = torch.tensor(sample["codes_16"], dtype=torch.long, device=device) # [16, T]
    T = sample_codes.shape[1]
    print(f"Testing on Romanian utterance {sample['sample_id']} (duration: {sample['duration']}s, frames: {T})")
    
    # Build speech embedding inputs: sum of all 16 codebook embeddings for each frame
    # Layer 0 embedding from talker.model.codec_embedding
    layer0_codes = sample_codes[0] # [T]
    codec_embed = peft_talker.model.model.codec_embedding(layer0_codes) # [T, 1024]
    
    # Code predictor embeddings for layers 1..15
    pred_embeds = peft_talker.model.code_predictor.model.codec_embedding
    all_embeds = codec_embed.clone()
    for j in range(15):
        c_j = sample_codes[j + 1]
        all_embeds = all_embeds + pred_embeds[j](c_j)
        
    # Input embeddings: [1, T, 1024]
    inputs_embeds = all_embeds.unsqueeze(0)
    attention_mask = torch.ones((1, T), dtype=torch.long, device=device)
    
    # Target labels: predict next Layer 0 code token
    labels = layer0_codes.unsqueeze(0) # [1, T]
    
    optimizer = torch.optim.AdamW(peft_talker.parameters(), lr=2e-5)
    
    t0 = time.time()
    print("Running Forward Pass through 3.32B Talker...")
    outputs = peft_talker(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False
    )
    
    logits = outputs.logits # [1, T, vocab_size]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print(f"  Forward Pass complete! Initial Talker Loss: {loss.item():.4f}")
    
    print("Running Backward Pass...")
    optimizer.zero_grad()
    loss.backward()
    print("  Backward Pass complete!")
    
    print("Running Optimizer Step...")
    optimizer.step()
    print("  Optimizer Step complete!")
    
    step_duration = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    current_vram = torch.cuda.memory_allocated() / (1024**3)
    
    print("\n" + "="*50)
    print("TALKER TRAINING STEP BENCHMARK RESULTS:")
    print(f"  Initial Loss: {loss.item():.4f}")
    print(f"  Step Duration: {step_duration:.2f}s")
    print(f"  Peak VRAM: {peak_vram:.2f} GB / 15.89 GB")
    print(f"  Current VRAM: {current_vram:.2f} GB")
    print(f"  Safety Margin: {15.89 - peak_vram:.2f} GB free VRAM!")
    print(f"  Pass Criterion (<= 15.0 GB): {'PASS' if peak_vram <= 15.0 else 'FAIL'}")
    print("="*50)

if __name__ == "__main__":
    test_talker_step()
