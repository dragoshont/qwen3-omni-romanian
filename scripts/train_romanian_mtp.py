import gc
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

def prevent_windows_sleep():
    """Instruct Windows not to sleep during background training."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("Windows Sleep Prevention: ACTIVE (PC will not sleep while script runs)")
    except Exception as e:
        print(f"Windows Sleep Prevention notice: {e}")

def run_mtp_training(num_epochs=10, lr=1e-4, lora_rank=16, lora_alpha=32):
    prevent_windows_sleep()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"STARTING REAL ROMANIAN MTP LoRA TRAINING ON {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    torch.cuda.empty_cache()
    gc.collect()

    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    cp_config = full_config.talker_config.code_predictor_config

    print("Instantiating CodePredictor model...")
    cp_model = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(cp_config)

    # Load MTP weights from shards
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

    print(f"Loaded {len(weights)} MTP tensors strictly from shards.")
    cp_model.load_state_dict(weights, strict=True)
    del weights
    gc.collect()

    # Move to CUDA in BF16
    cp_model = cp_model.to(device=device, dtype=torch.bfloat16)

    # Layer 0 embedding table
    layer0_embedding = nn.Embedding.from_pretrained(layer0_weight.to(device=device, dtype=torch.bfloat16), freeze=True)
    print(f"Loaded Layer 0 embedding table: {layer0_embedding.weight.shape}")

    # Attach LoRA (rank=16, alpha=32, target all attention + MLP projections for deep Romanian adaptation)
    print("\nApplying LoRA adapter to MTP...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    peft_model = get_peft_model(cp_model, lora_config)
    peft_model.print_trainable_parameters()

    # Load 30 Romanian clips codes
    print("\nLoading Romanian training dataset from reports/mimi_codes.jsonl...")
    dataset = []
    with open("reports/mimi_codes.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # codes_16 is list of 16 lists of ints [T]
            codes = torch.tensor(data["codes_16"], dtype=torch.long) # [16, T]
            dataset.append({
                "sample_id": data["sample_id"],
                "transcript": data["transcript"],
                "codes": codes
            })
    print(f"Loaded {len(dataset)} Romanian speech utterances.")

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataset), eta_min=1e-6)

    peft_model.train()
    total_steps = num_epochs * len(dataset)
    print(f"Training Plan: {num_epochs} Epochs × {len(dataset)} samples = {total_steps} total training steps.")
    print("-" * 70)

    loss_history = []
    global_step = 0
    start_time = time.time()
    
    os.makedirs("models/romanian_mtp_lora", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for sample_idx, sample in enumerate(dataset):
            global_step += 1
            sample_codes = sample["codes"].to(device) # [16, T]
            T = sample_codes.shape[1]
            
            # Layer 0 tokens and embeddings
            layer0_codes = sample_codes[0] # [T]
            layer0_embeds = layer0_embedding(layer0_codes) # [T, 1024]
            
            # Predictor embeddings for layers 1..15
            predictor_embeds = peft_model.model.model.codec_embedding
            
            # Sample 4 layers randomly or compute average over key codebook layers to be fast and memory efficient
            # To train robustly, we select a subset of 3 codebook depth steps per frame or full pass
            # Let's train across 4 depth steps: layer 1, layer 4, layer 8, layer 12
            depth_indices = [0, 3, 7, 11] # predicting codebook 1, 4, 8, 12
            step_loss = 0.0
            
            hidden_flat = layer0_embeds.unsqueeze(1) # [T, 1, 1024]
            layer0_flat = layer0_embeds.unsqueeze(1) # [T, 1, 1024]
            
            for mtp_layer_idx in depth_indices:
                embed_list = [hidden_flat, layer0_flat]
                for prev_layer in range(mtp_layer_idx):
                    prev_codes = sample_codes[prev_layer + 1] # [T]
                    prev_embed = predictor_embeds[prev_layer](prev_codes).unsqueeze(1) # [T, 1, 1024]
                    embed_list.append(prev_embed)
                    
                mtp_inputs = torch.cat(embed_list, dim=1) # [T, prefix_len, 1024]
                target_labels = sample_codes[mtp_layer_idx + 1] # [T]
                
                outputs = peft_model(
                    inputs_embeds=mtp_inputs,
                    generation_steps=mtp_layer_idx,
                    use_cache=False
                )
                logits = outputs.logits[:, -1, :] # [T, 2048]
                layer_loss = F.cross_entropy(logits, target_labels)
                step_loss = step_loss + layer_loss
                
            loss = step_loss / len(depth_indices)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            
            if global_step % 10 == 0 or global_step == 1:
                cur_vram = torch.cuda.memory_allocated() / (1024**3)
                print(f"Epoch {epoch:2d}/{num_epochs:2d} | Step {global_step:4d}/{total_steps:4d} | "
                      f"Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e} | VRAM: {cur_vram:.2f} GB")
                
                loss_history.append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "lr": round(lr_scheduler.get_last_lr()[0], 7),
                    "vram_gb": round(cur_vram, 3),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
        avg_epoch_loss = epoch_loss / len(dataset)
        epoch_sec = time.time() - epoch_start
        print(f"--> Epoch {epoch:2d} Complete! Avg Loss: {avg_epoch_loss:.4f} ({epoch_sec:.2f}s)")
        
        # Save checkpoint at epoch 5 and 10
        if epoch % 5 == 0 or epoch == num_epochs:
            ckpt_path = f"models/romanian_mtp_lora/checkpoint-epoch-{epoch}"
            peft_model.save_pretrained(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = "models/romanian_mtp_lora/final"
    peft_model.save_pretrained(final_path)
    print(f"\nFinal Romanian MTP LoRA saved to {final_path}!")

    total_sec = time.time() - start_time
    print(f"\nTraining completed in {total_sec:.2f} seconds ({total_sec/60:.2f} minutes).")

    # Save training history
    with open("reports/mtp_training_loss.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)

    # Save Markdown progress curve
    with open("reports/mtp_training_curve.md", "w", encoding="utf-8") as f:
        f.write("# Romanian MTP LoRA Training Curve\n\n")
        f.write(f"- **GPU**: {torch.cuda.get_device_name(0)}\n")
        f.write(f"- **Epochs**: {num_epochs}\n")
        f.write(f"- **Total Steps**: {total_steps}\n")
        f.write(f"- **Final Loss**: {loss_history[-1]['loss']:.4f}\n")
        f.write(f"- **Total Duration**: {total_sec:.2f}s\n\n")
        f.write("| Step | Epoch | Loss | Learning Rate | VRAM (GB) |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for entry in loss_history:
            f.write(f"| {entry['step']} | {entry['epoch']} | {entry['loss']} | {entry['lr']} | {entry['vram_gb']} |\n")

    print("Saved reports/mtp_training_loss.json and reports/mtp_training_curve.md")

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_mtp_training(num_epochs=epochs)
