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

def train_ro150(num_epochs=5):
    prevent_windows_sleep()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"TRAINING ROMANIAN MTP LORA ON EXPANDED 150-CLIP DATASET ({torch.cuda.get_device_name(0)})")
    print(f"{'='*70}\n")

    torch.cuda.empty_cache()
    gc.collect()

    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    cp_config = full_config.talker_config.code_predictor_config

    cp_model = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(cp_config)

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

    cp_model.load_state_dict(weights, strict=True)
    del weights
    gc.collect()

    cp_model = cp_model.to(device=device, dtype=torch.bfloat16)
    layer0_embedding = nn.Embedding.from_pretrained(layer0_weight.to(device=device, dtype=torch.bfloat16), freeze=True)

    # Continue from Phase 1 checkpoint if present
    if os.path.exists("models/romanian_mtp_lora/final/adapter_model.safetensors"):
        print("Continuing fine-tuning from Phase 1 checkpoint (models/romanian_mtp_lora/final)...")
        peft_model = PeftModel.from_pretrained(cp_model, "models/romanian_mtp_lora/final", is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        peft_model = get_peft_model(cp_model, lora_config)

    peft_model.print_trainable_parameters()

    # Load all 150 clips
    dataset = []
    with open("reports/mimi_codes_ro150.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dataset.append({
                "sample_id": d["sample_id"],
                "codes": torch.tensor(d["codes_16"], dtype=torch.long)
            })
    print(f"Loaded {len(dataset)} Romanian speech utterances from ro150!")

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = num_epochs * len(dataset) # 750 steps
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    peft_model.train()
    loss_history = []
    global_step = 0
    start_time = time.time()
    depth_indices = [0, 2, 4, 7, 10, 14]

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.time()
        for sample in dataset:
            global_step += 1
            sample_codes = sample["codes"].to(device)
            layer0_codes = sample_codes[0]
            layer0_embeds = layer0_embedding(layer0_codes)
            predictor_embeds = peft_model.model.model.codec_embedding

            step_loss = 0.0
            hidden_flat = layer0_embeds.unsqueeze(1)
            layer0_flat = layer0_embeds.unsqueeze(1)

            for mtp_layer_idx in depth_indices:
                embed_list = [hidden_flat, layer0_flat]
                for prev_layer in range(mtp_layer_idx):
                    prev_codes = sample_codes[prev_layer + 1]
                    prev_embed = predictor_embeds[prev_layer](prev_codes).unsqueeze(1)
                    embed_list.append(prev_embed)

                mtp_inputs = torch.cat(embed_list, dim=1)
                target_labels = sample_codes[mtp_layer_idx + 1]

                outputs = peft_model(
                    inputs_embeds=mtp_inputs,
                    generation_steps=mtp_layer_idx,
                    use_cache=False
                )
                logits = outputs.logits[:, -1, :]
                step_loss = step_loss + F.cross_entropy(logits, target_labels)

            loss = step_loss / len(depth_indices)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()

            if global_step % 25 == 0 or global_step == 1:
                cur_vram = torch.cuda.memory_allocated() / (1024**3)
                print(f"[ro150 Epoch {epoch:2d}/{num_epochs:2d}] Step {global_step:4d}/{total_steps:4d} | "
                      f"Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e} | VRAM: {cur_vram:.2f} GB")
                loss_history.append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "lr": round(lr_scheduler.get_last_lr()[0], 7)
                })

        avg_loss = epoch_loss / len(dataset)
        epoch_dur = time.time() - epoch_start
        print(f"--> [ro150] Epoch {epoch:2d} Complete! Avg Loss: {avg_loss:.4f} ({epoch_dur:.2f}s)")
        
        # Save checkpoint
        ckpt_dir = f"models/romanian_mtp_lora_ro150/checkpoint-epoch-{epoch}"
        os.makedirs(ckpt_dir, exist_ok=True)
        peft_model.save_pretrained(ckpt_dir)

    final_ro150 = "models/romanian_mtp_lora_ro150/final"
    os.makedirs(final_ro150, exist_ok=True)
    peft_model.save_pretrained(final_ro150)
    print(f"\nFinal 150-clip Romanian MTP LoRA saved to {final_ro150}!")

    total_time = time.time() - start_time
    print(f"Training on 150 clips completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

    with open("reports/mtp_ro150_training_loss.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)

    restore_windows_sleep()

if __name__ == "__main__":
    train_ro150(num_epochs=5)
