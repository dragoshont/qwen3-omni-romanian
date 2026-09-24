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

def train_ro500_mtp(num_epochs=15, batch_size=1):
    prevent_windows_sleep()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f"STAGE 12B: Deep Romanian MTP Training on 500 Utterances ({num_epochs} Epochs)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

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

    # Continue from Phase 3 checkpoint if present, or initialize fresh
    init_ckpt = "models/romanian_mtp_lora_ro150/final"
    if os.path.exists(os.path.join(init_ckpt, "adapter_model.safetensors")):
        print(f"Transfer Learning: Initializing from previous best checkpoint ({init_ckpt})...")
        peft_model = PeftModel.from_pretrained(cp_model, init_ckpt, is_trainable=True)
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

    # Load 500-clip dataset
    codes_file = "reports/mimi_codes_ro500.jsonl"
    dataset = []
    with open(codes_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                dataset.append({
                    "sample_id": d["sample_id"],
                    "codes": torch.tensor(d["codes_16"], dtype=torch.long)
                })

    print(f"Loaded {len(dataset)} Romanian speech utterances for deep training!")
    total_steps = num_epochs * len(dataset)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=4e-5, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    loss_history = []
    global_step = 0
    start_time = time.time()
    peft_model.train()

    depth_indices = [0, 1, 2, 4, 6, 8, 10, 12, 14] # 9 depth levels for richer acoustic resolution

    os.makedirs("models/romanian_mtp_lora_ro500", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        t_epoch_start = time.time()
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

            if global_step % 50 == 0 or global_step == 1:
                cur_vram = torch.cuda.memory_allocated() / (1024**3)
                print(f"  [Epoch {epoch:2d}/{num_epochs:2d}] Step {global_step:4d}/{total_steps:4d} | "
                      f"Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e} | VRAM: {cur_vram:.2f} GB")
                loss_history.append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "lr": round(lr_scheduler.get_last_lr()[0], 7)
                })

        avg_loss = epoch_loss / len(dataset)
        epoch_dur = time.time() - t_epoch_start
        print(f"==> Epoch {epoch}/{num_epochs} Complete! Avg Loss: {avg_loss:.4f} | Time: {epoch_dur:.1f}s")

        if epoch % 5 == 0 or epoch == num_epochs:
            ckpt_path = f"models/romanian_mtp_lora_ro500/checkpoint-epoch-{epoch}"
            peft_model.save_pretrained(ckpt_path)
            print(f"--> Saved Checkpoint: {ckpt_path}")

    final_path = "models/romanian_mtp_lora_ro500/final"
    peft_model.save_pretrained(final_path)
    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    print("\n" + "="*50)
    print("STAGE 12B TRAINING COMPLETE!")
    print(f"  Total Duration: {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"  Initial Loss: {loss_history[0]['loss']:.4f}")
    print(f"  Final Loss: {loss_history[-1]['loss']:.4f}")
    print(f"  Peak VRAM: {peak_vram:.2f} GB / 15.89 GB")
    print(f"  Saved Final Checkpoint: {final_path}")
    print("="*50)

    with open("reports/mtp_ro500_training_loss.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)

    del peft_model, layer0_embedding
    torch.cuda.empty_cache()
    gc.collect()
    restore_windows_sleep()
    return loss_history

if __name__ == "__main__":
    train_ro500_mtp(num_epochs=15)
