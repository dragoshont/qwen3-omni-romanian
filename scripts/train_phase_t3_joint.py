import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
import json
import time
import gc
import math
import torch
import torch.nn.functional as F
import safetensors.torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from research_runtime import (
    EpochShuffleSampler,
    assert_adapter_isolation,
    environment_record,
    git_commit,
    load_resume_state,
    save_resume_state,
    seed_everything,
    sha256_file,
    write_json,
)

sys.stdout.reconfigure(encoding="utf-8")

def train_phase_t3():
    training_seed = int(os.environ.get("TRAINING_SEED", "42"))
    seed_everything(training_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit training protocol requires CUDA")
    device = torch.device("cuda:0")
    run_name = os.environ.get("RUN_NAME", "component_joint_1h_1000")
    print("=" * 70, flush=True)
    print("PHASE T3: Joint Training: Romanian Talker LoRA + Romanian MTP LoRA", flush=True)
    print("=" * 70, flush=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    output_dir = f"models/controlled/{run_name}/seed_{training_seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load tokenizer & Embeddings
    print("1. Loading Tokenizer and Thinker Embeddings...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    sd_embed = safetensors.torch.load_file("models/thinker_embed_tokens.safetensors")
    embed_weight = sd_embed["thinker.model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    
    # 2. Load 4-Bit Talker
    print("2. Loading 4-Bit Talker Backbone...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    talker = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        "models/qwen3-omni-talker",
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    
    # 3. Configure LoRA on both Talker and Code Predictor
    print("3. Configuring Joint LoRA on Talker and MTP...", flush=True)
    # Prepare the quantized base before attaching any adapters. The PEFT helper
    # freezes all parameters that already exist on the model.
    talker = prepare_model_for_kbit_training(talker, use_gradient_checkpointing=True)

    # First attach LoRA to Code Predictor
    mtp_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    talker.code_predictor = get_peft_model(talker.code_predictor, mtp_lora_config)
    print("Attached LoRA to MTP Code Predictor.", flush=True)
    
    # Then attach LoRA to the Talker proper. Exclusion is mandatory because
    # PEFT otherwise matches q_proj/v_proj recursively inside code_predictor.
    talker_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        exclude_modules=r".*code_predictor.*",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    talker = get_peft_model(talker, talker_lora_config)
    talker.print_trainable_parameters()
    
    # Precompute special token embeds
    tts_special_tokens = torch.tensor([[151672, 151673, 151671]], device=device, dtype=torch.long)
    tts_special_embed = F.embedding(tts_special_tokens, embed_weight)
    
    tp = talker.base_model.model.text_projection if hasattr(talker, "base_model") else talker.text_projection
    tts_projected = tp(tts_special_embed)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_projected.chunk(3, dim=1)
    
    speaker_id = 2302 # Ethan
    codec_special_tokens = torch.tensor([[2155, 2156, 2157, speaker_id, 2148, 2149]], device=device, dtype=torch.long)
    
    model_obj = talker.base_model.model if hasattr(talker, "base_model") else talker
    codec_special_embeds = model_obj.get_input_embeddings()(codec_special_tokens)
    assistant_codec_hidden = torch.cat(
        (torch.zeros((1, 3, 1024), device=device, dtype=torch.bfloat16), codec_special_embeds), dim=1
    )
    
    # 4. Load Training Data
    data_file = os.environ.get("TRAIN_DATA_FILE", "reports/mimi_codes_ro1000.jsonl")
    print(f"4. Loading training clips from {data_file}...", flush=True)
    train_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                codes = item.get("codes_16", [])
                if len(codes) > 0 and 15 <= len(codes[0]) <= 250:
                    train_samples.append(item)
                    
    print(f"Loaded {len(train_samples)} valid training samples.", flush=True)
    
    # 5. Training Hyperparameters
    NUM_STEPS = int(os.environ.get("NUM_STEPS", "1000"))
    GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
    LEARNING_RATE_TALKER = float(os.environ.get("LEARNING_RATE_TALKER", "2e-5"))
    LEARNING_RATE_MTP = float(os.environ.get("LEARNING_RATE_MTP", "4e-5"))
    WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "50"))
    SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "250"))
    
    # Distinct optimizer parameter groups for Talker and MTP
    talker_params, mtp_params, trainable_manifest = assert_adapter_isolation(
        talker, require_mtp=True
    )
    talker_trainable_count = trainable_manifest["talker_parameter_count"]
    mtp_trainable_count = trainable_manifest["mtp_parameter_count"]
    print(
        f"Verified adapter isolation: Talker={talker_trainable_count:,}, MTP={mtp_trainable_count:,}",
        flush=True,
    )
    
    optimizer = torch.optim.AdamW([
        {"params": talker_params, "lr": LEARNING_RATE_TALKER},
        {"params": mtp_params, "lr": LEARNING_RATE_MTP},
    ], betas=(0.9, 0.95), weight_decay=0.01)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_STEPS,
    )

    run_config = {
        "condition": "joint_talker_mtp",
        "run_name": run_name,
        "training_seed": training_seed,
        "data_file": data_file,
        "data_sha256": sha256_file(data_file),
        "sample_count": len(train_samples),
        "num_steps": NUM_STEPS,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate_talker": LEARNING_RATE_TALKER,
        "learning_rate_mtp": LEARNING_RATE_MTP,
        "warmup_steps": WARMUP_STEPS,
        "mtp_frame_subsample": 64,
        "sampler": "deterministic epoch shuffle",
        "speaker_token_id": speaker_id,
    }
    
    talker.train()
    optimizer.zero_grad()
    
    step_losses = []
    loss_history = []
    t_start = time.perf_counter()
    total_samples = len(train_samples)
    sampler = EpochShuffleSampler(total_samples, training_seed)
    start_step = 0
    resume_path = os.path.join(output_dir, "training_state.pt")
    if os.path.exists(resume_path) and os.environ.get("FORCE_FRESH", "0") != "1":
        start_step, loss_history, step_losses, _ = load_resume_state(
            resume_path,
            model=talker,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            sampler=sampler,
            expected_config=run_config,
        )
        print(f"Resumed exact training state after step {start_step}.", flush=True)
    
    print("\nStarting Phase T3 Joint Training Loop (1000 steps)...", flush=True)
    for step in range(start_step + 1, NUM_STEPS + 1):
        step_loss = 0.0
        step_talker_loss = 0.0
        step_mtp_loss = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            sample = train_samples[sampler.next()]
            
            transcript = sample["transcript"]
            codes_16 = torch.tensor(sample["codes_16"], dtype=torch.long, device=device) # [16, num_frames]
            num_frames = codes_16.shape[1]
            
            prompt = f"<|im_start|>user\nVorbește în limba română.<|im_end|>\n<|im_start|>assistant\n{transcript}<|im_end|>\n"
            token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
            thinker_embed = F.embedding(token_ids, embed_weight)
            
            im_start_positions = torch.nonzero(token_ids[0] == 151644).view(-1)
            assistant_im_start = im_start_positions[-1].item()
            assistant_hidden = tp(thinker_embed[:, assistant_im_start:])
            
            assistant_text_hidden = torch.cat(
                (
                    assistant_hidden[:, :3],
                    tts_pad_embed.expand(1, 4, -1),
                    tts_bos_embed.expand(1, -1, -1),
                    assistant_hidden[:, 3:4],
                ),
                dim=1,
            )
            talker_input_embed = assistant_text_hidden + assistant_codec_hidden
            trailing_text_hidden = torch.cat((assistant_hidden[:, 4:], tts_eos_embed.expand(1, -1, -1)), dim=1)
            
            # Codes embedding
            layer0_codes = codes_16[0:1, :]
            layer0_embeds = model_obj.get_input_embeddings()(layer0_codes)
            
            predictor_embeds = model_obj.code_predictor.get_input_embeddings()
            all_layer_embeds_sum = layer0_embeds.clone()
            for j in range(len(predictor_embeds)):
                layer_j_codes = codes_16[j + 1 : j + 2, :]
                emb = predictor_embeds[j](layer_j_codes)
                all_layer_embeds_sum = all_layer_embeds_sum + emb
                
            text_len = trailing_text_hidden.shape[1]
            codec_input_embeds_list = []
            for pos in range(num_frames):
                if pos == 0:
                    continue
                prev_pos = pos - 1
                text_hidden = trailing_text_hidden[:, prev_pos : prev_pos + 1, :] if prev_pos < text_len else tts_pad_embed
                pos_embed = all_layer_embeds_sum[:, prev_pos : prev_pos + 1, :] + text_hidden
                codec_input_embeds_list.append(pos_embed)
                
            last_pos = num_frames - 1
            eos_text_hidden = trailing_text_hidden[:, last_pos : last_pos + 1, :] if last_pos < text_len else tts_pad_embed
            eos_input_embed = all_layer_embeds_sum[:, last_pos : last_pos + 1, :] + eos_text_hidden
            codec_input_embeds_list.append(eos_input_embed)
            
            codec_input_embeds = torch.cat(codec_input_embeds_list, dim=1).to(dtype=torch.bfloat16)
            full_inputs_embeds = torch.cat([talker_input_embed, codec_input_embeds], dim=1)
            
            prefix_len = talker_input_embed.shape[1]
            labels_prefix = torch.full((1, prefix_len - 1), -100, dtype=torch.long, device=device)
            labels_code = layer0_codes
            codec_eos_id = model_obj.config.codec_eos_token_id
            labels_eos = torch.tensor([[codec_eos_id]], device=device, dtype=torch.long)
            labels = torch.cat([labels_prefix, labels_code, labels_eos], dim=1)
            
            attention_mask = torch.ones((1, full_inputs_embeds.shape[1]), device=device, dtype=torch.long)
            
            # Forward pass through Talker
            outputs = talker(
                inputs_embeds=full_inputs_embeds,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 1. Talker loss (stream 0)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            t_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            
            # 2. MTP loss (streams 1..15)
            # Extract codec hidden states from talker
            # outputs.hidden_states is tuple of layer states
            h_states = outputs.hidden_states[0] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
            last_hidden = h_states[-1] # [1, seq_len, 1024]
            codec_hidden = last_hidden[:, prefix_len - 1 : prefix_len - 1 + num_frames, :] # [1, num_frames, 1024]
            
            mtp_total_loss = 0.0
            num_mtp_layers = 15
            hidden_flat = codec_hidden.reshape(-1, 1, 1024)
            layer0_flat = layer0_embeds.reshape(-1, 1, 1024)
            
            # Subsample up to 64 frames per utterance for efficient MTP backprop
            sub_len = min(64, num_frames)
            frame_indices = torch.linspace(0, num_frames - 1, sub_len).long()
            
            hidden_sub = codec_hidden[:, frame_indices, :].reshape(-1, 1, 1024)
            layer0_sub = layer0_embeds[:, frame_indices, :].reshape(-1, 1, 1024)
            codes_sub = codes_16[:, frame_indices]
            
            cp_model = model_obj.code_predictor
            for mtp_layer_idx in range(num_mtp_layers):
                emb_list = [hidden_sub, layer0_sub]
                for prev in range(mtp_layer_idx):
                    p_codes = codes_sub[prev + 1 : prev + 2, :]
                    p_emb = predictor_embeds[prev](p_codes).reshape(-1, 1, 1024)
                    emb_list.append(p_emb)
                mtp_in = torch.cat(emb_list, dim=1).to(dtype=torch.bfloat16)
                tgt_codes = codes_sub[mtp_layer_idx + 1].reshape(-1)
                
                cp_out = cp_model(inputs_embeds=mtp_in, generation_steps=mtp_layer_idx, use_cache=False)
                layer_loss = F.cross_entropy(cp_out.logits[:, -1, :], tgt_codes)
                mtp_total_loss += layer_loss
                
            mtp_loss = mtp_total_loss / num_mtp_layers
            
            joint_loss = (t_loss + mtp_loss) / GRAD_ACCUM_STEPS
            joint_loss.backward()
            
            step_loss += joint_loss.item()
            step_talker_loss += t_loss.item() / GRAD_ACCUM_STEPS
            step_mtp_loss += mtp_loss.item() / GRAD_ACCUM_STEPS
            
        torch.nn.utils.clip_grad_norm_(talker.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        step_losses.append(step_loss)
        
        if step % 20 == 0 or step == 1:
            avg_loss = sum(step_losses[-20:]) / len(step_losses[-20:])
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            elapsed = time.perf_counter() - t_start
            sec_per_step = elapsed / max(1, step - start_step)
            eta_min = (NUM_STEPS - step) * sec_per_step / 60.0
            
            print(f"Step [{step:04d}/{NUM_STEPS}] | Joint Loss: {avg_loss:.4f} (T: {step_talker_loss:.4f}, MTP: {step_mtp_loss:.4f}) | VRAM: {vram_gb:.2f} GB | ETA: {eta_min:.1f}m", flush=True)
            loss_history.append({
                "step": step,
                "joint_loss": round(avg_loss, 4),
                "talker_loss": round(step_talker_loss, 4),
                "mtp_loss": round(step_mtp_loss, 4),
                "vram_gb": round(vram_gb, 2),
            })
            
        if step % SAVE_EVERY == 0 or step == NUM_STEPS:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}")
            talker.save_pretrained(ckpt_path)
            # Also save MTP adapter explicitly
            model_obj.code_predictor.save_pretrained(os.path.join(ckpt_path, "mtp_adapter"))
            save_resume_state(
                resume_path,
                model=talker,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                sampler=sampler,
                completed_step=step,
                run_config=run_config,
                loss_history=loss_history,
                step_losses=step_losses,
            )
            print(f"Saved checkpoint to {ckpt_path}", flush=True)

    print("\nPhase T3 Training completed! Saving final adapter model...", flush=True)
    talker.save_pretrained(output_dir)
    model_obj.code_predictor.save_pretrained(os.path.join(output_dir, "mtp_adapter"))
    
    total_time = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    metrics = {
        "phase": "T3_talker_mtp_joint",
        "evidence_status": "confirmatory rerun",
        "run_name": run_name,
        "steps": NUM_STEPS,
        "training_seed": training_seed,
        "sampler": "deterministic epoch shuffle",
        "talker_trainable_parameters": talker_trainable_count,
        "mtp_trainable_parameters": mtp_trainable_count,
        "trainable_parameter_manifest": trainable_manifest,
        "final_loss": round(sum(step_losses[-20:]) / min(20, len(step_losses)), 4),
        "session_training_time_s": round(total_time, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "loss_history": loss_history,
        "checkpoint_path": output_dir,
        "run_config": run_config,
        "git_commit": git_commit(),
        "environment": environment_record(),
    }
    
    report_path = f"reports/{run_name}_seed{training_seed}_training.json"
    write_json(report_path, metrics)
        
    print(f"Saved metrics to {report_path}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a repaired joint Talker+MTP condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--run-name", default="component_joint_1h_1000")
    parser.add_argument("--data-file", default="reports/mimi_codes_ro1000.jsonl")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--force-fresh", action="store_true")
    args = parser.parse_args()
    os.environ["TRAINING_SEED"] = str(args.seed)
    os.environ["NUM_STEPS"] = str(args.steps)
    os.environ["WARMUP_STEPS"] = str(args.warmup_steps)
    os.environ["RUN_NAME"] = args.run_name
    os.environ["TRAIN_DATA_FILE"] = args.data_file
    os.environ["SAVE_EVERY"] = str(args.save_every)
    if args.force_fresh:
        os.environ["FORCE_FRESH"] = "1"
    train_phase_t3()
