import os
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

sys.stdout.reconfigure(encoding="utf-8")

def train_phase_t2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("PHASE T2: Training Romanian Talker-Only LoRA (1h Romanian Data)", flush=True)
    print("=" * 70, flush=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    output_dir = "models/T2_talker_only/final"
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
    
    # Freeze Code Predictor (MTP) strictly
    for param in talker.code_predictor.parameters():
        param.requires_grad = False
    talker.code_predictor.eval()
    print("MTP Code Predictor frozen strictly.", flush=True)
    
    # 3. Configure LoRA on Talker
    print("3. Configuring LoRA on Talker...", flush=True)
    talker = prepare_model_for_kbit_training(talker, use_gradient_checkpointing=True)
    
    # Target q_proj and v_proj across talker attention layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    talker = get_peft_model(talker, lora_config)
    talker.print_trainable_parameters()
    
    # Precompute special token embeds
    tts_special_tokens = torch.tensor([[151672, 151673, 151671]], device=device, dtype=torch.long)
    tts_special_embed = F.embedding(tts_special_tokens, embed_weight)
    
    # Use base_layer if text_projection wrapped
    tp = talker.base_model.model.text_projection if hasattr(talker, "base_model") else talker.text_projection
    tts_projected = tp(tts_special_embed)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_projected.chunk(3, dim=1)
    
    speaker_id = 2302 # Ethan
    codec_special_tokens = torch.tensor([[2155, 2156, 2157, speaker_id, 2148, 2149]], device=device, dtype=torch.long)
    
    # Codec embeddings
    model_obj = talker.base_model.model if hasattr(talker, "base_model") else talker
    codec_special_embeds = model_obj.get_input_embeddings()(codec_special_tokens)
    assistant_codec_hidden = torch.cat(
        (torch.zeros((1, 3, 1024), device=device, dtype=torch.bfloat16), codec_special_embeds), dim=1
    )
    
    # 4. Load Training Data (1000 clips with pre-extracted codes)
    data_file = "reports/mimi_codes_ro1000.jsonl"
    print(f"4. Loading training clips from {data_file}...", flush=True)
    train_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Keep utterances between 15 and 250 codec frames for stable training
                codes = item.get("codes_16", [])
                if len(codes) > 0 and 15 <= len(codes[0]) <= 250:
                    train_samples.append(item)
                    
    print(f"Loaded {len(train_samples)} valid training samples.", flush=True)
    
    # 5. Training Hyperparameters
    NUM_STEPS = 1000
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 50
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, talker.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_STEPS,
    )
    
    talker.train()
    optimizer.zero_grad()
    
    step_losses = []
    loss_history = []
    t_start = time.perf_counter()
    sample_idx = 0
    total_samples = len(train_samples)
    
    print("\nStarting Phase T2 Training Loop (1000 steps)...", flush=True)
    for step in range(1, NUM_STEPS + 1):
        step_loss = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            sample = train_samples[sample_idx % total_samples]
            sample_idx += 1
            
            transcript = sample["transcript"]
            codes_16 = torch.tensor(sample["codes_16"], dtype=torch.long, device=device) # [16, num_frames]
            num_frames = codes_16.shape[1]
            
            # Format text prompt
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
            
            # Build codec input sequence
            layer0_codes = codes_16[0:1, :] # [1, num_frames]
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
            
            # Construct labels for causal language modeling of stream 0
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
                output_hidden_states=False,
                return_dict=True,
            )
            
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            step_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(talker.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        step_losses.append(step_loss)
        
        if step % 20 == 0 or step == 1:
            avg_loss = sum(step_losses[-20:]) / len(step_losses[-20:])
            current_lr = lr_scheduler.get_last_lr()[0]
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            elapsed = time.perf_counter() - t_start
            sec_per_step = elapsed / step
            eta_min = (NUM_STEPS - step) * sec_per_step / 60.0
            
            print(f"Step [{step:04d}/{NUM_STEPS}] | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | VRAM: {vram_gb:.2f} GB | ETA: {eta_min:.1f}m", flush=True)
            loss_history.append({
                "step": step,
                "loss": round(avg_loss, 4),
                "learning_rate": current_lr,
                "vram_gb": round(vram_gb, 2),
            })
            
        if step % 250 == 0 or step == NUM_STEPS:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}")
            talker.save_pretrained(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}", flush=True)

    print("\nTraining completed! Saving final adapter model...", flush=True)
    talker.save_pretrained(output_dir)
    
    total_time = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    metrics = {
        "phase": "T2_talker_only",
        "steps": NUM_STEPS,
        "initial_loss": round(step_losses[0] * GRAD_ACCUM_STEPS, 4),
        "final_loss": round(sum(step_losses[-20:]) / 20, 4),
        "total_training_time_s": round(total_time, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "loss_history": loss_history,
        "checkpoint_path": output_dir,
    }
    
    with open("reports/T2_talker_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved metrics to reports/T2_talker_training_metrics.json")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    train_phase_t2()
