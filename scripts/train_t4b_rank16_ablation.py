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

def train_extended():
    if not os.path.exists("models/GATE_AUTHORIZED.json"):
        print("=" * 70, flush=True)
        print("[HARD SCIENTIFIC GATE BLOCKED] Scaling curriculum is paused.", flush=True)
        print("Awaiting controlled 40-sentence matrix evaluation, T2 vs T3 comparison, and gate authorization.", flush=True)
        print("=" * 70, flush=True)
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("SCALING CURRICULUM: Deep Extended Romanian Training", flush=True)
    print("=" * 70, flush=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    output_dir = "models/deep_extended_curriculum/final"
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
    
    # Configure Joint LoRA on Talker and MTP
    print("3. Configuring LoRA...", flush=True)
    mtp_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    talker.code_predictor = get_peft_model(talker.code_predictor, mtp_lora_config)
    
    talker = prepare_model_for_kbit_training(talker, use_gradient_checkpointing=True)
    talker_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    
    # Load Training Data
    data_file = "reports/mimi_codes_ro1000.jsonl"
    train_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                codes = item.get("codes_16", [])
                if len(codes) > 0 and 15 <= len(codes[0]) <= 250:
                    train_samples.append(item)
                    
    print(f"Loaded {len(train_samples)} training samples for extended curriculum.", flush=True)
    
    NUM_STEPS = 2500
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE_TALKER = 2e-5
    LEARNING_RATE_MTP = 4e-5
    WARMUP_STEPS = 100
    
    talker_params = [p for n, p in talker.named_parameters() if p.requires_grad and "code_predictor" not in n]
    mtp_params = [p for n, p in talker.named_parameters() if p.requires_grad and "code_predictor" in n]
    
    optimizer = torch.optim.AdamW([
        {"params": talker_params, "lr": LEARNING_RATE_TALKER},
        {"params": mtp_params, "lr": LEARNING_RATE_MTP},
    ], betas=(0.9, 0.95), weight_decay=0.01)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_STEPS,
    )
    
    talker.train()
    optimizer.zero_grad()
    
    step_losses = []
    t_start = time.perf_counter()
    sample_idx = 0
    total_samples = len(train_samples)
    
    print(f"\nStarting Extended Curriculum Loop ({NUM_STEPS} steps)...", flush=True)
    for step in range(1, NUM_STEPS + 1):
        step_loss = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            sample = train_samples[sample_idx % total_samples]
            sample_idx += 1
            
            transcript = sample["transcript"]
            codes_16 = torch.tensor(sample["codes_16"], dtype=torch.long, device=device)
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
            
            outputs = talker(
                inputs_embeds=full_inputs_embeds,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                output_hidden_states=True,
                return_dict=True,
            )
            
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            t_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            
            h_states = outputs.hidden_states[0] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
            last_hidden = h_states[-1]
            codec_hidden = last_hidden[:, prefix_len - 1 : prefix_len - 1 + num_frames, :]
            
            mtp_total_loss = 0.0
            num_mtp_layers = 15
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
            
        torch.nn.utils.clip_grad_norm_(talker.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        step_losses.append(step_loss)
        
        if step % 50 == 0 or step == 1:
            avg_loss = sum(step_losses[-50:]) / len(step_losses[-50:])
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            elapsed = time.perf_counter() - t_start
            sec_per_step = elapsed / step
            eta_min = (NUM_STEPS - step) * sec_per_step / 60.0
            print(f"Step [{step:04d}/{NUM_STEPS}] | Extended Loss: {avg_loss:.4f} | VRAM: {vram_gb:.2f} GB | ETA: {eta_min:.1f}m", flush=True)
            
        if step % 500 == 0 or step == NUM_STEPS:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}")
            talker.save_pretrained(ckpt_path)
            model_obj.code_predictor.save_pretrained(os.path.join(ckpt_path, "mtp_adapter"))
            print(f"Checkpoint saved: {ckpt_path}", flush=True)

    print("Extended training complete! Saving final model...", flush=True)
    talker.save_pretrained(output_dir)
    model_obj.code_predictor.save_pretrained(os.path.join(output_dir, "mtp_adapter"))
    print("=" * 70, flush=True)

if __name__ == "__main__":
    train_extended()
