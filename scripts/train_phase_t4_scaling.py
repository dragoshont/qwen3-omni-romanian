import os
import sys
import json
import time
import gc
import math
import random
import torch
import torch.nn.functional as F
import safetensors.torch
import soundfile as sf
import numpy as np
import re
import jiwer
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeConfig,
    get_cosine_schedule_with_warmup,
    pipeline,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.stdout.reconfigure(encoding="utf-8")

def detect_repetition(text):
    if not text:
        return False
    t_clean = text.lower().strip()
    words = re.findall(r'\b\w+\b', t_clean)
    if len(words) >= 4:
        repeat_count = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeat_count += 1
                if repeat_count >= 4:
                    return True
            else:
                repeat_count = 1
        if len(words) >= 6:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            for i in range(2, len(bigrams)):
                if bigrams[i] == bigrams[i-2]:
                    if i >= 4 and bigrams[i] == bigrams[i-4]:
                        return True
        if len(words) >= 9:
            trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            for i in range(3, len(trigrams)):
                if trigrams[i] == trigrams[i-3]:
                    if i >= 6 and trigrams[i] == trigrams[i-6]:
                        return True
    if re.search(r'(-[a-z]){5,}', t_clean):
        return True
    if len(words) >= 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return True
    return False

def evaluate_quick_40(talker, tokenizer, embed_weight, code2wav, step, output_dir, device):
    print(f"\n--- [CHECKPOINT EVALUATION @ STEP {step}] Running Quick-40 Held-Out Benchmark ---", flush=True)
    eval_file = "eval/ro_holdout_quick_40.jsonl"
    eval_samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_samples.append(json.loads(line))
                
    model_obj = talker.base_model.model if hasattr(talker, "base_model") else talker
    tp = model_obj.text_projection
    
    tts_special_tokens = torch.tensor([[151672, 151673, 151671]], device=device, dtype=torch.long)
    tts_special_embed = F.embedding(tts_special_tokens, embed_weight)
    tts_projected = tp(tts_special_embed)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_projected.chunk(3, dim=1)
    
    speaker_id = 2302
    codec_special_tokens = torch.tensor([[2155, 2156, 2157, speaker_id, 2148, 2149]], device=device, dtype=torch.long)
    codec_special_embeds = model_obj.get_input_embeddings()(codec_special_tokens)
    assistant_codec_hidden = torch.cat(
        (torch.zeros((1, 3, 1024), device=device, dtype=torch.bfloat16), codec_special_embeds), dim=1
    )
    
    results = []
    eval_wav_dir = os.path.join(output_dir, f"eval_audio_step_{step}")
    os.makedirs(eval_wav_dir, exist_ok=True)
    
    talker.eval()
    t_start_eval = time.time()
    
    for idx, sample in enumerate(eval_samples, 1):
        s_id = sample["id"]
        text = sample["text"]
        prompt = f"<|im_start|>user\nVorbește în limba română.<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>\n"
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
        inputs_embeds = assistant_text_hidden + assistant_codec_hidden
        talker_input_ids = torch.full((1, inputs_embeds.shape[1]), fill_value=151671, dtype=torch.long, device=device)
        trailing_text_hidden = torch.cat((assistant_hidden[:, 4:], tts_eos_embed.expand(1, -1, -1)), dim=1)
        attention_mask = torch.ones((1, inputs_embeds.shape[1]), device=device, dtype=torch.long)
        
        words = len(text.split())
        max_tokens = min(350, max(80, words * 20))
        max_dur_approx = max_tokens * 0.08
        
        torch.manual_seed(42 + idx)
        with torch.no_grad():
            res = talker.generate(
                inputs_embeds=inputs_embeds,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                talker_input_ids=talker_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.15,
                eos_token_id=model_obj.config.codec_eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            code_steps = [hid[-1] for hid in res.hidden_states if hid[-1] is not None]
            if len(code_steps) > 0:
                codes = torch.stack(code_steps, dim=1).transpose(1, 2)
                wav = code2wav(codes)
                wav_np = wav.squeeze().cpu().float().numpy()
            else:
                wav_np = np.zeros(24000, dtype=np.float32)
                
        dur = len(wav_np) / 24000.0
        wav_path = os.path.join(eval_wav_dir, f"{s_id}.wav")
        sf.write(wav_path, wav_np, 24000)
        
        hit_max = dur >= (max_dur_approx - 0.25)
        results.append({
            "id": s_id,
            "category": sample["category"],
            "reference_text": text,
            "wav_path": wav_path,
            "audio_duration_s": round(dur, 2),
            "hit_max_tokens": hit_max,
            "eos_success": not hit_max
        })
        
    # Free memory and transcribe with Whisper
    torch.cuda.empty_cache()
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        torch_dtype=torch.float16,
    )
    
    cers, wers = [], []
    repetition_count = 0
    for item in results:
        audio_data, sr = sf.read(item["wav_path"])
        with torch.no_grad():
            asr_res = asr({"raw": audio_data, "sampling_rate": sr}, generate_kwargs={"language": "ro", "task": "transcribe"})
            hyp = asr_res["text"].strip()
            
        ref = item["reference_text"].strip()
        c_err = jiwer.cer(ref.lower(), hyp.lower())
        w_err = jiwer.wer(ref.lower(), hyp.lower())
        has_rep = detect_repetition(hyp)
        if has_rep:
            repetition_count += 1
            
        item["whisper_transcription"] = hyp
        item["cer"] = round(c_err, 4)
        item["wer"] = round(w_err, 4)
        item["has_repetition"] = has_rep
        cers.append(c_err)
        wers.append(w_err)
        
    del asr
    torch.cuda.empty_cache()
    talker.train()
    
    n = len(results)
    eval_summary = {
        "step": step,
        "mean_cer": round(float(np.mean(cers)), 4),
        "median_cer": round(float(np.median(cers)), 4),
        "mean_wer": round(float(np.mean(wers)), 4),
        "median_wer": round(float(np.median(wers)), 4),
        "eos_rate": round(float(sum(1 for x in results if x["eos_success"]) / n * 100), 2),
        "max_token_hit_rate": round(float(sum(1 for x in results if x["hit_max_tokens"]) / n * 100), 2),
        "repetition_rate": round(float(repetition_count / n * 100), 2),
        "mean_duration_s": round(float(np.mean([x["audio_duration_s"] for x in results])), 2),
        "eval_time_s": round(time.time() - t_start_eval, 1),
        "detailed_results": results
    }
    
    print(f"[@STEP {step}] Mean CER: {eval_summary['mean_cer']*100:.2f}% | Median CER: {eval_summary['median_cer']*100:.2f}% | Mean WER: {eval_summary['mean_wer']*100:.2f}% | EOS: {eval_summary['eos_rate']}% | Repetition: {eval_summary['repetition_rate']}%", flush=True)
    return eval_summary

def train_phase_t4():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("PHASE T4: FRESH DATA-SCALE EXPERIMENT (5 Hours, Stock Base + Fresh LoRA)", flush=True)
    print("Architecture: Talker LoRA (r=8, alpha=16, q/v_proj) + MTP LoRA (r=8, alpha=16, all 6 targets)")
    print("Thinker & Code2Wav: 100% Frozen | Talker: 4-Bit NF4 Quantized")
    print("=" * 70, flush=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    output_dir = "models/T4_data_scale_5h/final"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Tokenizer & Embeddings
    print("1. Loading Tokenizer and Thinker Embeddings...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    sd_embed = safetensors.torch.load_file("models/thinker_embed_tokens.safetensors")
    embed_weight = sd_embed["thinker.model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    
    # 2. Load Frozen Code2Wav
    print("2. Loading Code2Wav for Checkpoint Evaluations...", flush=True)
    full_config = Qwen3OmniMoeConfig.from_pretrained("models/qwen3-omni-partial")
    code2wav = Qwen3OmniMoeCode2Wav(full_config.code2wav_config).to(device=device, dtype=torch.bfloat16)
    c2w_sd = safetensors.torch.load_file("models/code2wav.safetensors")
    code2wav.load_state_dict(c2w_sd, strict=True)
    code2wav.eval()
    
    # 3. Load 4-Bit Talker Backbone (STOCK BASE)
    print("3. Loading 4-Bit Talker Backbone (Stock Base)...", flush=True)
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
    
    # 4. Attach FRESH LoRA Adapters (Strictly Preserving Verified T3 Architecture)
    print("4. Attaching FRESH LoRA Adapters (Preserving T3 Architecture)...", flush=True)
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
    
    talker = prepare_model_for_kbit_training(talker, use_gradient_checkpointing=True)
    talker_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    talker = get_peft_model(talker, talker_lora_config)
    print("FRESH LoRA Adapters attached. Zero weights copied from T3.")
    talker.print_trainable_parameters()
    
    # Precompute special token embeds
    tts_special_tokens = torch.tensor([[151672, 151673, 151671]], device=device, dtype=torch.long)
    tts_special_embed = F.embedding(tts_special_tokens, embed_weight)
    tp = talker.base_model.model.text_projection if hasattr(talker, "base_model") else talker.text_projection
    tts_projected = tp(tts_special_embed)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_projected.chunk(3, dim=1)
    
    speaker_id = 2302
    codec_special_tokens = torch.tensor([[2155, 2156, 2157, speaker_id, 2148, 2149]], device=device, dtype=torch.long)
    model_obj = talker.base_model.model if hasattr(talker, "base_model") else talker
    codec_special_embeds = model_obj.get_input_embeddings()(codec_special_tokens)
    assistant_codec_hidden = torch.cat(
        (torch.zeros((1, 3, 1024), device=device, dtype=torch.bfloat16), codec_special_embeds), dim=1
    )
    
    # 5. Load 5-Hour Dataset
    data_file = "reports/mimi_codes_ro_5h.jsonl"
    print(f"Loading 5-hour Romanian training dataset from {data_file}...", flush=True)
    train_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_samples.append(json.loads(line))
                
    num_samples = len(train_samples)
    print(f"Loaded {num_samples} training utterances from 5-hour dataset.", flush=True)
    
    # Hyperparameters
    NUM_STEPS = 2500
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE_TALKER = 2e-5
    LEARNING_RATE_MTP = 4e-5
    WARMUP_STEPS = 100
    
    total_presentations = NUM_STEPS * GRAD_ACCUM_STEPS # 10,000
    effective_epochs = total_presentations / num_samples
    print(f"Training Plan: {NUM_STEPS} optimizer steps with grad_accum={GRAD_ACCUM_STEPS} ({total_presentations} presentations).")
    print(f"Effective Epochs: {effective_epochs:.2f} passes over the 5-hour dataset.")
    
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
    
    eval_checkpoints = [500, 1000, 1500, 2000, 2500]
    checkpoint_eval_history = []
    loss_history = []
    step_losses = []
    
    t_start = time.perf_counter()
    talker.train()
    optimizer.zero_grad()
    
    sample_idx = 0
    total_samples = len(train_samples)
    
    num_mtp_layers = 15
    consecutive_regressions = 0
    best_eval_cer = float("inf")
    best_eval_checkpoint = 2500
    
    print("\nStarting T4 Training with Scheduled Quick-40 Checkpoint Evaluations...\n", flush=True)
    
    for step in range(1, NUM_STEPS + 1):
        step_loss = 0.0
        step_talker_loss = 0.0
        step_mtp_loss = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            sample = train_samples[sample_idx % total_samples]
            sample_idx += 1
            
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
            h_states = outputs.hidden_states[0] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
            last_hidden = h_states[-1]
            codec_hidden = last_hidden[:, prefix_len - 1 : prefix_len - 1 + num_frames, :]
            
            mtp_total_loss = 0.0
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
            sec_per_step = elapsed / step
            eta_min = (NUM_STEPS - step) * sec_per_step / 60.0
            
            print(f"Step [{step:04d}/{NUM_STEPS}] | Joint Loss: {avg_loss:.4f} (T: {step_talker_loss:.4f}, MTP: {step_mtp_loss:.4f}) | VRAM: {vram_gb:.2f} GB | ETA: {eta_min:.1f}m", flush=True)
            loss_history.append({
                "step": step,
                "joint_loss": round(avg_loss, 4),
                "talker_loss": round(step_talker_loss, 4),
                "mtp_loss": round(step_mtp_loss, 4),
                "vram_gb": round(vram_gb, 2),
            })
            
        # Checkpoint save and evaluation
        if step in eval_checkpoints:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}")
            talker.save_pretrained(ckpt_path)
            model_obj.code_predictor.save_pretrained(os.path.join(ckpt_path, "mtp_adapter"))
            print(f"Saved checkpoint to {ckpt_path}", flush=True)
            
            # Run quick-40 evaluation
            eval_metrics = evaluate_quick_40(talker, tokenizer, embed_weight, code2wav, step, output_dir, device)
            checkpoint_eval_history.append(eval_metrics)
            
            with open("reports/t4_checkpoint_eval_history.json", "w", encoding="utf-8") as f:
                json.dump(checkpoint_eval_history, f, indent=2, ensure_ascii=False)
                
            cur_cer = eval_metrics["mean_cer"]
            if cur_cer < best_eval_cer:
                best_eval_cer = cur_cer
                best_eval_checkpoint = step
                consecutive_regressions = 0
                print(f"*** New best checkpoint at step {step} (Mean CER: {best_eval_cer*100:.2f}%) ***", flush=True)
            else:
                rel_degradation = (cur_cer - best_eval_cer) / best_eval_cer
                if rel_degradation > 0.20:
                    consecutive_regressions += 1
                    print(f"[REGRESSION WARNING] Step {step} CER is {rel_degradation*100:.1f}% worse than best ({best_eval_cer*100:.2f}%). Consecutive regressions: {consecutive_regressions}.", flush=True)
                else:
                    consecutive_regressions = 0
                    
            if consecutive_regressions >= 2:
                print(f"\n[EARLY STOPPING TRIGGERED] Two consecutive clear held-out regressions observed. Halting training at step {step}.", flush=True)
                break
            if eval_metrics["repetition_rate"] > 10.0:
                print(f"\n[EARLY STOPPING TRIGGERED] Catastrophic repetition behavior detected ({eval_metrics['repetition_rate']}%). Halting training at step {step}.", flush=True)
                break
                
    # Final save
    print("\nPhase T4 Training Complete! Saving final model adapter...", flush=True)
    talker.save_pretrained(output_dir)
    model_obj.code_predictor.save_pretrained(os.path.join(output_dir, "mtp_adapter"))
    
    total_time = time.perf_counter() - t_start
    final_report = {
        "curriculum_phase": "T4_data_scale_5h",
        "dataset": data_file,
        "sample_count": num_samples,
        "effective_epochs": round(effective_epochs, 2),
        "total_training_time_s": round(total_time, 2),
        "total_training_time_min": round(total_time / 60, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        "loss_history": loss_history,
        "checkpoint_eval_history": checkpoint_eval_history,
        "best_checkpoint_step": best_eval_checkpoint,
        "best_checkpoint_cer": best_eval_cer
    }
    
    with open("reports/t4_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    print(f"Saved T4 training summary to reports/t4_training_summary.json", flush=True)
    print(f"Best checkpoint identified: Step {best_eval_checkpoint} (CER: {best_eval_cer*100:.2f}%)", flush=True)
    return best_eval_checkpoint

if __name__ == "__main__":
    train_phase_t4()
