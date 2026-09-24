import os
import sys
import json
import time
import gc
import torch
import soundfile as sf
import safetensors.torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeConfig,
    pipeline,
)
from peft import PeftModel
import jiwer

sys.stdout.reconfigure(encoding="utf-8")

def eval_phase_t2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("EVALUATING PHASE T2: Romanian Talker LoRA + Stock MTP (Held-out Quick 40)", flush=True)
    print("=" * 70, flush=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    output_dir = "outputs/T2_talker_only"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load tokenizer & Embeddings
    print("1. Loading Tokenizer and Thinker Embeddings...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    sd_embed = safetensors.torch.load_file("models/thinker_embed_tokens.safetensors")
    embed_weight = sd_embed["thinker.model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    
    # 2. Load Code2Wav
    print("2. Loading Code2Wav...", flush=True)
    full_config = Qwen3OmniMoeConfig.from_pretrained("models/qwen3-omni-partial")
    code2wav = Qwen3OmniMoeCode2Wav(full_config.code2wav_config).to(device=device, dtype=torch.bfloat16)
    c2w_sd = safetensors.torch.load_file("models/code2wav.safetensors")
    code2wav.load_state_dict(c2w_sd, strict=True)
    code2wav.eval()
    
    # 3. Load 4-bit Talker and attach T2 adapter
    print("3. Loading 4-Bit Talker Backbone...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_talker = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        "models/qwen3-omni-talker",
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    
    t2_adapter_path = "models/T2_talker_only/final"
    print(f"Attaching Romanian Talker LoRA from {t2_adapter_path}...", flush=True)
    talker = PeftModel.from_pretrained(
        base_talker,
        t2_adapter_path,
        is_trainable=False,
    )
    talker.eval()
    print("Talker + T2 LoRA successfully assembled.", flush=True)
    
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
    
    # 4. Load held-out quick 40 evaluation set
    eval_file = "eval/ro_holdout_quick_40.jsonl"
    eval_samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_samples.append(json.loads(line))
                
    print(f"Loaded {len(eval_samples)} test samples from {eval_file}.", flush=True)
    
    results = []
    total_gen_time = 0.0
    total_audio_duration = 0.0
    
    print("\nStarting generation across all 40 held-out sentences with Romanian Talker...", flush=True)
    for idx, sample in enumerate(eval_samples, 1):
        s_id = sample["id"]
        cat = sample["category"]
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
        
        max_tokens = min(350, max(80, len(text.split()) * 20))
        
        t0 = time.perf_counter()
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
                
        gen_time = time.perf_counter() - t0
        audio_dur = len(wav_np) / 24000.0
        total_gen_time += gen_time
        total_audio_duration += audio_dur
        
        wav_path = os.path.join(output_dir, f"{s_id}.wav")
        sf.write(wav_path, wav_np, 24000)
        
        rtf = gen_time / audio_dur if audio_dur > 0 else 0
        print(f"[{idx:02d}/40] {s_id} ({cat}): {audio_dur:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f})", flush=True)
        
        results.append({
            "id": s_id,
            "category": cat,
            "reference_text": text,
            "wav_path": wav_path,
            "audio_duration_s": round(audio_dur, 2),
            "generation_time_s": round(gen_time, 2),
            "rtf": round(rtf, 2),
        })

    vram_peak_gen = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\nGeneration completed. Peak VRAM: {vram_peak_gen:.2f} GB.", flush=True)

    # 5. Free generation memory before ASR
    del talker, base_talker, code2wav, embed_weight, sd_embed
    gc.collect()
    torch.cuda.empty_cache()
    
    # 6. Evaluate with Whisper large-v3-turbo
    print("Loading Whisper large-v3-turbo pipeline...", flush=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        torch_dtype=torch.float16,
    )
    
    total_wer = 0.0
    total_cer = 0.0
    print("\nRunning Whisper Transcription & Error Rate Calculation...", flush=True)
    for idx, item in enumerate(results, 1):
        audio_data, sr = sf.read(item["wav_path"])
        audio_dict = {"raw": audio_data, "sampling_rate": sr}
        with torch.no_grad():
            asr_res = asr(audio_dict, generate_kwargs={"language": "ro", "task": "transcribe"})
            transcription = asr_res["text"].strip()
            
        ref = item["reference_text"].strip()
        w_err = jiwer.wer(ref.lower(), transcription.lower())
        c_err = jiwer.cer(ref.lower(), transcription.lower())
        
        item["whisper_transcription"] = transcription
        item["wer"] = round(w_err, 4)
        item["cer"] = round(c_err, 4)
        
        total_wer += w_err
        total_cer += c_err
        
        print(f"[{idx:02d}/40] {item['id']}:", flush=True)
        print(f"   REF: {ref}", flush=True)
        print(f"   HYP: {transcription}", flush=True)
        print(f"   WER: {w_err*100:.1f}%, CER: {c_err*100:.1f}%\n", flush=True)
        
    avg_wer = total_wer / len(results)
    avg_cer = total_cer / len(results)
    
    benchmark_report = {
        "phase": "T2_talker_only",
        "model_configuration": "Romanian 4-Bit Talker LoRA + Stock MTP + Stock Code2Wav",
        "held_out_dataset": eval_file,
        "sample_count": len(results),
        "mean_wer": round(avg_wer, 4),
        "mean_cer": round(avg_cer, 4),
        "total_audio_duration_s": round(total_audio_duration, 2),
        "total_generation_time_s": round(total_gen_time, 2),
        "overall_rtf": round(total_gen_time / total_audio_duration, 2) if total_audio_duration > 0 else 0,
        "peak_vram_gb": round(vram_peak_gen, 2),
        "provenance_level": 3,
        "teacher_forcing": False,
        "detailed_results": results,
    }
    
    report_path = "reports/T2_talker_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
        
    print("=" * 70, flush=True)
    print(f"PHASE T2 EVALUATION COMPLETE!", flush=True)
    print(f"Mean WER: {avg_wer*100:.2f}% | Mean CER: {avg_cer*100:.2f}%", flush=True)
    print(f"Saved benchmark report to {report_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    eval_phase_t2()
