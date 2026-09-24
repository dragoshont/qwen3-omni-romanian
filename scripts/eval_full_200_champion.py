import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import json
import time
import gc
import re
import random
import hashlib
import importlib.metadata
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
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

ASR_MODEL = "openai/whisper-large-v3-turbo"
ASR_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

def set_generation_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prompt_seed(base_seed, prompt_id):
    """Stable, order-independent seed shared across checkpoint candidates."""
    payload = f"t4-seeded-v2:{base_seed}:{prompt_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)

def normalize_romanian_asr(text):
    """Minimal documented normalization for intelligibility scoring."""
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("ş", "ș").replace("ţ", "ț")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())

def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def hash_checkpoint(path):
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    members = {}
    for file_path in files:
        member_hash = hash_file(file_path)
        relative = file_path.relative_to(root).as_posix()
        members[relative] = member_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(member_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), members

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def detect_repetition(text):
    t_clean = text.lower().strip()
    words = t_clean.split()
    if len(words) >= 6:
        for n in [1, 2, 3]:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            for i in range(n, len(ngrams)):
                if ngrams[i] == ngrams[i-n]:
                    if i >= 2*n and ngrams[i] == ngrams[i-2*n]:
                        return True
    if re.search(r'(-[a-z]){5,}', t_clean):
        return True
    if len(words) >= 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return True
    return False

def eval_champion_200(adapter_path="models/T3_talker_mtp/final", report_path=None, output_dir=None, seed=42):
    set_generation_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print(f"FULL 200-SENTENCE HELD-OUT EVALUATION ON MODEL: {adapter_path}", flush=True)
    print(f"Generation seed: {seed}", flush=True)
    print("=" * 70, flush=True)
    
    if report_path is None:
        report_path = sys.argv[2] if len(sys.argv) > 2 else "reports/champion_200_benchmark.json"
    if output_dir is None:
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs/champion_200"
        
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
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
    
    # 3. Load 4-Bit Talker
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
    
    print(f"Attaching Champion Talker LoRA from {adapter_path}...", flush=True)
    talker = PeftModel.from_pretrained(base_talker, adapter_path, is_trainable=False)
    
    mtp_adapter_dir = os.path.join(adapter_path, "mtp_adapter")
    if os.path.exists(mtp_adapter_dir):
        print(f"Attaching Champion MTP LoRA from {mtp_adapter_dir}...", flush=True)
        model_obj = talker.base_model.model if hasattr(talker, "base_model") else talker
        model_obj.code_predictor = PeftModel.from_pretrained(model_obj.code_predictor, mtp_adapter_dir, is_trainable=False)
        
    talker.eval()
    print("Champion model fully loaded.", flush=True)
    
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
    
    # Load 200 held-out sentences
    eval_file = "eval/ro_holdout_200.jsonl"
    eval_samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_samples.append(json.loads(line))
                
    print(f"Loaded exactly {len(eval_samples)} held-out sentences across 10 categories.", flush=True)
    
    results = []
    total_gen_time = 0.0
    total_audio_duration = 0.0
    
    print("\nStarting generation across all 200 held-out sentences...", flush=True)
    for idx, sample in enumerate(eval_samples, 1):
        s_id = sample["id"]
        cat = sample["category"]
        text = sample["text"]
        sample_seed = prompt_seed(seed, s_id)
        set_generation_seed(sample_seed)
        
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
        
        max_tokens = min(380, max(80, len(text.split()) * 20))
        
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
            sequence_ids = res.sequences[0].detach().cpu().tolist()
            eos_token_id = int(model_obj.config.codec_eos_token_id)
            emitted_eos = eos_token_id in sequence_ids
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
        if idx % 10 == 0 or idx == 1:
            print(f"[{idx:03d}/200] {s_id} ({cat}): {audio_dur:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f})", flush=True)
            
        hit_max = not emitted_eos
        
        results.append({
            "id": s_id,
            "category": cat,
            "reference_text": text,
            "wav_path": wav_path,
            "audio_duration_s": round(audio_dur, 2),
            "generation_time_s": round(gen_time, 2),
            "rtf": round(rtf, 2),
            "generation_seed": sample_seed,
            "generated_sequence_length": len(sequence_ids),
            "final_sequence_token_id": sequence_ids[-1] if sequence_ids else None,
            "emitted_codec_eos": emitted_eos,
            "hit_max_tokens": hit_max,
            "eos_success": emitted_eos,
        })

    vram_peak_gen = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\n200-sentence synthesis completed. Peak VRAM: {vram_peak_gen:.2f} GB.", flush=True)

    # Free memory before Whisper
    del talker, base_talker, code2wav, embed_weight, sd_embed
    gc.collect()
    torch.cuda.empty_cache()
    
    # Evaluate with Whisper large-v3-turbo
    print("Loading Whisper large-v3-turbo for 200-sentence evaluation...", flush=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
        revision=ASR_REVISION,
        device=device,
        torch_dtype=torch.float16,
    )
    
    cat_metrics = {}
    total_wer = 0.0
    total_cer = 0.0
    
    print("\nRunning Whisper Transcription across 200 sentences...", flush=True)
    for idx, item in enumerate(results, 1):
        audio_data, sr = sf.read(item["wav_path"])
        if sr != 24000 or not np.isfinite(audio_data).all():
            raise ValueError(f"Invalid generated audio for {item['id']}: sample_rate={sr}")
        audio_dict = {"raw": audio_data, "sampling_rate": sr}
        with torch.no_grad():
            asr_res = asr(audio_dict, generate_kwargs={"language": "ro", "task": "transcribe"})
            transcription = asr_res["text"].strip()
            
        ref = unicodedata.normalize("NFC", item["reference_text"].strip())
        hyp = unicodedata.normalize("NFC", transcription.strip())
        strict_w_err = jiwer.wer(ref.lower(), hyp.lower())
        strict_c_err = jiwer.cer(ref.lower(), hyp.lower())
        normalized_ref = normalize_romanian_asr(ref)
        normalized_hyp = normalize_romanian_asr(hyp)
        w_err = jiwer.wer(normalized_ref, normalized_hyp)
        c_err = jiwer.cer(normalized_ref, normalized_hyp)
        has_rep = detect_repetition(transcription)
        
        item["whisper_transcription"] = transcription
        item["normalized_reference_text"] = normalized_ref
        item["normalized_whisper_transcription"] = normalized_hyp
        item["strict_wer"] = round(strict_w_err, 4)
        item["strict_cer"] = round(strict_c_err, 4)
        item["wer"] = round(w_err, 4)
        item["cer"] = round(c_err, 4)
        item["has_repetition"] = has_rep
        
        cat = item["category"]
        if cat not in cat_metrics:
            cat_metrics[cat] = {"wer": [], "cer": []}
        cat_metrics[cat]["wer"].append(w_err)
        cat_metrics[cat]["cer"].append(c_err)
        
        total_wer += w_err
        total_cer += c_err
        
        if idx % 20 == 0:
            print(f"[{idx:03d}/200] Progress: Running Mean WER: {total_wer/idx*100:.1f}%, CER: {total_cer/idx*100:.1f}%", flush=True)
            
    cers = [x["cer"] for x in results]
    wers = [x["wer"] for x in results]
    n = len(results)
    repetition_count = sum(1 for x in results if x.get("has_repetition", False))
    eos_count = sum(1 for x in results if x.get("eos_success", True))
    max_hit_count = sum(1 for x in results if x.get("hit_max_tokens", False))
    
    # Category summary
    category_summary = {}
    for cat, vals in cat_metrics.items():
        cat_cers = vals["cer"]
        cat_wers = vals["wer"]
        category_summary[cat] = {
            "mean_wer": round(float(np.mean(cat_wers)), 4),
            "median_wer": round(float(np.median(cat_wers)), 4),
            "mean_cer": round(float(np.mean(cat_cers)), 4),
            "median_cer": round(float(np.median(cat_cers)), 4),
            "p90_cer": round(float(np.percentile(cat_cers, 90)), 4),
            "count": len(cat_wers),
        }
        
    checkpoint_sha256, checkpoint_members = hash_checkpoint(adapter_path)
    holdout_sha256 = hash_file(eval_file)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    final_report = {
        "evaluation_name": "Full 200 Held-Out Romanian Evaluation",
        "champion_model": adapter_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_member_sha256": checkpoint_members,
        "holdout_sha256": holdout_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "total_samples": n,
        "mean_wer": round(float(np.mean(wers)), 4),
        "median_wer": round(float(np.median(wers)), 4),
        "p90_wer": round(float(np.percentile(wers, 90)), 4),
        "mean_cer": round(float(np.mean(cers)), 4),
        "median_cer": round(float(np.median(cers)), 4),
        "p90_cer": round(float(np.percentile(cers, 90)), 4),
        "eos_rate": round(float(eos_count / n * 100), 2),
        "repetition_rate": round(float(repetition_count / n * 100), 2),
        "max_token_hit_rate": round(float(max_hit_count / n * 100), 2),
        "overall_rtf": round(total_gen_time / total_audio_duration, 2) if total_audio_duration > 0 else 0,
        "total_audio_duration_s": round(total_audio_duration, 2),
        "peak_vram_gb": round(vram_peak_gen, 2),
        "evaluation_config": {
            "seed": seed,
            "per_prompt_seed_scheme": "sha256('t4-seeded-v2:{base_seed}:{prompt_id}') first 64 bits modulo 2^63-1",
            "deterministic_algorithms": "warn_only",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "allow_tf32": False,
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "max_tokens_policy": "min(380, max(80, word_count * 20))",
            "primary_scoring": "NFC, lowercase, Romanian cedilla-to-comma normalization, Unicode punctuation removal, whitespace collapse",
            "strict_scoring": "NFC and lowercase only",
            "asr_model": ASR_MODEL,
            "asr_revision": ASR_REVISION,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "jiwer": package_version("jiwer"),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "category_breakdown": category_summary,
        "detailed_results": results,
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    print("=" * 70, flush=True)
    print(f"CHAMPION 200-SENTENCE EVALUATION COMPLETE!")
    print(f"Overall Mean CER: {final_report['mean_cer']*100:.2f}% | Median CER: {final_report['median_cer']*100:.2f}%")
    print(f"Overall Mean WER: {final_report['mean_wer']*100:.2f}% | Median WER: {final_report['median_wer']*100:.2f}%")
    print(f"EOS Rate: {final_report['eos_rate']}% | Repetition: {final_report['repetition_rate']}%")
    print(f"Saved report to {report_path}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    adapter = sys.argv[1] if len(sys.argv) > 1 else "models/T3_talker_mtp/final"
    report = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    eval_champion_200(adapter, report, out_dir, seed)
