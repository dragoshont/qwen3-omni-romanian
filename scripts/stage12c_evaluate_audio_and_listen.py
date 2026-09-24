import gc
import json
import os
import sys
import time
import torch
import torch.nn as nn
import soundfile as sf
import safetensors.torch
from peft import PeftModel
from transformers import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
    Qwen3OmniMoeCode2Wav
)

sys.stdout.reconfigure(encoding="utf-8")

def evaluate_and_synthesize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print("STAGE 12C: Audio Synthesis & Speech Evaluation with Trained ro500 Model")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")
    
    out_dir = "outputs/romanian_mtp_trained_eval"
    os.makedirs(out_dir, exist_ok=True)
    
    config_dir = "models/qwen3-omni-partial"
    full_config = Qwen3OmniMoeConfig.from_pretrained(config_dir)
    
    # 1. Load Code2Wav Synthesizer
    print("Loading Qwen Code2Wav Synthesizer (216M parameters) strictly in BF16...")
    code2wav = Qwen3OmniMoeCode2Wav(full_config.code2wav_config).to(device=device, dtype=torch.bfloat16)
    c2w_sd = safetensors.torch.load_file(os.path.join(config_dir, "model-00015-of-00015.safetensors"))
    c2w_weights = {k[len("code2wav."):]: v for k, v in c2w_sd.items() if k.startswith("code2wav.")}
    code2wav.load_state_dict(c2w_weights, strict=True)
    code2wav.eval()
    del c2w_sd, c2w_weights
    
    # 2. Load Base MTP + Trained ro500 LoRA Checkpoint
    print("Loading Trained Romanian MTP Model from models/romanian_mtp_lora_ro500/final...")
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
    
    peft_model = PeftModel.from_pretrained(cp_model, "models/romanian_mtp_lora_ro500/final").to(device)
    peft_model.eval()
    predictor_embeds = peft_model.model.model.codec_embedding
    
    # 3. Select 5 diverse test utterances from ro500
    test_samples = []
    with open("reports/mimi_codes_ro500.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in [0, 1, 2, 50, 150]:
                test_samples.append(json.loads(line))
                
    generated_audio_records = []
    print(f"Generating and synthesizing {len(test_samples)} Romanian speech utterances...")
    
    for s in test_samples:
        sample_id = s["sample_id"]
        text = s.get("transcript", "")
        true_codes = torch.tensor(s["codes_16"], dtype=torch.long, device=device) # [16, T]
        T = true_codes.shape[1]
        
        # Conditioning: true Layer 0 token stream
        layer0_codes = true_codes[0]
        layer0_embeds = layer0_embedding(layer0_codes)
        
        predicted_codes_list = [layer0_codes.unsqueeze(0)] # Channel 0
        hidden_flat = layer0_embeds.unsqueeze(1)
        layer0_flat = layer0_embeds.unsqueeze(1)
        
        t_gen_start = time.time()
        with torch.no_grad():
            # Autoregressively predict depth channels 1..15
            for depth in range(15):
                embed_list = [hidden_flat, layer0_flat]
                for prev_d in range(depth):
                    prev_c = predicted_codes_list[prev_d + 1].squeeze(0)
                    prev_emb = predictor_embeds[prev_d](prev_c).unsqueeze(1)
                    embed_list.append(prev_emb)
                    
                mtp_inputs = torch.cat(embed_list, dim=1)
                outputs = peft_model(
                    inputs_embeds=mtp_inputs,
                    generation_steps=depth,
                    use_cache=False
                )
                pred_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
                predicted_codes_list.append(pred_token)
                
            gen_codes_16 = torch.cat(predicted_codes_list, dim=0).unsqueeze(0) # [1, 16, T]
            
            # Synthesize waveform via Qwen Code2Wav
            wav = code2wav(gen_codes_16)
            wav_np = wav.squeeze().float().cpu().numpy()
            
        gen_time = time.time() - t_gen_start
        out_wav_path = os.path.join(out_dir, f"{sample_id}_ro500_mtp.wav")
        sf.write(out_wav_path, wav_np, 24000)
        
        dur = len(wav_np) / 24000
        print(f"  ✓ Synthesized: {out_wav_path} ({dur:.2f}s audio in {gen_time:.2f}s, RTF: {gen_time/dur:.2f}x)")
        print(f"    Transcript: \"{text[:70]}...\"")
        
        generated_audio_records.append({
            "sample_id": sample_id,
            "transcript": text,
            "wav_path": out_wav_path,
            "duration": round(dur, 2)
        })
        
    print("\nAll Romanian test audio synthesized cleanly!")
    return generated_audio_records

if __name__ == "__main__":
    evaluate_and_synthesize()
