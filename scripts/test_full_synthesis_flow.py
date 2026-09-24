import os
import sys
import torch
import soundfile as sf
from transformers import BitsAndBytesConfig, Qwen3OmniMoeTalkerForConditionalGeneration, Qwen3OmniMoeCode2Wav, Qwen3OmniMoeConfig
import safetensors.torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

def test_full_synthesis_flow():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing end-to-end synthesis on {device}...")
    
    # 1. Load code2wav
    print("Loading Code2Wav...")
    full_config = Qwen3OmniMoeConfig.from_pretrained("models/qwen3-omni-partial")
    code2wav = Qwen3OmniMoeCode2Wav(full_config.code2wav_config).to(device=device, dtype=torch.bfloat16)
    c2w_sd = safetensors.torch.load_file("models/code2wav.safetensors")
    code2wav.load_state_dict(c2w_sd, strict=True)
    code2wav.eval()
    print("Code2Wav loaded.")
    
    # 2. Load 4-bit Talker
    print("Loading 4-bit Talker...")
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
    talker.eval()
    print("Talker loaded.")
    
    # 3. Load embed tokens
    sd_embed = safetensors.torch.load_file("models/thinker_embed_tokens.safetensors")
    embed_weight = sd_embed["thinker.model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    
    tts_special_tokens = torch.tensor([[151672, 151673, 151671]], device=device, dtype=torch.long)
    tts_special_embed = F.embedding(tts_special_tokens, embed_weight)
    tts_projected = talker.text_projection(tts_special_embed)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_projected.chunk(3, dim=1)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    text = "Bună ziua, acesta este un test de vorbire românească."
    prompt = f"<|im_start|>user\nVorbește în limba română.<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>\n"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    
    thinker_embed = F.embedding(token_ids, embed_weight)
    im_start_positions = torch.nonzero(token_ids[0] == 151644).view(-1)
    assistant_im_start = im_start_positions[-1].item()
    assistant_hidden = talker.text_projection(thinker_embed[:, assistant_im_start:])
    
    assistant_text_hidden = torch.cat(
        (
            assistant_hidden[:, :3],
            tts_pad_embed.expand(1, 4, -1),
            tts_bos_embed.expand(1, -1, -1),
            assistant_hidden[:, 3:4],
        ),
        dim=1,
    )
    
    speaker_id = 2302 # Ethan
    codec_special_tokens = torch.tensor(
        [[2155, 2156, 2157, speaker_id, 2148, 2149]],
        device=device,
        dtype=torch.long,
    )
    codec_special_embeds = talker.get_input_embeddings()(codec_special_tokens)
    assistant_codec_hidden = torch.cat(
        (
            torch.zeros((1, 3, 1024), device=device, dtype=torch.bfloat16),
            codec_special_embeds,
        ),
        dim=1,
    )
    
    inputs_embeds = assistant_text_hidden + assistant_codec_hidden
    talker_input_ids = torch.full(
        (1, inputs_embeds.shape[1]),
        fill_value=151671,
        dtype=torch.long,
        device=device,
    )
    trailing_text_hidden = torch.cat(
        (
            assistant_hidden[:, 4:],
            tts_eos_embed.expand(1, -1, -1),
        ),
        dim=1,
    )
    attention_mask = torch.ones((1, inputs_embeds.shape[1]), device=device, dtype=torch.long)
    
    print("Generating speech codes with Talker...")
    with torch.no_grad():
        res = talker.generate(
            inputs_embeds=inputs_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=talker.config.codec_eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    
    # Extract 16-group codes
    code_steps = [hid[-1] for hid in res.hidden_states if hid[-1] is not None]
    print(f"Generated {len(code_steps)} audio frames.")
    if len(code_steps) == 0:
        print("Error: No audio codes produced!")
        return
        
    codes = torch.stack(code_steps, dim=1).transpose(1, 2) # [1, 16, num_frames]
    print(f"Stacked codes shape: {codes.shape}")
    
    # Decode with Code2Wav
    print("Decoding with Code2Wav...")
    with torch.no_grad():
        wav = code2wav(codes) # or chunked_decode
    
    wav_np = wav.squeeze().cpu().float().numpy()
    os.makedirs("outputs/test_full_flow", exist_ok=True)
    out_wav_path = "outputs/test_full_flow/test_sample.wav"
    sf.write(out_wav_path, wav_np, 24000)
    print(f"Decoded waveform length: {len(wav_np)} samples ({len(wav_np)/24000:.2f} s)")
    print(f"Saved test audio to {out_wav_path}")
    
    del talker, code2wav
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run Whisper evaluation on this audio
    from transformers import pipeline
    print("Evaluating with Whisper large-v3-turbo...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        torch_dtype=torch.float16,
    )
    audio_data, sr = sf.read(out_wav_path)
    audio_dict = {"raw": audio_data, "sampling_rate": sr}
    asr_res = asr(audio_dict, generate_kwargs={"language": "ro", "task": "transcribe"})
    print("Whisper transcription:", asr_res["text"])

if __name__ == "__main__":
    test_full_synthesis_flow()
