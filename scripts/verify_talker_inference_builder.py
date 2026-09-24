import sys
import os
import torch
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3OmniMoeConfig, Qwen3OmniMoeTalkerForConditionalGeneration, Qwen3OmniMoeCode2Wav

sys.stdout.reconfigure(encoding="utf-8")

def test_inference_builder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    print(f"Tokenizer loaded, vocab size: {len(tokenizer)}")
    
    # 2. Load thinker embed tokens
    print("Loading thinker embed tokens...")
    sd_embed = safetensors.torch.load_file("models/thinker_embed_tokens.safetensors")
    embed_weight = sd_embed["thinker.model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    print(f"Embed weight shape: {embed_weight.shape}")
    
    # 3. Load 4-bit Talker
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
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    talker.eval()
    print("Talker loaded successfully!")
    
    # 4. Prepare Romanian text
    text = "Bună ziua, cum vă simțiți astăzi?"
    prompt = f"<|im_start|>user\nVorbește în limba română.<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>\n"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    print(f"Prompt tokens: {token_ids.shape}, tokens: {token_ids[0].tolist()}")
    
    # Get thinker embeddings for prompt
    thinker_embed = F.embedding(token_ids, embed_weight) # [1, seq_len, 2048]
    print(f"Thinker embed shape: {thinker_embed.shape}")
    
    # 5. Build Talker inputs
    # Special tokens: tts_bos=151672, tts_eos=151673, tts_pad=151671
    tts_special_tokens = torch.tensor([[151672, 151673, 151671]], device=device, dtype=torch.long)
    tts_special_embed = F.embedding(tts_special_tokens, embed_weight)
    tts_projected = talker.text_projection(tts_special_embed) # [1, 3, 1024]
    tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_projected.chunk(3, dim=1)
    
    # Find assistant start index
    # <|im_start|> is 151644, assistant is 77091 (or similar)
    im_start_positions = torch.nonzero(token_ids[0] == 151644).view(-1)
    print(f"im_start_positions: {im_start_positions.tolist()}")
    # The last im_start is assistant
    assistant_im_start = im_start_positions[-1].item()
    assistant_tokens = token_ids[:, assistant_im_start:]
    print(f"Assistant tokens shape: {assistant_tokens.shape}")
    
    # Project assistant text through talker text_projection
    assistant_thinker_embed = thinker_embed[:, assistant_im_start:]
    assistant_hidden = talker.text_projection(assistant_thinker_embed) # [1, seq_len, 1024]
    print(f"assistant_hidden shape: {assistant_hidden.shape}")
    
    # Construct assistant_text_hidden
    # assistant_hidden[:, :3] is typically <|im_start|> assistant \n
    assistant_text_hidden = torch.cat(
        (
            assistant_hidden[:, :3],
            tts_pad_embed.expand(1, 4, -1),
            tts_bos_embed.expand(1, -1, -1),
            assistant_hidden[:, 3:4],
        ),
        dim=1,
    ) # length 9
    
    speaker_id = 2302 # Ethan
    codec_special_tokens = torch.tensor(
        [[2155, 2156, 2157, speaker_id, 2148, 2149]],
        device=device,
        dtype=torch.long,
    )
    codec_special_embeds = talker.get_input_embeddings()(codec_special_tokens) # [1, 6, 1024]
    
    assistant_codec_hidden = torch.cat(
        (
            torch.zeros((1, 3, 1024), device=device, dtype=torch.bfloat16),
            codec_special_embeds,
        ),
        dim=1,
    ) # length 9
    
    inputs_embeds = assistant_text_hidden + assistant_codec_hidden
    talker_input_ids = torch.full(
        (1, inputs_embeds.shape[1]),
        fill_value=151671, # tts_pad_token_id
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
    
    print(f"inputs_embeds shape: {inputs_embeds.shape}")
    print(f"trailing_text_hidden shape: {trailing_text_hidden.shape}")
    
    # 6. Test talker.generate
    print("Testing talker.generate (50 tokens max)...")
    with torch.no_grad():
        talker_result = talker.generate(
            inputs_embeds=inputs_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=talker.config.codec_eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    print("talker.generate succeeded!")
    print("Sequences shape:", talker_result.sequences.shape)
    print("Generated token count:", talker_result.sequences.shape[1])
    print("Hidden states count:", len(talker_result.hidden_states) if talker_result.hidden_states else 0)

if __name__ == "__main__":
    test_inference_builder()
