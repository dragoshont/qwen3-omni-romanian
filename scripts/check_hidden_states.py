import torch
from transformers import BitsAndBytesConfig, Qwen3OmniMoeTalkerForConditionalGeneration
import safetensors.torch
import torch.nn.functional as F

device = 'cuda:0'
sd_embed = safetensors.torch.load_file('models/thinker_embed_tokens.safetensors')
embed_weight = sd_embed['thinker.model.embed_tokens.weight'].to(device=device, dtype=torch.bfloat16)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
talker = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
    'models/qwen3-omni-talker',
    quantization_config=bnb_config,
    device_map=device,
    torch_dtype=torch.bfloat16,
)
talker.eval()

tts_special_tokens = torch.tensor([[151672, 151673, 151671]], device=device, dtype=torch.long)
tts_special_embed = F.embedding(tts_special_tokens, embed_weight)
tts_projected = talker.text_projection(tts_special_embed)
tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_projected.chunk(3, dim=1)

inputs_embeds = torch.randn(1, 9, 1024, device=device, dtype=torch.bfloat16)
trailing_text_hidden = torch.randn(1, 10, 1024, device=device, dtype=torch.bfloat16)
talker_input_ids = torch.full((1, 9), fill_value=151671, dtype=torch.long, device=device)
attention_mask = torch.ones((1, 9), device=device, dtype=torch.long)

with torch.no_grad():
    res = talker.generate(
        inputs_embeds=inputs_embeds,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        talker_input_ids=talker_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

print('res.sequences:', res.sequences)
print('res.hidden_states length:', len(res.hidden_states))
for i, h in enumerate(res.hidden_states):
    print(f'step {i}: type={type(h)}, len={len(h) if isinstance(h, tuple) else None}')
    if isinstance(h, tuple):
        last_e = h[-1]
        print(f'  last elem in tuple: type={type(last_e)}, shape={last_e.shape if hasattr(last_e, "shape") else None}')
