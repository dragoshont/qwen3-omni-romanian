# Architecture Notes

## Current working path

```text
text / multimodal context
        |
        v
     Thinker
      frozen
        |
        v
     Talker
 predicts stream 0
        |
        +----------------+
        |                |
        v                v
       MTP           codec EOS
predicts streams 1..15
        |
        v
  16 codec streams
        |
        v
    Code2Wav
      frozen
        |
        v
     waveform
```

## Target-code hypothesis

Third-party Qwen3-Omni-derived training implementations and our bridge test support:

- 24 kHz waveform input;
- Mimi encoding;
- first 16 codebooks;
- ~12.5 Hz / ~80 ms frames;
- Talker supervised on stream 0;
- MTP supervised on streams 1–15;
- Code2Wav frozen.

Qwen Code2Wav applies codebook offsets internally. Do **not** manually add those offsets during target preparation.

## Trainable modules

### Talker
- LoRA rank 8
- alpha 16
- targets: `q_proj`, `v_proj`

### MTP
- LoRA rank 8
- alpha 16
- targets: `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Frozen
- Thinker during current RTX experiments
- Code2Wav
- Mimi target encoder

The narrow adapter surface is intentional: early experiments isolate target correctness, supervision, memory feasibility and data scaling before expanding the trainable model.
