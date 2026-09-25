# Confirmatory component and data-scaling protocol

Frozen before the repaired training campaign. Machine-readable conditions are
in `configs/confirmatory_matrix.json`.

## Questions

1. At a fixed one-hour corpus, 1,000-update schedule, does joint Talker+MTP
   adaptation outperform a true Talker-only control?
2. At 2,500 matched optimizer updates, does the five-hour corpus outperform the
   one-hour corpus?
3. At approximately 2.32 matched corpus passes, does the five-hour corpus
   outperform the one-hour corpus?

These are separate contrasts. No result will be described as a pure data-scale
effect unless its corresponding compute/exposure control supports that wording.

## Frozen design

- training seeds: 42, 314, and 2718;
- deterministic epoch shuffle keyed by training seed;
- fresh adapters from the same local Qwen base for every run;
- Talker LoRA: rank 8, alpha 16, `q_proj` and `v_proj` only;
- joint MTP LoRA: rank 8, alpha 16, `q_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`;
- explicit runtime assertions reject any MTP trainable tensor in Talker-only;
- AdamW betas 0.9/0.95, weight decay 0.01;
- Talker LR 2e-5 and MTP LR 4e-5;
- cosine schedule and the warmup/step counts in the matrix config;
- gradient accumulation 4;
- fixed final endpoints, without evaluation-driven early stopping;
- resumable state includes adapters, optimizer, scheduler, sampler, and RNGs.

The matched-pass one-hour condition uses 580 optimizer updates (2,320 example
presentations over 1,000 rows). The five-hour condition uses 2,500 updates
(10,000 presentations over 4,311 rows, approximately 2.32 passes). Because the
pass count is approximate and clip durations differ, both presentation counts
and audio-duration exposure will be reported.

## Evaluation and analysis

Full-200 is development data because it has already been used for checkpoint
selection. Confirmatory claims use the frozen external FLEURS test described in
`eval/EXTERNAL_TEST_PROTOCOL.md`. Every condition is evaluated only after all
training endpoints exist. Evaluation is paired by prompt and generation seed.

Primary analysis uses normalized CER with prompt-cluster bootstrap intervals.
Training seeds are independent experimental replicates; generation seeds are
repeated measurements, not independent models. Secondary results include WER,
strict scores, EOS, max-token termination, repetition proxy, clipping, duration,
and latency. Raw per-prompt results remain available for audit.

Automated intelligibility does not establish naturalness, prosody, speaker
identity, safety, or conversational intelligence. Those require the separately
specified native-listener study and end-to-end conversational evaluation.
