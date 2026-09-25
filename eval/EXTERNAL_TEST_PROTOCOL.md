# External Romanian test protocol

This protocol is frozen before the confirmatory T2/T3/T4 reruns. The external
test is used only after all training conditions and endpoints are complete.

## Source and selection

- source: `google/fleurs`, configuration `ro_ro`, split `test`;
- immutable dataset revision:
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`;
- 200 rows selected by the smallest SHA-256 values of
  `ro-omni-fleurs-external-v1-20260924:{audio_path}:{raw_transcription}`;
- any exact normalized transcript found in either training manifest is excluded;
- prompt manifest: `eval/ro_external_fleurs_v1.jsonl`;
- natural reference audio is retained locally for an ASR measurement-floor and
  signal-quality check, but audio is not used to condition model generation.

The selection is algorithmic rather than hand-curated. The project will not
change prompts, metrics, conditions, training duration, or model-selection
rules after viewing test results.

## Confirmatory outcomes

The primary automated outcome is normalized character error rate (CER), using
the normalizer in `scripts/eval_protocol.py`. Secondary outcomes are normalized
WER, strict CER/WER, direct codec-EOS rate, max-token rate, the declared
ASR-text repetition proxy, duration, clipping, non-finite samples, and runtime.

All compared systems receive prompt text only and use paired per-prompt seeds.
Training-seed uncertainty is reported across seeds 42, 314, and 2718. Paired
prompt-cluster bootstrap intervals are reported for planned contrasts.

## Planned contrasts

1. true Talker-only versus joint Talker+MTP at one hour, 1,000 updates;
2. one-hour versus five-hour joint training at 2,500 matched updates;
3. one-hour versus five-hour joint training at approximately 2.32 matched
   corpus passes (580 versus 2,500 updates respectively).

The fixed final endpoint is used for every confirmatory condition; external
test results do not select checkpoints. A model cannot be called a final
release solely from ASR metrics. Blinded native-Romanian listening remains a
separate release gate.
