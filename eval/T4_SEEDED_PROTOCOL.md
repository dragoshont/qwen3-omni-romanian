# T4 seeded checkpoint-selection protocol

Protocol frozen before the corrective evaluation for GitHub issue #1.

## Scope

This evaluation compares the already-trained T4 checkpoints at steps 1,000 and
2,500. It can select an operational candidate between those checkpoints. It
cannot establish a causal data-scaling effect, training-seed robustness,
naturalness, speaker similarity, or conversational ability.

Step 2,000 is not in the correction matrix because both its Quick-40 screen and
its prior Full-200 run showed catastrophic-tail instability. That exclusion is
based on earlier observed results and is therefore not an independent test-set
decision.

## Fixed design

- prompts: the 200 rows in `ro_holdout_200.jsonl`;
- checkpoints: steps 1,000 and 2,500;
- base generation seeds: 42, 314 and 2718;
- one generation per prompt, checkpoint and base seed (1,200 generations);
- prompt seed: first 64 bits of
  `SHA256("t4-seeded-v2:{base_seed}:{prompt_id}")`, reduced modulo `2^63-1`;
- identical prompt seeds across checkpoints;
- stochastic decoding: temperature 0.8, top-k 50, top-p 0.9 and repetition
  penalty 1.15;
- Whisper `openai/whisper-large-v3-turbo` revision
  `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`.

Per-prompt seeds make results independent of evaluation order and prevent a
long generation for one checkpoint from changing the random stream of every
later prompt.

## Outcomes

The primary automated outcome is normalized character error rate (CER). The
normalizer applies Unicode NFC, lowercasing, Romanian cedilla-to-comma mapping,
Unicode punctuation removal and whitespace collapse. Strict CER and WER are
also retained. Termination is read from generated codec token IDs. Repetition
is an ASR-transcript heuristic and is labeled as such.

## Uncertainty and decision rule

Uncertainty is computed by resampling the 200 prompts as clusters and retaining
all three fixed-seed observations within each selected prompt. The 600
prompt-seed rows are not treated as 600 independent test cases.

If the prompt-clustered 95% interval for the mean normalized-CER difference
excludes zero, the lower-CER checkpoint is the automated-metric selection. If
the interval includes zero, the result is statistically inconclusive; the
lower point estimate may be named only as the operational candidate.

No checkpoint is a final release champion until a preregistered, blinded native
Romanian listening study and an untouched external test set are completed.
