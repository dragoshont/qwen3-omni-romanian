# Scripts

The local experimental implementation exists outside this scaffold and is being cleaned before publication.

Planned public scripts:

```text
scripts/
├── encode_mimi_targets.py
├── validate_codec_bridge.py
├── build_ro_manifest.py
├── check_eval_leakage.py
├── train_talker_mtp_lora.py
├── generate_prompt_only.py
├── score_asr.py
├── compare_checkpoints.py
└── summarize_experiment.py
```

Publication criteria:
- no local machine paths
- no credentials
- exact upstream revision
- deterministic seed
- CLI help
- resume-safe checkpoints
- machine-readable outputs
