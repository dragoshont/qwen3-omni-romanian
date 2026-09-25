# Qwen3-Omni Romanian

## Target: intelligent spoken conversations in Romanian

The long-term target is a Qwen3-Omni system that can understand Romanian
speech, reason over a multi-turn conversation, and answer directly with fluent,
natural Romanian speech while retaining the model's broader multimodal
abilities.

This repository currently studies one necessary part of that target: adapting
the Qwen3-Omni **Talker + multi-token-prediction (MTP) speech-generation path**
to Romanian on a 16 GB consumer GPU. It does not yet demonstrate a complete
conversational assistant, preserved multimodal capability, a validated speaker
identity, natural prosody, or low-latency duplex behavior.

> **Status: research preview under confirmatory rerun.** Historical T0-T4
> results are useful exploratory evidence, but they are not final model claims.

## Why the experiment is being rerun

An adversarial audit found three issues that materially limit the original
interpretation:

1. the Full-200 evaluator sampled stochastically without applying its recorded
   seed;
2. the condition labeled “Talker-only” also trained 204,800 MTP parameters
   because PEFT matched `q_proj`/`v_proj` recursively;
3. the one-hour and five-hour runs differed in data order, steps, warmup,
   schedule, exposure, and uncontrolled training randomness, so they did not
   isolate a pure data-scaling effect.

No evidence has been deleted or silently rewritten. Original reports remain as
historical artifacts. The full finding register and remediation requirements
are in [`reports/adversarial_peer_review.md`](reports/adversarial_peer_review.md).

## Corrected confirmatory design

The repaired campaign is frozen in
[`configs/confirmatory_matrix.json`](configs/confirmatory_matrix.json) and
[`experiments/CONFIRMATORY_PROTOCOL.md`](experiments/CONFIRMATORY_PROTOCOL.md).

- training seeds: 42, 314, and 2718;
- deterministic epoch shuffling;
- fresh adapters and fixed final endpoints;
- runtime assertions for exact Talker/MTP trainable-module isolation;
- resumable adapter, optimizer, scheduler, sampler, and RNG state;
- true Talker-only versus joint Talker+MTP at matched data and updates;
- one-hour versus five-hour joint training at matched updates;
- one-hour versus five-hour joint training at approximately matched corpus
  passes;
- paired multi-seed decoding with pinned Whisper scoring;
- a 200-prompt external Romanian FLEURS test frozen before retraining.

Full-200 is now explicitly a **development/selection set**, because Quick-40 is
its subset and the set has already been inspected repeatedly. Confirmatory
claims use the external protocol in
[`eval/EXTERNAL_TEST_PROTOCOL.md`](eval/EXTERNAL_TEST_PROTOCOL.md).

## Historical exploratory evidence

The original work established that the implementation can train and synthesize
prompt-only Romanian speech on an RTX 5080 16 GB. These values are retained for
audit, not presented as confirmatory comparisons.

| Historical run | Evaluation | Mean CER | Median CER | EOS | Repetition proxy | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T3 joint, ~1 h | Full-200 | 38.53% | 29.30% | 100% | 1.5% | exploratory single run |
| T4 step 1,000, ~5 h | Full-200 | 30.09% | 23.91% | 100% | 0.5% | unseeded development result |
| T4 step 2,000, ~5 h | Full-200 | 36.00% | 23.46% | 100% | 1.0% | unstable-tail development result |
| T4 step 2,500, ~5 h | Full-200 | 30.73% | 22.85% | 100% | 1.0% | provisional development candidate |

The old T2 label is invalid: it was Talker plus partial-MTP adaptation, not a
Talker-only control. The old T3/T4 contrast also mixed data volume with compute
and schedule. Those results must not be cited as causal ablations.

## Codec-bridge evidence

Thirty single-speaker clips were encoded with Mimi, restricted to 16 codebooks,
and decoded through Qwen Code2Wav. Whisper WER was approximately 5.8% on the
original clips and 6.4% after reconstruction. The paired difference was about
+0.58 percentage points with a bootstrap interval spanning zero.

This is narrow feasibility evidence for that dataset and configuration—not
proof that Mimi is Qwen's officially documented canonical encoder, and not a
general compatibility result across speakers, domains, or recording conditions.

## Architecture under test

```text
Qwen3-Omni base
├─ Thinker                                  frozen / cached embeddings
├─ Talker                                   NF4 base, LoRA r=8 alpha=16
│  └─ q_proj, v_proj
├─ MTP / code predictor                     joint conditions only
│  └─ q_proj, v_proj, o_proj,
│     gate_proj, up_proj, down_proj          LoRA r=8 alpha=16
└─ Code2Wav                                 frozen
```

The training targets contain 16 codec streams. The current single-speaker
corpus is reported by its pinned dataset card as the Romanian female speaker
“Sanda”; inference uses Qwen's `Ethan` speaker token. That conditioning mismatch
is documented and must be resolved before any voice-identity claim or release.

## Reproducing the repair

The exact direct-package snapshot is in `requirements-lock.txt`; Python 3.11.16,
PyTorch 2.11.0+cu128, Transformers 5.17.0, PEFT 0.21.0, and bitsandbytes 0.50.2
were used for the repair campaign.

```powershell
# Dependency-light repository checks
python -m compileall -q scripts tests
python -m unittest discover -s tests -v

# Saved-checkpoint correction for issue #1 (resumable by completed report)
python scripts/run_t4_seeded_full200.py

# Long, resumable 15-run confirmatory training matrix
python scripts/run_confirmatory_training_matrix.py
```

Base weights, codec targets, training audio, generated WAVs, and adapters are
intentionally excluded from ordinary Git history. Public manifests contain
stable source rows and content hashes without redistributing audio or codec
tokens. See [`docs/reproducibility.md`](docs/reproducibility.md) and
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## Publication plan

- **GitHub:** source, protocols, manifests, tests, report summaries, and issue
  history;
- **Hugging Face model repository:** versioned Talker/MTP adapters, model card,
  base-model relationship, immutable hashes, intended use, and limitations;
- **Hugging Face dataset repository:** only redistribution-cleared manifests,
  evaluation metadata, and licensed audio examples;
- **Zenodo:** an immutable GitHub release and report bundle with a DOI;
- **Unsloth:** compatibility/export path if the final architecture is supported,
  not the canonical scientific archive.

No adapter is a final release candidate until clean-environment reproduction,
rights review, the external test, independent-ASR sensitivity analysis, and a
blinded native-Romanian listening study are complete. Detailed gates are in
[`docs/artifact-publication-plan.md`](docs/artifact-publication-plan.md).

## Repository map

```text
configs/       frozen machine-readable experiment definitions
data/          documentation only; local audio is ignored
docs/          methodology, evidence, provenance, and release plans
eval/          development prompts and frozen external-test specification
experiments/   per-phase records and confirmatory protocol
manifests/     public code-free training manifests with hashes
models/        documentation only; local weights are ignored
reports/       historical evidence and generated audit reports
scripts/       preparation, training, evaluation, aggregation, and validation
tests/         dependency-light protocol and manifest integrity checks
```

## Responsible interpretation

ASR CER/WER measure an intelligibility proxy. They do not measure naturalness,
prosody, conversational intelligence, safety, consent, or speaker similarity.
Please cite a specific commit and label historical versus confirmatory evidence
when discussing results.
