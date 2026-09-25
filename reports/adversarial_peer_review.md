# Adversarial peer review and rerun requirements

**Review date:** 24 September 2026  
**Repository state reviewed:** `c3f98b20614de407d6e9f1b5ff3321ce6dc821fc`  
**Review posture:** attempt to falsify the principal claims before peer sharing

> **Audit snapshot:** findings describe the reviewed commit, not a claim that
> every item remains unfixed. Repair commit `be1f1b1` adds a true Talker-only
> control, explicit adapter-isolation assertions, deterministic shuffled
> training, resumable state, a frozen three-seed control matrix, source-pinned
> manifests, an external FLEURS test protocol, and CI integrity checks. Results
> remain pending until the preregistered reruns finish. Human listening,
> independent-ASR analysis, rights review, and clean-machine reproduction are
> still open release gates.

## Executive verdict

This repository contains a useful and unusually transparent engineering research
log, plus promising evidence that a Qwen3-Omni Talker path can be adapted to
produce intelligible Romanian speech on a 16 GB GPU. It is **not yet a
peer-review-ready causal experiment or a release-ready model evaluation**.

The saved artifacts support these narrower statements:

1. the reported T3/T4 adapter training ran on an RTX 5080 within approximately
   12.6 GB peak allocated VRAM;
2. the saved Talker and MTP adapters contain nonzero, checkpoint-varying LoRA
   tensors, so both adapter sets were updated;
3. prompt-only generation was evaluated on the published 200-prompt diagnostic
   set, and the old per-row CER/WER aggregates are internally recomputable;
4. no exact normalized transcript match was found between the public T4 manifest
   and the 200 diagnostic prompts;
5. the results justify continued research.

They do **not** yet establish that five hours of data caused the reported gain,
that step 2,500 is generally superior, that the system is a natural Romanian
voice, or that it can conduct an intelligent spoken conversation.

## Standards used for this review

The review uses the [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist)
for reproducibility, statistical reporting, assets and compute disclosure; the
[ACM artifact criteria](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
for documented, consistent, complete and exercisable artifacts; PyTorch's
[reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html);
the [Model Cards](https://arxiv.org/abs/1810.03993) and
[Datasheets for Datasets](https://arxiv.org/abs/1803.09010) reporting frameworks;
and [ITU-T P.808](https://www.itu.int/rec/T-REC-P.808-202106-I/en) plus recent
speech-synthesis evaluation critiques for subjective testing.

The base-model scope is checked against the official
[Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni) and
[technical report](https://arxiv.org/abs/2509.17765). Qwen lists Romanian among
neither the released model's ten speech-output languages nor its native speech
claims, so Romanian generation here is properly an experimental extension.

## Severity model

- **Critical:** invalidates or materially changes a headline claim.
- **Major:** prevents independent reproduction or release selection.
- **Moderate:** weakens interpretation, comparison or auditability.
- **Minor:** repository-quality issue that peers will notice but that does not
  change the numerical result.

## Critical findings

### C1. T3 versus T4 is not a controlled data-scaling experiment

The repository repeatedly calls T4 a controlled five-hour data-scaling
experiment. The implementation changes more than data volume:

- T3 uses 1,000 optimizer steps and 50 warmup steps;
- T4 uses 2,500 optimizer steps and 100 warmup steps;
- their cosine schedules therefore have different learning rates even at the
  nominally comparable step 1,000;
- T3 presents 4,000 examples; T4 presents 10,000 examples;
- neither run records a training seed or repeats training across seeds;
- fresh LoRA initialization is stochastic;
- the T4 loader iterates the manifest sequentially without shuffling.

The step-2,500 result can be described as a **larger-data, larger-compute
recipe**, not as the isolated effect of increasing data from one to five hours.
Any causal language about data scaling must be removed until the controlled
reruns below are complete.

### C2. The final benchmark has become a model-selection set

Quick-40 is a subset of Full-200. Quick-40 was evaluated repeatedly during
training and Full-200 was then used to compare steps 1,000, 2,000 and 2,500 and
select a champion. The same Full-200 set is being reused for the seeded
correction. It is therefore a **development/selection benchmark**, not an
untouched final test.

The public Git history begins after the results were generated and does not
provide a preregistered timestamp proving that prompts, candidate rules and
analyses were frozen before observation. This is not evidence of leakage, but it
limits confirmatory interpretation. A new external test must be frozen before
the next training results are inspected.

### C3. Training randomness is uncontrolled

The original training scripts do not set Python, NumPy, CPU or CUDA seeds before
adapter initialization and training, and they do not request deterministic
algorithms. A three-seed inference rerun measures **decoding variation only**;
it says nothing about variation caused by training data order, LoRA
initialization or CUDA kernels. At least three independent training runs per
headline condition are required for a defensible robustness claim.

### C4. Sequential training order creates unequal exposure

T4 cycles over the 4,311 manifest rows in fixed order. Across 10,000
presentations, the first 1,378 utterances are seen three times and the remaining
2,933 are seen twice. The triply exposed group also has a longer mean duration
(approximately 4.47 s versus 4.04 s), so order is correlated with training
weight. At step 1,000, only the first 4,000 of 4,311 rows have been seen.

This makes checkpoint comparisons order-dependent and means the step-1,000
checkpoint has not consumed the complete advertised five-hour corpus. Future
runs require deterministic epoch shuffling or a documented sampler.

### C5. T2 is not a Talker-only control

PEFT matches target module names recursively. T2 applies `q_proj` and `v_proj`
LoRA to the entire Talker object without excluding `code_predictor`. Direct
inspection of the saved T2 adapter finds:

- 80 non-Code-Predictor tensors / 696,320 parameters in the Talker proper;
- 20 Code-Predictor tensors / 204,800 parameters in MTP `q_proj`/`v_proj`;
- 901,120 adapter parameters in total.

T2 therefore trained **Talker q/v plus partial MTP q/v**, not Talker alone. T3
trained Talker q/v plus a broader MTP adapter on q/v/o/gate/up/down. The T2/T3
result cannot support “joint Talker+MTP beats Talker-only”; it supports only the
narrower observation that the fuller MTP-target recipe beat the partial-MTP
recipe in those single runs. A true Talker-only control must explicitly exclude
every `code_predictor` module and be retrained.

## Major findings

### M1. The original Full-200 evaluator did not apply its configured seed

The completed reports used stochastic decoding with `do_sample=True` but no RNG
initialization. This is the defect tracked in GitHub issue #1. The hardened
protocol in `eval/T4_SEEDED_PROTOCOL.md` corrects it with order-independent
per-prompt seeds and prevents 600 prompt-seed rows from being treated as 600
independent prompts.

The corrective run can select between saved checkpoints on automated
intelligibility metrics. It cannot repair training randomness or create an
untouched test set.

### M2. Old CER/WER scoring is under-specified and overly literal

The original evaluator lowercases strings but otherwise passes raw Whisper text
and references directly to JiWER. Punctuation, Unicode normalization,
Romanian comma-below versus cedilla forms and formatting choices can therefore
count as speech errors. The Whisper model revision is not recorded.

The hardened evaluator pins Whisper and reports both a documented normalized
score and the old strict-style score. Future papers must name the primary
normalizer in advance. ASR error must remain an intelligibility proxy, not an
audio-naturalness metric; the official
[Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md)
also documents model limitations and uneven performance across languages.

### M3. No blinded native-speaker evaluation exists

CER/WER cannot measure naturalness, prosody, timbre, accent appropriateness or
listener preference. The literature shows that TTS rankings can change with
instructions and listener setup; see
[Chiang et al. (2023)](https://www.isca-archive.org/interspeech_2023/chiang23_interspeech.html)
and [Kirkland et al. (2023)](https://www.isca-archive.org/ssw_2023/kirkland23_ssw.html).

A blinded, randomized native-Romanian study is a release gate, not optional
polish. The protocol must report recruitment, native-language background,
instructions, scale anchors, listening controls, exclusions, sample assignment,
compensation and uncertainty.

### M4. Dataset and codec provenance is incomplete

The preparation code loads the dataset and Mimi by mutable Hub names without a
revision. The local cache reveals revisions that were not recorded in the
experiment:

- dataset snapshot: `81231a262c34d30bacbcda4a0ba7ddc13ab88bc0`;
- Mimi snapshot: `89091b3e466eb6a9d11e537bf26b144f194978f7`;
- local Qwen model snapshot: `26291f793822fb6be9555850f06dfe95f2d7e695`.

The public T4 manifest omits the original dataset row IDs even though the private
code-target file recorded them. Its `seed: 42` is misleading because selection
does not use randomness. The config says `expected_unique_utterances: 3350`,
while the run used 4,311 rows, and labels selection as “coverage-aware” although
the script applies a threshold and keeps qualifying rows in stream order.

The current dataset card identifies 24,379 single-speaker clips under CC-BY-SA
4.0. The exact revision, attribution, share-alike implications for the tracked
transcripts and the status of derived weights need a release-specific review.

### M5. Artifact reproduction is not currently exercisable

There is no environment lock or installation command. `docs/reproducibility.md`
says exact lockfiles will be added later and even disagrees with the recorded
Python version. Scripts assume locally assembled model directories and private
ignored code-target files. Adapter checkpoints and generated audio are absent
from GitHub, and no immutable external revision is linked.

The repository can currently audit reports, but an independent peer cannot
reproduce the principal tables from a clean machine using only public artifacts.

### M6. The remaining component-ablation conclusions are single-run and asymmetric

T2 versus T3 shares the same nominal 1,000-step schedule and data family, but C5
shows that it is not the labeled ablation. In addition, each condition was
trained once and decoded once stochastically on Quick-40. The earlier MTP-only
curriculum used a different training history and data volume. Treat the old
T0/T1/T2/T3 matrix as exploratory and rerun a correctly isolated factorial
design.

### M7. The codec-bridge gate is too narrow for a general compatibility claim

The 30-clip bridge test is valuable feasibility evidence, but it uses a small
sample from the same narrow voice/domain family and reports only aggregate ASR
WER. It does not establish equivalence of codec semantics across speakers,
prosody, noise, duration or domains. “Practical bridge observed on 30 clips” is
supported; “validated compatibility” is too strong without a larger stratified
test and listening evidence.

### M8. Speaker conditioning is scientifically ambiguous

Training targets come from the female single-speaker Sanda dataset, while both
training and evaluation hard-code Qwen's `Ethan` speaker token. The experiment
does not measure whether the output follows Sanda, Ethan or a hybrid, and it has
no consent-aware speaker-similarity evaluation. This is a potential conditioning
confound and a model-release risk.

## Moderate findings

### R1. EOS and repetition were proxies, not direct failure labels

The old evaluator inferred EOS from waveform duration versus an assumed 12.5 Hz
limit. Repetition is detected only in Whisper text and can miss acoustic loops
or invent them through ASR. The hardened evaluator reads codec EOS directly;
repetition remains explicitly labeled an ASR-text heuristic. Manual acoustic
failure annotation is still needed.

### R2. Full-200 is diagnostic, not population-representative

The set contains exactly 20 author-created prompts in each of ten categories,
including project-specific technical prose and unusually long sentences. This
is useful for failure analysis but has no defined sampling frame for everyday
Romanian conversation. Aggregate averages must not be interpreted as expected
real-world user performance.

### R3. Exact-overlap checking is necessary but insufficient

The zero exact normalized match is correctly reported. It does not test audio
identity, source-document overlap, near-duplicate templates, memorized phrases
from the base model or benchmark exposure through iterative author inspection.
Future provenance should include stable source IDs, audio hashes, near-duplicate
thresholds and speaker-disjoint splits where possible.

### R4. No independent ASR sensitivity analysis exists

All headline objective intelligibility metrics depend on one Whisper model.
At minimum, repeat scoring with a Romanian-capable independent ASR family and
report disagreement, plus an ASR score on natural reference recordings for the
same test prompts to estimate the evaluator's floor.

### R5. Checkpoint selection used an arbitrary composite

The old score averages mean and median CER and adds a ten-percentage-point CER
penalty when repetition exceeds 1%. The units and threshold have no external
justification, and the rule was not publicly frozen before results. The hardened
protocol replaces this with a declared primary metric and prompt-clustered
interval, while preserving secondary outcomes.

### R6. Training checkpoints are not resumable experiments

Scripts save adapter weights but not optimizer state, scheduler state, sampler
position or RNG states. “Resumable/preemptible training” is therefore a roadmap
goal rather than a demonstrated property.

### R7. Important training details are absent from configs

The YAML files omit optimizer betas/weight decay, learning rates, warmup,
64-frame MTP subsampling, sample order, speaker token, dependency revisions,
trainable-parameter counts and determinism settings. The code contains these
facts, but peer reproduction should not require forensic reading.

## Repository consistency findings

Before peer sharing, the following contradictions should be corrected or marked
historical:

- `RESEARCH_QUESTIONS.md` says T4 Full-200 is pending;
- the README T4 table gives T3 repetition as 0.0%, while the verified value is
  1.5%;
- `evidence/experiment_registry.csv` and `models/README.md` call step 2,500 the
  champion/release candidate before seeded correction and listening;
- `configs/t4_joint_5h.yaml` expects 3,350 utterances but the manifest has 4,311;
- `scripts/README.md` says implementation is outside the scaffold although the
  repository now contains it, and lists filenames that do not exist;
- `ZIP_CONTENTS.md` records PEFT/bitsandbytes versions inconsistent with the
  checkpoint metadata and calls a 5.88 GB dry-run probe full-training memory;
- the preserved T2 manifest says rank 16 / alpha 32, while the script and saved
  adapter say rank 8 / alpha 16;
- reported trainable-parameter counts are inconsistent with the saved tensors,
  and the T2 count silently includes partial MTP adaptation;
- `CITATION.cff` has a placeholder author;
- the manuscript has no formal bibliography;
- the Apache license file is an abbreviated notice rather than the complete
  standard license text, and dataset-derived content needs separate licensing
  labels;
- there is no CI validation, test suite or report-schema check.

The common-secret signature scan found no obvious token or private-key pattern
in tracked text, and Python compilation succeeds. Those are useful hygiene
checks, not a security audit.

## Required reruns for peer-facing claims

### P0: Freeze a new evaluation design before training

1. Split evaluation into a visible development set and a genuinely untouched,
   versioned external test.
2. Define primary and secondary metrics, normalization, candidate selection,
   exclusions and statistical analysis in a committed protocol.
3. Use prompts drawn from documented Romanian sources or a documented sampling
   procedure, not only author-written diagnostic sentences.
4. Prepare natural reference recordings for the same prompts when licensing and
   consent permit.

### P1: Rerun the T2/T3 component ablation

Use at least three training seeds for true Talker-only and joint Talker+MTP with:

- the same pinned base, data rows, shuffle seeds, updates and scheduler;
- exact include/exclude filters that assert zero trainable Code-Predictor
  parameters in Talker-only and the intended nonzero count in joint training;
- explicit assertions and logged counts for trainable Talker and MTP parameters;
- deterministic epoch shuffling;
- a training-internal validation split not used for final claims;
- optimizer, scheduler, sampler and RNG state in checkpoints;
- hardened multi-seed decoding on development data;
- one final evaluation on the untouched test.

Rerun MTP-only under the same data and update budget if the paper retains a
three-way component-responsibility claim.

### P2: Rerun data scaling as a two-control study

For each of at least three training seeds, run:

1. **1 h, matched updates** — same optimizer-step budget as the 5 h run;
2. **1 h, matched epochs** — same number of data passes as the 5 h run;
3. **5 h, primary condition** — shuffled full corpus.

The matched-update comparison controls compute; the matched-epoch comparison
shows what happens when additional data also receives proportional exposure.
Using both prevents “data” and “optimization budget” from being silently
conflated. Freeze checkpoint selection on validation data before touching the
external test.

### P3: Rerun the codec bridge

Use a preregistered stratified sample spanning multiple speakers, domains,
durations, phonetic categories and recording conditions. Report per-clip
original-versus-reconstruction CER/WER, paired uncertainty, clipping and signal
checks, and blinded listener intelligibility/quality. Pin both Mimi and Code2Wav
artifacts by hash.

### P4: Conduct native-listener evaluation

Use at least 30 native Romanian listeners if feasible, balanced incomplete
blocks, randomized and blinded system labels, headphone/environment checks, and
separate questions for intelligibility, naturalness, prosody and preference.
Include natural speech and an appropriate anchor. Report confidence intervals
and a listener-aware statistical model rather than only a grand MOS average.

### P5: Re-evaluate all retained baselines

T0 needs no training rerun but should be decoded under the same pinned evaluator.
T1/T2/T3/T4 reports should use the same prompt split, per-prompt seeds,
normalizer, ASR revisions and failure annotation. Old reports remain historical
evidence and must not be mixed numerically with the corrected protocol without
clear labels.

## Recommended execution order

1. Finish the current hardened comparison of saved T4 steps 1,000 and 2,500.
2. Correct repository claims and archive labels; publish this review.
3. Pin data/model revisions and implement trainable-parameter assertions,
   deterministic shuffling, resumable state and environment locks.
4. Freeze the new dev/test protocol.
5. Rerun codec bridge and T2/T3 component experiments.
6. Run the controlled 1 h/5 h scaling study.
7. Run blinded listening and independent-ASR sensitivity analyses.
8. Only then freeze and publish a model-card “final” checkpoint.

## Bottom line for peers

The project has promising engineering evidence and has preserved enough raw
per-sample information to discover and correct several weaknesses. That is a
strength. The correct current label is **research preview with exploratory
results**, not validated Romanian conversational model and not confirmed causal
data-scaling result.
