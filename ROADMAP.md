# Roadmap

The end goal is natural, intelligent spoken conversation in Romanian with the
Qwen3-Omni architecture. The current work validates and improves the Romanian
speech-generation path; full conversational and duplex behavior comes later.

## Phase A — feasibility (complete)

- [x] Isolate the speech-generation path.
- [x] Establish a practical 16-stream target-token bridge.
- [x] Validate frozen Code2Wav reconstruction.
- [x] Fit real Talker/MTP adaptation on an RTX 5080 16 GB.
- [x] Demonstrate autonomous prompt-text-only Romanian generation.
- [x] Establish a learning signal with ~1 hour of Romanian speech.

## Phase B — controlled scaling through T4 (complete)

- [x] T3: ~1-hour joint Talker/MTP run.
- [x] T4: ~5-hour run with the same architecture and fresh adapters.
- [x] Evaluate T4 steps 1,000, 2,000 and 2,500 on Full-200.
- [x] Recompute aggregates from per-sample reports and audit train/eval alignment.
- [x] Preserve steps 1,000, 2,000 and 2,500.
- [x] Name step 2,500 the provisional composite-score champion.

## Phase C — release-candidate verification (next)

The original Full-200 runs used stochastic decoding without applying the
documented seed. Complete this phase before making a final champion claim.

- [x] Repair the derived step-2,500 report's UTF-8 text and record provenance.
- [x] Confirm zero exact normalized overlap between 4,311 T4 training transcripts
  and 200 held-out prompts.
- [x] Add explicit seed handling and evaluation metadata to the evaluator.
- [ ] Re-evaluate steps 1,000 and 2,500 on Full-200 with seeds 42, 314 and 2718.
- [ ] If the ranking remains close, extend both candidates to five total seeds.
- [ ] Report across-seed mean, median, trimmed mean, P90, catastrophic-failure and
  repetition statistics with paired uncertainty intervals.
- [ ] Run a blinded native-Romanian A/B study on a stratified prompt set, including
  conversational, technical-English and long-sentence failures.
- [ ] Freeze the final checkpoint selection rule and release candidate.

Decision rule: step 2,500 remains the provisional candidate; step 1,000 remains
the stable-mean challenger. ASR metrics alone do not finalize the model.

## Phase D — Romanian corpus audit and expansion

### D1. Candidate inventory and rights matrix

Audit each candidate independently before downloading or mixing it into a public
release lane:

- [ ] **MARA**;
- [ ] Mozilla Common Voice Romanian;
- [ ] VoxPopuli Romanian;
- [ ] USPDATRO;
- [ ] RTASC;
- [ ] `datadriven-company/TTS-Romanian`;
- [ ] `eduardem/romanian-speech-v2`;
- [ ] the existing `eduardem/romanian-tts-single-speaker` source and exact T4
  revision.

For every source, record the stable identifier/revision, license, redistribution
terms, speaker/performer consent and voice rights, hours, speakers, domains,
sampling rates, transcript quality, known duplication, and train/evaluation
exclusions. “Available online” is not sufficient release clearance.

### D2. Controlled equal-duration comparisons

- [ ] Build a deterministic, release-cleared 5-hour subset for each viable source.
- [ ] Use the same stock base, adapter targets, optimizer budget, seeds, decoding
  configuration and Full-200 protocol.
- [ ] Compare sources individually before constructing mixtures.
- [ ] Measure CER/WER tails, native intelligibility, naturalness, prosody, speaker
  similarity, category coverage and catastrophic failures.
- [ ] Test whether MARA or another source improves conversational and long-form
  behavior without sacrificing the clean-corpus baseline.

### D3. Multi-source curriculum

- [ ] Create a deduplicated public-release mixture only from cleared sources.
- [ ] Balance speakers, recording conditions, sentence length, phonetic coverage,
  numbers/names and Romanian–English code switching.
- [ ] Compare curriculum ordering against uniform mixing at equal total duration.
- [ ] Add a larger homogeneous-data point only if it answers a defined scaling
  question.

### D4. Separate private natural-narration lane

- [ ] Build the legally separated audiobook alignment pipeline described in
  `docs/secondary_data_plan.md`.
- [ ] Compare 5 hours of clean TTS-style speech with 5 hours of natural narration.
- [ ] Keep audiobook-derived data and checkpoints private until text, recording,
  performer/voice, model-distribution and intended-use rights are cleared.

## Phase E — full-model integration

- [ ] Generate canonical frozen Thinker conditioning from full BF16 Qwen3-Omni.
- [ ] Verify cached-feature parity with end-to-end generation.
- [ ] Measure preserved non-Romanian and multimodal capability.
- [ ] Evaluate multi-turn Romanian conversational contexts.

## Phase F — duplex Romanian conversation

- [ ] Study Moshi/J-Moshi and DuplexOmni behavioral patterns.
- [ ] Add interruption, overlap, backchannels and turn-taking scenarios.
- [ ] Measure first-audio latency, real-time factor and interruption response.
- [ ] Prototype Romanian duplex behavior after acoustic quality is stable.

## Phase G — public release

- [ ] Reproduce training/inference from a clean locked environment.
- [ ] Complete dataset and voice-rights review.
- [ ] Publish the final Talker + MTP adapters and model card on Hugging Face.
- [ ] Publish evaluation prompts/metadata and cleared generated audio separately.
- [ ] Add responsible-use, limitations and synthetic-audio transparency notes.
- [ ] Tag the matching GitHub release and archive it on Zenodo for a DOI.
- [ ] Publish the paper/preprint with immutable artifact revisions and checksums.
