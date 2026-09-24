# Roadmap

## Phase A — feasibility

- [x] Isolate the speech-generation path.
- [x] Establish a practical 16-stream target-token bridge.
- [x] Validate frozen Code2Wav reconstruction.
- [x] Fit real Talker/MTP adaptation on RTX 5080 16 GB.
- [x] Demonstrate autonomous prompt-text-only Romanian generation.
- [x] Establish a learning signal with ~1 h Romanian.

## Phase B — controlled scaling

- [x] T3: ~1 h joint Talker/MTP.
- [ ] T4: ~5 h, same architecture, fresh adapters.
- [ ] Select T4 champion on Full-200.
- [ ] Add a larger homogeneous-data point if it remains informative.
- [ ] Report a learning curve with tail/failure analysis.

## Phase C — data quality and prosody

- [ ] Build a legally separated high-quality audiobook alignment lane.
- [ ] Compare equal-duration clean TTS-style speech vs natural narration.
- [ ] Evaluate prosody, speaker similarity and native preference separately from ASR.
- [ ] Keep rights-unclear/private R&D data out of the public release path.

## Phase D — full-model integration

- [ ] Generate canonical frozen Thinker conditioning from full BF16 Qwen3-Omni.
- [ ] Verify cached-feature parity with end-to-end generation.
- [ ] Measure preserved non-Romanian capability.
- [ ] Evaluate longer conversational contexts.

## Phase E — duplex

- [ ] Study Moshi/J-Moshi and DuplexOmni behavioral patterns.
- [ ] Add interruption, overlap, backchannels and turn-taking scenarios.
- [ ] Measure latency as a first-class metric.
- [ ] Prototype Romanian duplex behavior after acoustic quality is stable.

## Phase F — release

- [ ] Reproduce from a clean environment.
- [ ] Audit dataset rights.
- [ ] Decide adapter/weight release scope.
- [ ] Add model card and responsible-use notes.
- [ ] Add synthetic-audio marking/transparency where required.
- [ ] Publish paper/preprint and exact artifacts.
