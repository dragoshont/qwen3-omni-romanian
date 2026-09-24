# Data Provenance

The public repository contains manifests, experiment metadata and evaluation specifications — **not redistributed third-party speech corpora**.

## Primary early corpus

### `eduardem/romanian-tts-single-speaker`

Research notes recorded during selection:

- ~34.3 hours
- ~24,379 utterances
- single female speaker ("Sanda")
- 24 kHz
- dataset card reported CC-BY-SA-4.0

The early 1 h / 5 h curriculum uses this kind of clean single-speaker speech to reduce confounding variables.

Before any public model release, the exact dataset revision and licence obligations must be re-verified.

## Other Romanian sources under consideration

- `datadriven-company/TTS-Romanian`
- `eduardem/romanian-speech-v2`
- VoxPopuli Romanian
- Mozilla Common Voice Romanian
- USPDATRO
- RTASC
- MARA

These differ substantially in licence, provenance and speaker diversity and should not be treated as one interchangeable pool.

## Private-R&D audiobook lane

Professionally narrated audiobooks paired with matching text are being investigated as a secondary private-research lane.

No audiobook audio or book text is committed here.

## Manifest requirements

Every publishable training corpus should record:
- source URL / dataset ID
- revision or snapshot date
- licence
- speaker IDs where available
- original sampling rate
- selected duration
- deterministic selection rule
- train/eval exclusion
- source hashes or stable row IDs where possible
- preprocessing steps
- manual removals
