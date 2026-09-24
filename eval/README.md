# Evaluation

## Full-200 Romanian benchmark

200 unseen prompts, 20 per category:

1. conversational
2. ă/â/î
3. ș/ț
4. ce/ci/ge/gi/che/chi/ghe/ghi
5. consonant clusters
6. numbers/dates/currency
7. Romanian names/places
8. English technical loanwords / code-switching
9. questions/exclamations
10. long natural sentences

## Quick-40

Four prompts per category. Used for checkpoint screening, not final claims.

## Preservation benchmark

Planned `E_PRESERVE_100`:
- 25 English
- 25 French
- 25 Italian
- 25 Spanish

## Required metrics

- independent ASR WER/CER
- mean, median and P90
- EOS success
- max-token termination
- repetition/collapse
- latency / RTF
- peak memory
- per-category breakdown
- native-human listening when comparing mature checkpoints
