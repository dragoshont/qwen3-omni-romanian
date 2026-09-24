# Secondary Data Plan — Natural Romanian Narration

This lane must **not** alter the controlled T4 experiment.

## Why

After proving learning from clean 1–5 h data, the next question is whether professionally narrated Romanian gives better long-form language and prosody at equal duration.

Potential advantages:
- one consistent speaker across many hours;
- professional recording;
- long varied sentences;
- names, dates and uncommon vocabulary;
- natural rhythm and prosody;
- matching written text suitable for forced alignment.

## Private-R&D candidate

Neagu Djuvara self-narrated audiobooks are of particular research interest.

No Djuvara audio, book text or derived dataset is included in this public repository.

## Proposed pipeline

```text
lawfully acquired audio + lawfully accessed text
            |
            v
      normalization
            |
     ASR / forced alignment
            |
       fuzzy text match
            |
      confidence filter
            |
  short aligned utterances
            |
       Mimi encode
            |
    private R&D training
```

## Controlled comparison

After T4:

```text
same base + same adapters + same benchmark

5 h existing clean corpus
          vs
5 h natural audiobook corpus
```

Measure CER/WER, catastrophic generations, prosody, intelligibility, speaker similarity and native preference.

## Release boundary

Audiobook-derived checkpoints remain private by default until rights for text, recording, performance/voice, model distribution and intended use have been reviewed.
