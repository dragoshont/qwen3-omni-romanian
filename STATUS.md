# Status

_Last updated: 2026-09-25 EEST_

## Active correction

GitHub issue #1 tracks the stochastic-seed defect in the historical Full-200
checkpoint comparison. The repaired evaluation compares saved T4 steps 1,000
and 2,500 over 200 development prompts and three paired generation seeds
(1,200 generations). No final selection is reported until that matrix and its
prompt-clustered analysis complete.

## Confirmatory retraining

The adversarial audit also found issues that saved-checkpoint evaluation cannot
repair. A frozen 15-run campaign (five conditions × three training seeds) is
specified in `configs/confirmatory_matrix.json`:

- true Talker-only versus joint Talker+MTP at 1,000 updates;
- one-hour versus five-hour joint training at 2,500 matched updates;
- one-hour versus five-hour joint training at approximately 2.32 matched
  corpus passes.

All new runs use deterministic shuffling, explicit adapter isolation, fixed
endpoints, complete resumable state, seed-specific outputs, and pinned local
input hashes. A 200-prompt Romanian FLEURS external test was frozen before the
new training campaign.

## Historical evidence label

Original T0-T4 reports remain available and unchanged as exploratory evidence.
The T2 report is mislabeled historically because the adapter contains partial
MTP weights. The original T3/T4 comparison confounds data volume with schedule
and compute. Step 2,500 is therefore only a historical development candidate,
not a release champion.

## Release gates

External-test analysis, independent-ASR sensitivity, blinded native-Romanian
listening, clean-environment reproduction, and data/voice-rights review remain
required before model publication.
