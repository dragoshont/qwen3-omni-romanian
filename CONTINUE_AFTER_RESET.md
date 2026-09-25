# Continue after the PC reset and RTX 5090 upgrade

This is the durable handoff for continuing the Romanian Qwen3-Omni research.
The public repository is the canonical source for code, protocols, and public
reports. The OneDrive snapshot holds the large and redistribution-sensitive
local artifacts that do not belong in GitHub.

## Authoritative locations

- Repository: <https://github.com/dragoshont/qwen3-omni-romanian>
- Corrective evaluation issue: <https://github.com/dragoshont/qwen3-omni-romanian/issues/1>
- Merged hardening review: <https://github.com/dragoshont/qwen3-omni-romanian/pull/2>
- OneDrive backup: `OneDrive/Research Backups/qwen3-omni-romanian/2026-09-25-pre-reset/`
- Backup verification and restore instructions: `docs/reset-backup-and-restore.md`

Use the newest `origin/main`. The parent revision immediately before this
handoff was `cd264bd`; the backup metadata records the exact final source
revision. Do not treat the backup as complete unless `BACKUP_COMPLETE.json`
exists and its checksum verification status is successful.

## Exact state at handoff

- GitHub issue #1 is deliberately open.
- PR #2 is merged; CI and the dependency-light unit suite passed.
- The seeded corrective T4 evaluation is **not complete**. One attempted run
  produced 187 of 200 WAV files for step 1,000 / seed 42, but no JSON report.
  That directory is an invalid, partial execution artifact and is not evidence.
- No confirmatory matrix job has completed. The frozen matrix specifies five
  conditions by three training seeds, for 15 training runs.
- Historical T0-T4 outputs remain exploratory. Historical T2 was mislabeled
  because its adapter contains partial MTP weights; historical T3 versus T4
  confounds data volume with schedule and compute.
- The external Romanian FLEURS set is frozen, but the confirmatory evaluation
  and aggregation runner still needs to be implemented and reviewed.
- Independent-ASR sensitivity, blinded native-Romanian listening,
  clean-environment reproduction, and data/voice-rights review remain release
  gates.

Do not delete, overwrite, or silently relabel legacy evidence. New results must
use distinct paths and must retain seeds, revisions, input hashes, environment,
hardware, and complete logs.

## Restore and verify

After OneDrive has downloaded the backup and made it available offline:

```powershell
git clone https://github.com/dragoshont/qwen3-omni-romanian.git ro-omni
Set-Location ro-omni
git pull --ff-only origin main

# Overlay the OneDrive backup's workspace/ tree into this checkout.
# Then follow docs/reset-backup-and-restore.md to verify backup-manifest.sha256.

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m compileall -q scripts tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The archived virtual environment is an emergency snapshot, not a portable
replacement for recreating Python 3.11 from the lock file. Windows virtual
environments can embed paths from the old machine.

## Next execution sequence

Run long jobs from a durable interactive terminal or a proper persistent job
runner. Do not launch them under a short-lived shell/tool process tree.

1. Complete issue #1 first:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_t4_seeded_full200.py
   .\.venv\Scripts\python.exe scripts\verify_t4_results.py
   ```

   Require all 1,200 generations: two checkpoints by three generation seeds by
   200 prompts. Inspect the aggregate prompt-clustered comparison. Commit only
   complete reports and logs; link the validating commit in issue #1 and close
   the issue only after the committed evidence passes review.

2. On the RTX 5090, record the new GPU, driver, CUDA, PyTorch, Python, package
   lock, Git revision, data hashes, wall-clock time, and peak VRAM. Then run the
   preregistered 15-job matrix:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_confirmatory_training_matrix.py
   ```

   The runner is resumable and skips only outputs that satisfy its completeness
   checks. Do not change the frozen design after looking at outcomes; document
   unavoidable hardware-driven deviations before continuing.

3. Before opening the external test, implement and test a confirmatory evaluator
   and aggregator that apply identical decoding and metric settings across all
   15 runs. Keep model selection independent of the external test. Report every
   seed, paired uncertainty, effect sizes, failures, and all preregistered
   contrasts rather than only the best run.

4. Complete the remaining release gates. Only then publish the redistributable
   adapter/checkpoint plus a model card and manifests to Hugging Face. Use a
   GitHub release and an archival DOI service such as Zenodo for the immutable
   research release. Unsloth is a training/runtime route, not the canonical
   artifact archive. Confirm upstream model, dataset, audio, voice, and derived
   checkpoint rights before public upload.

## Paste this prompt into Codex after restoration

```text
Continue the qwen3-omni-romanian research from the durable handoff in
CONTINUE_AFTER_RESET.md. First verify that this checkout is the latest
origin/main, read docs/reset-backup-and-restore.md, verify every OneDrive backup
file against backup-manifest.sha256, confirm BACKUP_COMPLETE.json says checksum
verification succeeded, and run the unit tests. Do not delete or overwrite
legacy evidence. Inspect GitHub issue #1 and the local reports before making any
claims.

Complete the seeded corrective T4 Full-200 evaluation in a durable process,
requiring two checkpoints x three generation seeds x 200 prompts and the final
aggregate report. Review the outputs adversarially, commit only valid reports
and relevant logs with issue #1 in the commit message, push them, comment with
the evidence, and close issue #1 only if all acceptance criteria are satisfied.

Then inventory the RTX 5090 software/hardware environment and run the frozen
five-condition x three-training-seed confirmatory matrix exactly as specified.
Implement and test the missing confirmatory external evaluator/aggregator before
opening the frozen FLEURS external test. Preserve provenance, hashes, seeds,
resumability, all-run reporting, paired uncertainty, and preregistered contrasts.
Stop publication if any data, voice, upstream-model, or derivative-weight right
is unresolved. Keep me updated with status and ETA during long runs.
```

## Temporary power setting used for backup

On 2026-09-25 the active Balanced plan's AC sleep timeout was changed from 15
minutes to Never so that OneDrive could finish. Hibernate on AC was already
Never. To restore the previous sleep timeout after the backup:

```powershell
powercfg /change standby-timeout-ac 15
```
