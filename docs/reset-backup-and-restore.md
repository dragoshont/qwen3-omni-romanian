# PC reset backup and restore

The public repository is the canonical source for code, protocols, public
manifests, and review reports. Large or redistribution-sensitive local research
artifacts are backed up separately under the configured OneDrive account.

## Backup layout

The reset backup is stored under:

```text
OneDrive/Research Backups/qwen3-omni-romanian/2026-09-25-pre-reset/
├── workspace/                    repository snapshot, excluding .git/.venv
├── git/qwen3-omni-romanian.bundle
├── environment/ro-omni-venv.tar.gz
├── source-archive/               original supplied ZIP when present
├── backup-manifest.sha256
└── BACKUP_COMPLETE.json
```

`workspace/` includes the Git-ignored model weights, adapter checkpoints,
training audio, private codec-target JSONL files, generated audio, logs, audit
snapshots, and upstream source clones. These files are not evidence of public
redistribution permission; restoring them does not change their licensing.

## Restore

1. Wait for OneDrive to finish downloading the backup and make it available
   offline.
2. Clone the public repository, or restore the Git bundle:

   ```powershell
   git clone https://github.com/dragoshont/qwen3-omni-romanian.git ro-omni
   # Alternative, including local refs:
   git clone qwen3-omni-romanian.bundle ro-omni
   ```

3. Overlay the contents of `workspace/` into the cloned `ro-omni/` directory.
4. Verify every file against `backup-manifest.sha256` before running training.
5. Recreate Python 3.11.16 from `requirements-lock.txt`. The archived virtual
   environment is an emergency snapshot, not a portable substitute for a clean
   reinstall because Windows launchers can contain absolute paths.
6. Run:

   ```powershell
   python -m compileall -q scripts tests
   python -m unittest discover -s tests -v
   python scripts/run_t4_seeded_full200.py
   python scripts/run_confirmatory_training_matrix.py
   ```

Both long runners are designed to skip complete outputs or resume exact saved
training state. Partial generated-audio folders without a complete JSON report
are not valid evidence and are regenerated.

## Integrity

`BACKUP_COMPLETE.json` records the source Git commit, branch, file count, byte
count, manifest hash, and creation time. A backup is complete only when that
file exists and OneDrive reports synchronization finished.
