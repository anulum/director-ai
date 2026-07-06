# Repository Backups

Director-AI release and incident workflows use git bundle backups for repository
history. A backup is not accepted until it has been restored into a fresh
checkout and the restored `HEAD` has been verified.

## Verify a Bundle

Run the verifier with the expected commit hash from the backup manifest:

```bash
python tools/verify_repository_backup.py \
  BACKUP/repository_20260520T010141Z/DIRECTOR-AI_20260520T010141Z.bundle \
  --expected-head EXPECTED_HEAD_FROM_BACKUP_MANIFEST
```

The verifier performs these checks:

- `git bundle verify` succeeds;
- the bundle clones into a fresh temporary directory;
- restored `HEAD` matches `--expected-head` when provided;
- `refs/heads/main` resolves inside the restored checkout;
- `git fsck --full --strict` exits successfully.

By default the temporary restore directory is removed after verification. Use
`--keep-restore` with `--restore-parent PATH` when an operator needs to inspect
the restored checkout.

## Machine-readable Evidence

Use JSON output for audit logs and automation:

```bash
python tools/verify_repository_backup.py \
  BACKUP/repository_20260520T010141Z/DIRECTOR-AI_20260520T010141Z.bundle \
  --expected-head EXPECTED_HEAD_FROM_BACKUP_MANIFEST \
  --json
```

The JSON report includes command return codes, restored `HEAD`, `main` ref,
restore path, and whether the temporary checkout was removed.

## Acceptance Rule

Do not mark a repository backup complete from checksum verification alone.
Checksums prove file integrity, not restoreability. A production backup record
must include both checksum verification and a successful restore verification
from `tools/verify_repository_backup.py`.
