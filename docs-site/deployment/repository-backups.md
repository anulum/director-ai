<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI — Repository history restore verification -->

# Repository Backups

Director-AI release and incident workflows verify repository history bundles
by restoring them into a fresh checkout and checking the restored `HEAD`.
Complete repository backup acceptance additionally requires the working-directory
archive, including ignored and untracked content, and its restoration evidence.

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

By default the verifier removes the checkout it created after success or
failure. An existing destination is refused and preserved, including a file
or symbolic link. The supplied parent directory is never removed.

The original directory is created and held before publication to the requested
name. Native Git accesses that held directory. Cleanup moves the candidate entry
into private custody without overwriting another name, checks it against the
held original object, then traverses that original descriptor. A replacement
is refused and returned without overwriting any concurrently acquired name.
If return is blocked, both entries remain and the failed report identifies the
retained recovery location; no uncertain entry is recursively removed.

Safe restoration requires Linux directory descriptors, procfs and
[`renameat2(RENAME_NOREPLACE)`](https://man7.org/linux/man-pages/man2/rename.2.html).
An unsupported operation fails rather than falling back to pathname cleanup.
A small `.director-ai-restore-custody-<uid>` directory remains in the supplied
parent, or the configured temporary base for a default parent. It must belong
to the invoking account and have mode `0700`; the verifier never relaxes these
permissions or removes this anchor. It removes only its invocation workspace.
Processes authorised as the same account are trusted inside that private
custody directory. Concurrent replacement of public restore names is handled
without deleting replacements. This account boundary is not isolation from
arbitrary same-account access to the private custody directory.

Use `--keep-restore` to retain the owned checkout for inspection, including
after a failed verification. This also works with the default temporary parent.
Use `--restore-parent PATH` to choose the parent location. Cleanup failures
produce a failed verification report; `restore_removed` records whether the
verifier actually removed its checkout. Ownership or recovery uncertainty is
reported as a failure, with retained entries preserved for inspection.

## Machine-readable Evidence

Use JSON output for audit logs and automation:

```bash
python tools/verify_repository_backup.py \
  BACKUP/repository_20260520T010141Z/DIRECTOR-AI_20260520T010141Z.bundle \
  --expected-head EXPECTED_HEAD_FROM_BACKUP_MANIFEST \
  --json
```

The JSON report includes command return codes, restored `HEAD`, `main` ref,
restore path, and whether the owned checkout was removed. Command records also
include the actual working directory and inherited directory descriptors with
their device/inode identities, so descriptor-based native commands remain
auditable after their descriptors close.

## Acceptance Rule

Do not mark a repository backup complete from checksum verification alone.
Checksums prove file integrity, not restoreability. A production backup record
must include checksum and restoration evidence for its working-directory
archive. Its history bundle additionally requires successful native Git restore
verification from `tools/verify_repository_backup.py`.
