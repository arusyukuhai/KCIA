"""Explicit, offline reset of KCIA-managed *local* persisted data.

This operation intentionally does not touch Hugging Face's shared cache,
source trees, or custom dataset paths. It is independent of PyTorch/GUI deps.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import stat
import sys
from pathlib import Path

from .storage_location import choose_data_root

GUI_DIR = '.kcia-gui'
CLI_JOURNAL = 'elastic-events.sqlite3'
CLI_CHECKPOINTS = 'live-checkpoints'
# Entries emitted by train-elastic for its default --journal. No wildcards:
# every file must have this exact application-specific prefix AND suffix.
JOURNAL_SUFFIXES = (
    '', '-wal', '-shm', '.fplog', '.identity.pem', '.room-manifest.json',
    '.trusted.json', '.trusted.json.migrated', '.trusted.sqlite3',
    '.trusted.sqlite3-wal', '.trusted.sqlite3-shm',
    '.live-epoch', '.generation.sqlite3', '.generation.sqlite3-wal',
    '.generation.sqlite3-shm', '.generation-compact.sqlite3',
    '.generation-compact.sqlite3-wal', '.generation-compact.sqlite3-shm',
    '.generation-compact-stage.sqlite3', '.generation-compact-stage.sqlite3-wal',
    '.generation-compact-stage.sqlite3-shm', '.generation-state',
    '.snapshots', '.fast-resume', '.epoch-v2.sqlite3',
    '.epoch-v2.sqlite3-wal', '.epoch-v2.sqlite3-shm',
    '.critic', '.critic-v77.sqlite3', '.critic-v78.sqlite3',
    '.critic-v78.sqlite3-wal', '.critic-v78.sqlite3-shm',
    '.trajectory-v82.sqlite3', '.trajectory-v82.sqlite3-wal',
    '.trajectory-v82.sqlite3-shm', '.handoff.json', '.gui.log',
    '.working.safetensors', '.working.json', '.policy.json',
)


def _check_root(raw: Path) -> Path:
    raw = raw.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f'data root is a symlink (refusing): {raw}')
    root = raw.resolve(strict=False)
    if root == root.parent or not root.is_dir():
        raise ValueError(f'data root must be an existing directory other than /: {root}')
    if root.name == GUI_DIR:
        raise ValueError(f'pass the PARENT of {GUI_DIR}, not the data directory itself')
    return root


def _targets(root: Path, include_cli_defaults: bool, cli_cwd: Path) -> list[Path]:
    gui = root / GUI_DIR
    if gui.is_symlink():
        raise ValueError(f'{gui} is a symlink; refusing reset')
    output = [gui] if gui.exists() else []
    if include_cli_defaults:
        # Explicit --include-cli-defaults is limited to the invocation cwd.
        # CLI --journal, --checkpoint-dir and offload overrides are not inferred.
        if cli_cwd.is_symlink() or not cli_cwd.is_dir():
            raise ValueError('CLI working directory must be a real directory')
        journal = cli_cwd / CLI_JOURNAL
        for suffix in JOURNAL_SUFFIXES:
            candidate = Path(str(journal) + suffix)
            if candidate.exists() or candidate.is_symlink():
                output.append(candidate)
        directory = cli_cwd / CLI_CHECKPOINTS
        if directory.exists() or directory.is_symlink():
            output.append(directory)
    return output


def _stat_bytes(path: Path) -> tuple[int, int]:
    """Count only local filesystem entries, without following any symlinks."""
    total, count = 0, 0
    def stat_one(p: Path) -> None:
        nonlocal total, count
        s = p.lstat()
        count += 1
        if stat.S_ISREG(s.st_mode):
            total += s.st_size
    stat_one(path)
    if path.is_dir() and not path.is_symlink():
        for parent, dirs, files in os.walk(path, topdown=True, followlinks=False):
            for name in dirs + files:
                sub = Path(parent) / name
                stat_one(sub)
    return total, count


@contextlib.contextmanager
def _assert_inactive(gui: Path):
    """Block reset when an existing GUI peer profile holds its live flock.

    This is a best-effort guard. CLI training with custom storage is outside
    this mechanism. Close all GUI and CLI workers before deletion.
    """
    locks = []
    try:
        if gui.is_dir():
            # A second, idle KCIA GUI may not own any per-peer lock.
            # Require an exclusive app lock in addition to journal locks.
            app_lock = gui / 'app.lock'
            if app_lock.exists() or app_lock.is_symlink():
                if app_lock.is_symlink():
                    raise RuntimeError(f'unsafe symbolic link in GUI lock: {app_lock}')
                import fcntl
                fd = os.open(app_lock, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, BlockingIOError):
                    os.close(fd)
                    raise RuntimeError('Another KCIA GUI is still open; close it before resetting.')
                locks.append(fd)
            rooms = gui / 'rooms'
            if rooms.is_dir() and not rooms.is_symlink():
                import fcntl  # Only available on POSIX (GUI peer-lock platform).
                for parent, dirs, files in os.walk(rooms, followlinks=False):
                    dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
                    for name in files:
                        if name not in ('active.lock', 'legacy.lock'):
                            continue
                        path = Path(parent) / name
                        if path.is_symlink():
                            raise RuntimeError(f'unsafe symbolic link in GUI lock: {path}')
                        fd = os.open(path, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except (OSError, BlockingIOError):
                            os.close(fd)
                            raise RuntimeError(
                                f'KCIA is still running (locked profile {path}); '
                                'stop every GUI and training worker before resetting.')
                        locks.append(fd)
        yield
    finally:
        for fd in locks:
            os.close(fd)


def preview_local_data(*, root: Path, include_cli_defaults: bool = False,
                       cli_cwd: Path | None = None) -> dict:
    """Read-only listing used by the double-confirmed desktop reset dialog."""
    root = _check_root(Path(root))
    cwd = Path.cwd() if cli_cwd is None else Path(cli_cwd)
    targets = _targets(root, include_cli_defaults, cwd)
    records = []
    for path in targets:
        if path.is_symlink():
            raise RuntimeError(f'unsafe symbolic link in deletion targets: {path}')
        size, count = _stat_bytes(path)
        records.append({'path': str(path), 'bytes':size, 'entries':count})
    return {'root':str(root), 'include_cli_defaults':include_cli_defaults,
            'targets':records, 'total_bytes':sum(r['bytes'] for r in records),
            'total_entries':sum(r['entries'] for r in records)}


def reset_local_data(*, root: Path, yes: bool = False,
                     include_cli_defaults: bool = False,
                     cli_cwd: Path | None = None, stream=None) -> int:
    out = stream if stream is not None else sys.stdout
    root = _check_root(Path(root))
    cwd = Path.cwd() if cli_cwd is None else Path(cli_cwd)
    targets = _targets(root, include_cli_defaults, cwd)
    print(f'KCIA storage root: {root}', file=out)
    print(f'CLI defaults: {"included (cwd: " + str(cwd) + ")" if include_cli_defaults else "excluded"}', file=out)
    if not targets:
        print('No matching KCIA saved data found; nothing deleted.', file=out)
        return 0
    total_bytes, total_entries = 0, 0
    for path in targets:
        b, n = _stat_bytes(path)
        total_bytes += b
        total_entries += n
        print(f'  {path}  ({b:,} bytes; {n:,} entries)', file=out)
    print(f'Total: {total_bytes:,} bytes; {total_entries:,} filesystem entries.', file=out)
    if not yes:
        print('DRY RUN ONLY. No data deleted. Add --yes to irreversibly delete the paths above.', file=out)
        return 0
    gui = root / GUI_DIR
    # Validate *every* target before touching any of them; abort atomically on
    # a detected link so a malformed CLI target cannot partially wipe GUI state.
    for path in targets:
        if path.is_symlink():
            raise RuntimeError(f'refusing to delete a symbolic link as a target: {path}')
    with _assert_inactive(gui):
        # Re-check symlinks at the exact point of destructive operation.
        for path in targets:
            if path.is_symlink():
                raise RuntimeError(f'refusing to delete a symbolic link as a target: {path}')
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    print('RESET COMPLETE. Local KCIA state deleted. Shared HF caches and source code retained.', file=out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python kcia.py reset-local-data',
        description='Preview or delete KCIA GUI stored rooms, models, checkpoints, '
                    'signed journals, peer identities, graphs, and logs. '
                    'This irreversibly resets local training progress.')
    parser.add_argument('--root', type=Path, help='Root containing .kcia-gui; '
                        'default: the same persistent root used by KCIA GUI')
    parser.add_argument('--yes', action='store_true', help='Actually delete (default is preview only)')
    parser.add_argument('--include-cli-defaults', action='store_true',
                        help='Also delete ONLY known default CLI journal and live-checkpoints '
                             'in the current working directory; never custom paths')
    args = parser.parse_args(argv)
    try:
        selected_root = args.root if args.root is not None else choose_data_root()
        if args.root is None and not selected_root.exists():
            print(f'KCIA storage root: {selected_root}')
            if not args.include_cli_defaults:
                print('No saved GUI state found; nothing deleted.')
                return 0
            # --include-cli-defaults may still delete exact known CLI artifacts.
            # Create no directories: select cwd as a benign empty preview root.
            selected_root = Path.cwd()
        return reset_local_data(
            root=selected_root, yes=args.yes,
            include_cli_defaults=args.include_cli_defaults)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'KCIA RESET ABORTED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
