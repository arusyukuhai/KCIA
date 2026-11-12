"""Fail-safe space checks and narrowly scoped disposable cache housekeeping.

Never prune signed events, origin cursors, trusted epoch certificates, active
optimizer moment files, checkpoints or user-supplied HF data. No global scan.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2
MIN_FREE_DEFAULT_MIB = 1024


class DiskPressureError(OSError):
    """Not enough *reserved* space for a durable operation (before ENOSPC)."""


def free_bytes(path):
    path = Path(path)
    while not path.exists():
        if path == path.parent:
            raise FileNotFoundError(path)
        path = path.parent
    return shutil.disk_usage(path).free


def require_space(path, *, additional=0, reserve_mib=MIN_FREE_DEFAULT_MIB,
                  action='disk write'):
    if additional < 0 or reserve_mib < 0:
        raise ValueError('invalid disk reservation')
    free = free_bytes(path)
    needed = additional + reserve_mib * MIB
    if free < needed:
        raise DiskPressureError(
            f'KAGAMI DISK PRESSURE: {action}: free={free // MIB} MiB, '
            f'required={needed // MIB} MiB (including {reserve_mib} MiB reserve). '
            'Stop training, run `python kagami.py disk-report`, free space or '
            'enable a certified epoch transition. Signed events were NOT deleted.')
    return free


def _safe_directory(path):
    return path.is_dir() and not path.is_symlink()


def cleanup_working_orphans(journal_path):
    """Offline-only after exclusive journal lock, before spawning saver threads.

    Only delete old disk optimizer *snapshot* folders not referenced by the
    current safetensors working checkpoint, whose metadata can be verified.
    An interrupted prior process cannot still own the same locked journal.
    """
    from safetensors import safe_open
    target = Path(str(journal_path) + '.working.safetensors')
    parent = target.parent
    if not _safe_directory(parent):
        return 0
    keep = None
    if target.is_file() and not target.is_symlink():
        try:
            with safe_open(str(target), framework='pt', device='cpu') as reader:
                meta = json.loads((reader.metadata() or {})['fedpara'])
                keep = (meta.get('disk_optimizer') or {}).get('folder')
        except (KeyError, OSError, ValueError, TypeError):
            # Could be a partially damaged recovery point. Never delete its
            # possible optimizer dependencies while provenance is uncertain.
            return 0
    removed = 0
    for candidate in parent.glob(target.name + '.moments-*'):
        if (not _safe_directory(candidate) or
                not re.fullmatch(re.escape(target.name) + r'\.moments-[0-9a-f]{32}', candidate.name) or
                candidate.name == keep):
            continue
        shutil.rmtree(candidate)
        removed += 1
    return removed


def snapshot_cache_report(journal_path):
    """Read-only, bounded top-level report. A file size is NOT its allocated size."""
    base = Path(journal_path)
    paths = {
        'signed journal': [base, Path(str(base) + '.fplog'),
                           Path(str(base) + '-wal'), Path(str(base) + '-shm')],
        'fast resume': [Path(str(base) + '.fast-resume')],
        'P2P snapshots': [Path(str(base) + '.snapshots')],
        'live epochs': [Path(str(base) + '.live-epoch')],
        'working state': [Path(str(base) + '.working.safetensors')],
        'disk optimizer backups': list(base.parent.glob(base.name + '.working.safetensors.moments-*')),
        'GUI log': [Path(str(base) + '.gui.log')],
    }
    # A GUI peer keeps its journal under rooms/<room>/peers/<profile>/,
    # while effective full models and manifests live in sibling room folders.
    if base.parent.parent.name == 'peers' and base.name == 'events.sqlite3':
        room = base.parent.parent.parent
        profile = base.parent.name
        if room.is_dir() and not room.is_symlink() and re.fullmatch(r'peer-[0-9a-f]{16}',profile):
            paths['GUI model object store'] = [room/'models'/profile]
            paths['GUI checkpoint manifests'] = [room/'checkpoints'/profile]
            paths['GUI critic state'] = [room/'critics'/profile]
            # .fedpara-gui/rooms/<room> -> .fedpara-gui/telemetry
            paths['GUI telemetry history'] = [
                room.parent.parent/'telemetry'/f'history-{room.name}-{profile}.jsonl']
    elif base.name == 'events.sqlite3' and base.parent.name not in ('peers','.'):
        # Legacy GUI profile stored directly in the room directory.
        room = base.parent
        if room.parent.name == 'rooms':
            paths['GUI model object store'] = [room/'models'/'legacy']
            paths['GUI checkpoint manifests'] = [room/'checkpoints'/'legacy']
            paths['GUI critic state'] = [room/'critics'/'legacy']
    if base.name == 'elastic-events.sqlite3':
        paths['CLI checkpoint exports'] = [base.parent/'live-checkpoints']
    def measure(p):
        if p.is_symlink(): return 0
        try:
            if p.is_file(): return p.stat().st_size
            if p.is_dir():
                total = 0
                for current, dirs, files in os.walk(p, followlinks=False):
                    dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
                    for name in files:
                        f = Path(current) / name
                        if f.is_file() and not f.is_symlink():
                            total += f.stat().st_size
                return total
        except OSError:
            return 0
        return 0
    return {name:sum(measure(x) for x in targets) for name,targets in paths.items()}


def _signed_log_quota(journal_path):
    """Read committed logical payload usage and last local cap, without opening a trainer."""
    import sqlite3
    path = Path(journal_path).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        return None
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=.3) as db:
            rows = dict(db.execute(
                "SELECT name,value FROM meta WHERE name IN "
                "('archive_used_bytes','last_local_journal_cap_bytes')"))
            count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return int(rows.get("archive_used_bytes", "0")), (
            int(rows["last_local_journal_cap_bytes"])
            if "last_local_journal_cap_bytes" in rows else None), count
    except (sqlite3.Error, ValueError, OSError):
        return None


def report(journal_path):
    path = Path(journal_path).expanduser().absolute()
    print(f'KAGAMI DISK REPORT: {path}')
    print(f'available on volume: {free_bytes(path) / GIB:.2f} GiB')
    signed = _signed_log_quota(path)
    if signed is not None:
        used, limit, events = signed
        cap_info = (f" / {limit/GIB:.3f} GiB "
                    f"({used*100/max(1,limit):.1f}% of configured cap)"
                    if limit else " (no recorded cap)" if limit is None
                    else " (uncapped)")
        print(f"signed update log: {used/GIB:.3f} GiB{cap_info}; "
              f"{events} indexed events")
    data = snapshot_cache_report(path)
    for key, val in sorted(data.items(),key=lambda pair:pair[1],reverse=True):
        print(f'  {val / GIB:9.3f} GiB   {key}')
    print('NOTE: Hugging Face cache, custom --shard-store, TMPDIR and other '
          'volumes are not counted; no files were modified.')
    return data


def reclaim(journal_path):
    """Offline maintenance of ONLY bounded, disposable caches; never signed log.

    Obtain EventJournal's exclusive .train.lock first; even the preview/report
    may run concurrently, but reclaim MUST NOT race active trainer/network IO.
    """
    import fcntl
    journal = Path(journal_path).expanduser().absolute()
    lock_path=Path(str(journal)+'.train.lock')
    if lock_path.is_symlink() or journal.is_symlink() or journal.parent.is_symlink():
        raise ValueError('unsafe symlink in journal path')
    fd=os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Kagami trainer is running; close GUI and training first') from exc
        from .snapshot_offer_gc_v86 import collect
        stats={}
        stats['orphan optimizer backups']=cleanup_working_orphans(journal)
        stats['outgoing offers removed'],stats['outgoing shards removed'] = collect(
            Path(str(journal)+'.snapshots')/'offers')
        # Cached full model is optional, but retain its verified current
        # reference and all objects reachable from it.
        from .fast_resume_v85 import (_pointer,_manifest_path,
                                      _factor_path,_clean_unreferenced)
        root=Path(str(journal)+'.fast-resume')
        current=_pointer(root)
        if current and not root.is_symlink():
            manifest=_manifest_path(root,current['manifest'])
            factor=_factor_path(root,current['factors'])
            if manifest.is_file() and factor.is_file():
                _clean_unreferenced(root,manifest,factor)
        print('KAGAMI OFFLINE RECLAIM: signed journal/epoch/model untouched')
        for name, count in stats.items():
            print(f'  {name}: {count}')
        report(journal)
        return stats
    finally:
        os.close(fd)
