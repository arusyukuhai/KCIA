"""GC for V86's *private* outgoing snapshot store only.

This namespace is never used for verified incoming state, certified epoch
transfers, genesis weights or an externally configured --shard-store. It is
safe to reclaim completed outgoing offers after ACK, retaining one latest
ready offer. Never delete while any link's outgoing transfer is unacknowledged.
"""
from __future__ import annotations

import re
from pathlib import Path

from .shards import read_manifest, shard_map

OFFER = re.compile(r'^offer-[A-Za-z0-9_-]+-[0-9]+-[0-9a-f]{16}\.fmanifest$')
OBJECT = re.compile(r'^[0-9a-f]{64}\.shard$')


def collect(offers_root: Path, *, busy: bool = False, keep_recent: int = 1) -> tuple[int, int]:
    """Call only while no export thread exists, and no outgoing transfer is busy.

    No symlinks, unknown filenames, external store paths or malformed manifests
    are followed or swept. Fail closed on bad references.
    """
    if busy:
        return (0, 0)
    offers_root = Path(offers_root)
    manifests = offers_root / 'manifests'
    store = offers_root / 'store' / 'objects'
    if (any(p.is_symlink() for p in (offers_root.parent,offers_root,
                                    manifests,offers_root/'store',store))
            or not manifests.is_dir()):
        return (0, 0)
    all_manifests = list(manifests.iterdir())
    if any(p.is_symlink() or not p.is_file() or not OFFER.fullmatch(p.name)
           for p in all_manifests):
        return (0, 0)
    all_manifests.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    recent = all_manifests[:keep_recent]
    # First validate every manifest. Do not delete anything if a malformed
    # current/old offer might reference a shard we would otherwise reclaim.
    indexed = {}
    for p in all_manifests:
        indexed[p] = set(shard_map(read_manifest(p)))
    keep = set().union(*(indexed[p] for p in recent)) if recent else set()
    removed_files = 0
    removed_manifests = 0
    if store.is_dir():
        for bucket in store.iterdir():
            if (bucket.is_symlink() or not bucket.is_dir() or
                    not re.fullmatch(r'[0-9a-f]{2}', bucket.name)):
                continue
            for path in bucket.iterdir():
                if (not path.is_file() or path.is_symlink() or
                        not OBJECT.fullmatch(path.name) or
                        path.name[:2] != bucket.name):
                    continue
                if path.name[:64] not in keep:
                    path.unlink()
                    removed_files += 1
    for old in all_manifests[keep_recent:]:
        old.unlink()
        removed_manifests += 1
    return removed_manifests, removed_files
