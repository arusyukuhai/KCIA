"""Kagami v85: crash-consistent, materialized local FedSum resume checkpoints.

A *private* exact committed-weight snapshot accelerates local restart. Signed
journal events stay intact for peer NEED/anti-entropy; only a separately
certified generation transition can retire them. A crash after a journal
append but before the next materialization replays just that signed suffix.

Never accept snapshots from the network or use them as consensus authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .core import PatchError
from .shards import (export_state_sharded, apply_sharded_effective_state,
                     read_manifest, verify_shards, shard_map, _blob_path)

HEX = re.compile(r'^[0-9a-f]{64}$')
FORMAT = 'kagami-committed-resume-v85'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _sync_parent(path):
    fd = os.open(Path(path).parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.with_name('.' + uuid.uuid4().hex + '.tmp')
    try:
        with name.open('xb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
        _sync_parent(path)
    finally:
        name.unlink(missing_ok=True)


def root_for(journal):
    return Path(str(journal.path) + '.fast-resume')


def _pointer(root):
    """Treat invalid/incomplete local caches as unavailable; journal remains authority."""
    path = root / 'latest.json'
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 32768:
        return None
    try:
        p = json.loads(path.read_text(encoding='utf8'))
        if (not isinstance(p, dict) or p.get('format') != FORMAT or
                not all(isinstance(p.get(k), str) and HEX.fullmatch(p[k])
                        for k in ('manifest', 'factors')) or
                not isinstance(p.get('cursor'), dict)):
            return None
        return p
    except (OSError, ValueError, UnicodeError):
        return None


def _factor_path(root, digest):
    if not HEX.fullmatch(digest): raise PatchError('invalid resume factor identifier')
    return root / 'factors' / (digest + '.safetensors')


def _manifest_path(root, digest):
    if not HEX.fullmatch(digest): raise PatchError('invalid resume state identifier')
    return root / 'manifests' / (digest + '.fmanifest')


def _factor_tensors(adapter):
    result = {}
    with adapter.lock, torch.no_grad():
        for i, group in enumerate(adapter.groups):
            for j, part in enumerate(group.overlay.local.parts):
                result[f'l.{i}.{j}'] = part.detach().to('cpu').contiguous().clone()
                result[f'a.{i}.{j}'] = getattr(group.overlay.anchor, f'part{j}').detach().to('cpu').contiguous().clone()
    return result


def _restore_factors(reader, adapter):
    expected = set()
    staged = []
    with adapter.lock, torch.no_grad():
        for i, group in enumerate(adapter.groups):
            for j, part in enumerate(group.overlay.local.parts):
                anchor = getattr(group.overlay.anchor, f'part{j}')
                for code, dest in [('l', part), ('a', anchor)]:
                    key = f'{code}.{i}.{j}'
                    expected.add(key)
                    src = reader.get_tensor(key)
                    if (src.shape != dest.shape or src.dtype != dest.dtype or
                            not torch.isfinite(src).all().item()):
                        raise PatchError('resume factor/anchor shape, dtype or finite check failed')
                    staged.append((dest, src))
        if set(reader.keys()) != expected:
            raise PatchError('unexpected factor inventory in materialized resume')
        # Apply only after all factors were validated (state has already passed
        # manifest/shard verification before importing the committed model).
        for dest, src in staged:
            dest.copy_(src.to(dest.device))
            if isinstance(dest, torch.nn.Parameter):
                dest.grad = None


def _clean_unreferenced(root, new_manifest, new_factors):
    """Reclaim only our *private* expired cache objects, never signed history.

    A crash during cleanup is harmless: latest.json is already durable and
    references only the surviving new checkpoint. Orphans are swept next time.
    """
    try:
        keep = set(shard_map(read_manifest(new_manifest))) if new_manifest is not None else set()
        store = root / 'store' / 'objects'
        if store.is_symlink(): return
        if store.is_dir():
            for bucket in store.iterdir():
                if bucket.is_symlink() or not bucket.is_dir() or not re.fullmatch(r'[0-9a-f]{2}',bucket.name):
                    continue
                for f in bucket.iterdir():
                    if f.is_symlink() or not f.is_file(): continue
                    if re.fullmatch(r'[0-9a-f]{64}\.shard',f.name) and f.name[:64] not in keep:
                        f.unlink()
        for d, suffix, selected in ((root/'factors','.safetensors',new_factors),
                                    (root/'manifests','.fmanifest',new_manifest)):
            if d.is_symlink() or not d.is_dir(): continue
            for f in d.iterdir():
                if f.is_symlink() or not f.is_file() or f == selected: continue
                if re.fullmatch(r'[0-9a-f]{64}' + re.escape(suffix), f.name): f.unlink()
    except (OSError, PatchError) as exc:
        print(f'KAGAMI RESUME cache GC deferred: {exc}', flush=True)


def save(journal, engine, *, max_state_bytes, shard_bytes=4 * 1024**2,
         min_free_bytes=2 * 1024**3, cursor=None, carry_archive=None, carry_cursor=None):
    """Publish a full committed base with a cursor, then reclaim expired caches.

    MUST be called at a sealed optimizer boundary under inbound_commit_lock.
    The whole export is fenced by this caller; no inbound event may commit.
    The journal and carry cursors are collected and compared on the owning
    event-loop thread, never accessed via SQLite within this model worker.
    It deliberately does not store an uncommitted local factor contribution in
    the effective base (which would double-apply local training on resume).
    """
    root = root_for(journal)
    if root.is_symlink(): raise PatchError('fast resume cache path is a symlink')
    manager = getattr(engine.adapter.model, '_fedpara_lazy_mutable_weights', None)
    if manager is not None and getattr(manager,'stores',None):
        # v60 page COW lacks a complete bounded-memory commit-only exporter.
        # Refuse rather than silently capturing *genesis* instead of weights.
        raise PatchError('fast resume full weight cache is unavailable with active paged COW')
    # One whole-model copy is unavoidable for an EXACT independent restart
    # base. Never exhaust physical disk to satisfy an optional speed cache.
    # For content-addressed shards this overestimates space when data repeats.
    from .shard_runtime import effective_state_sources
    from contextlib import nullcontext
    ledger = manager if getattr(manager, '_fedpara_is_factor_ledger', False) else None
    with (ledger.export_context() if ledger is not None else nullcontext()):
        expected = sum(t.numel()*t.element_size() for t, _ in
                       effective_state_sources(engine.adapter.model, engine.adapter).values())
    expected = min(expected, max_state_bytes)
    root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    if disk.free < expected + min_free_bytes:
        raise OSError(f'not enough free space for materialized checkpoint: '
                      f'free={disk.free}, estimated={expected}, reserve={min_free_bytes}')
    if cursor is None:
        raise PatchError("sealed main-thread journal cursor is required for safe async export")
    if carry_archive is not None:
        # Main and old-genesis carry journal MUST be fenced by the SAME
        # inbound_commit_lock for the entire materialization.  Both cursors
        # describe one atomic model snapshot.  A carry lane cannot be inferred
        # from the main journal tip or the signed generation certificate.
        # v98: SQLite connections are confined to their creating thread.
        # The caller captures BOTH journal cursors on the event-loop thread,
        # holds inbound_commit_lock across this worker export, and validates
        # BOTH cursors again after the worker returns (elasticnet.py).
        # Do not call carry_archive.journal.fast_resume_cursor() here: MPS
        # and CPU model_task() execute in different threads, and querying
        # sqlite3 here crashes the entire training step on every carry epoch.
        if not isinstance(carry_cursor, dict):
            raise PatchError('sealed main-thread carry cursor required for async export')
        if not getattr(carry_archive, '_replayed', False) or getattr(carry_archive, '_poisoned', False):
            raise PatchError('cannot materialize incomplete/poisoned carry state')
    elif carry_cursor is not None:
        raise PatchError('carry cursor supplied without carry archive')
    tensors = _factor_tensors(engine.adapter)
    root.joinpath('factors').mkdir(exist_ok=True)
    root.joinpath('manifests').mkdir(exist_ok=True)
    name = root / 'factors' / ('.' + uuid.uuid4().hex + '.part')
    try:
        save_file(tensors, str(name), metadata={'format': FORMAT})
        with name.open('rb') as fh: os.fsync(fh.fileno())
        factor_hash = _sha(name)
        factor_path = _factor_path(root, factor_hash)
        os.replace(name, factor_path)
        _sync_parent(factor_path)
    finally:
        name.unlink(missing_ok=True)
    # The export writes immutable hashed data blocks, then an atomic manifest.
    temp_manifest = root / 'manifests' / ('.' + uuid.uuid4().hex + '.part')
    try:
        manifest = export_state_sharded(
            engine.adapter.model, root/'store', temp_manifest,
            model_id=engine.adapter.model_id, revision=engine.adapter.revision,
            peer_id=engine.self_id, seq=cursor['count'], adapter=engine.adapter,
            shard_bytes=shard_bytes, max_state_bytes=max_state_bytes,
            committed_base_only=True)
        manifest_hash = _sha(temp_manifest)
        manifest_path = _manifest_path(root, manifest_hash)
        os.replace(temp_manifest, manifest_path)
        _sync_parent(manifest_path)
    except BaseException:
        # A failed optional export must not leave orphaned shard gigabytes
        # consuming the space needed by the signed journal.
        previous = _pointer(root)
        if previous:
            try:
                old_manifest = _manifest_path(root,previous['manifest'])
                old_factor = _factor_path(root,previous['factors'])
                if old_manifest.is_file():
                    _clean_unreferenced(root,old_manifest,old_factor)
            except (OSError,PatchError):
                pass
        else:
            try:
                _clean_unreferenced(root, None, None)
            except (OSError,PatchError): pass
        raise
    finally:
        temp_manifest.unlink(missing_ok=True)
    info = {'format': FORMAT, 'genesis': journal.genesis,
            'peer': engine.self_id, 'model_id':engine.adapter.model_id,
            'revision':engine.adapter.revision, 'layout':engine.adapter.layout,
            'kind':engine.adapter.kind, 'rank':engine.adapter.rank,
            'alpha':engine.adapter.alpha, 'factor_count':len(tensors),
            'cursor':cursor, 'manifest':manifest_hash, 'factors':factor_hash}
    if carry_archive is not None:
        info['carry'] = {'genesis':carry_archive.genesis, 'cursor':carry_cursor}
    _atomic(root/'latest.json', json.dumps(info,sort_keys=True,separators=(',',':')).encode())
    _clean_unreferenced(root, manifest_path, factor_path)
    print(f'KAGAMI MATERIALIZED COMMITTED BASE: {cursor["count"]} events covered; '
          f'{manifest["total_tensor_bytes"]} weight bytes; suffix-only resume ready',flush=True)
    return info


def restore(journal, engine, *, max_state_bytes, carry_archive=None):
    """Restore a fully validated local cached base, returning covered rowid.

    If the cache is missing/stale, return None BEFORE modifying any model.
    Once sharded import begins, any IO/model failure is fatal, not a replay
    fallback onto partially modified weights.
    """
    root = root_for(journal)
    if root.is_symlink(): raise PatchError('unsafe fast resume root')
    p = _pointer(root)
    if p is None:
        print('KAGAMI FAST RESUME: no valid committed checkpoint; signed-history replay required once',flush=True)
        return None
    if (p.get('genesis') != journal.genesis or p.get('peer') != engine.self_id or
            p.get('model_id') != engine.adapter.model_id or
            p.get('revision') != engine.adapter.revision or
            p.get('layout') != engine.adapter.layout or
            p.get('kind') != engine.adapter.kind or
            p.get('rank') != engine.adapter.rank or
            p.get('alpha') != engine.adapter.alpha):
        return None
    rowid = journal.fast_resume_suffix(p['cursor'])
    if rowid is None:
        print('KAGAMI FAST RESUME: cached current-generation cursor invalid; full replay required', flush=True)
        return None
    carry_rowid = None
    if carry_archive is not None:
        binding = p.get('carry')
        if not isinstance(binding, dict) or binding.get('genesis') != carry_archive.genesis:
            print('KAGAMI FAST RESUME: snapshot predates carry-aware checkpoint; full replay required', flush=True)
            return None
        carry_rowid = carry_archive.journal.fast_resume_suffix(binding.get('cursor'))
        if carry_rowid is None:
            print('KAGAMI FAST RESUME: cached carry cursor invalid; full replay required', flush=True)
            return None
        if not getattr(carry_archive, '_replayed', False) and getattr(carry_archive, '_poisoned', False):
            raise PatchError('poisoned carry archive cannot be restored')
        # Independent append-only lanes have no durable joint commit ordering.
        # Using both suffixes on an old snapshot could yield a DIFFERENT model
        # due to float addition / local anchor changes. Refuse that shortcut
        # until a unified, durably ordered transaction index is implemented.
        if (journal.count() != p['cursor']['count'] and
                carry_archive.journal.count() != binding['cursor']['count']):
            print('KAGAMI FAST RESUME: both journal lanes advanced after snapshot; exact-order full replay', flush=True)
            return None
    elif 'carry' in p:
        # Never restore a cache for a different active epoch/carry layout.
        return None
    m = _manifest_path(root,p['manifest'])
    f = _factor_path(root,p['factors'])
    if (not m.is_file() or not f.is_file() or m.is_symlink() or f.is_symlink() or
            _sha(m)!=p['manifest'] or _sha(f)!=p['factors']):
        print('KAGAMI FAST RESUME local checkpoint missing/corrupt; full signed replay',flush=True)
        return None
    meta = read_manifest(m,max_state_bytes=max_state_bytes)
    if (meta['model_id'] != engine.adapter.model_id or
            meta['revision'] != engine.adapter.revision or
            meta['peer_id'] != engine.self_id or
            meta['seq'] != p['cursor']['count']):
        return None
    # The complete checkpoint is verified before touching model parameters.
    verify_shards(meta,root/'store')
    with safe_open(str(f),framework='pt',device='cpu') as reader:
        if (reader.metadata() or {}).get('format') != FORMAT:
            raise PatchError('invalid local resume factor encoding')
        # Sharded import rebuilds the *committed* base and clears overlays;
        # afterward restore the exact committed local+anchor factor coordinates.
        apply_sharded_effective_state(
            engine.adapter.model,engine.adapter,m,root/'store',
            model_id=engine.adapter.model_id,revision=engine.adapter.revision,
            max_state_bytes=max_state_bytes)
        _restore_factors(reader,engine.adapter)
    print(f'KAGAMI FAST RESUME loaded materialized committed base '
          f'at event={p["cursor"]["count"]}',flush=True)
    return (rowid, carry_rowid) if carry_archive is not None else rowid
