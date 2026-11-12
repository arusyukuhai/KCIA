"""v88 local post-certification GC for compact v2 trainer generations.

Only an already-verified quorum-certified checkpoint and an EXACT, durable
signed-suffix carry archive can replace an old journal.  Each per-origin cut
prefix is independently verified against the certificate, and every post-cut
source identity is required in the immutable carry ledger with the same SHA.
Never run against an active old writer.  A durable retirement marker is fsynced
before any unlink, so a power loss during GC cannot trick restart into opening
an empty old writer. This module does not delete current-genesis history.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

from .core import PatchError, _canonical_json
from .epoch_live_rebase_v52 import verify_certified_cut
from .event_archive import payload_path, read_frame
from .epoch_trainer_boot_v52 import read_durable_epoch_pointer
from .shards import read_manifest, verify_shards, semantic_shard_sha256

FORMAT = 'kcia-v88-compact-certified-retirement'


def _read_receipt(carry_path):
    receipt_path = Path(str(carry_path)+'.handoff.json')
    if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.stat().st_size > 4096:
        raise PatchError('no safe durable compact handoff receipt')
    try: data = json.loads(receipt_path.read_bytes())
    except (ValueError, UnicodeError) as exc: raise PatchError('corrupt handoff receipt') from exc
    return data


def marker_for(boot, receipt):
    """Exact local witness bound to the certified epoch and committed handoff."""
    return {'format':FORMAT, 'epoch':boot.epoch,
            'manifest':boot.manifest_hash, 'source_genesis':boot.previous_genesis,
            'target_genesis':boot.current_genesis,
            'source_rowid':receipt['source_rowid'],
            'archive_genesis':receipt['archive_genesis'],
            'archive_digest':receipt['archive_digest'],
            'total_archived':receipt['total_archived'],
            'cut':boot.proposal['cut'],
            'new_journal':str(boot.journal_path.resolve())}


def verify_retirement_marker(boot, receipt):
    path=Path(str(boot.source_path)+'.retired.json')
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4096:
        raise PatchError('missing/unsafe certified compact retirement marker')
    if path.read_bytes() != _canonical_json(marker_for(boot, receipt)):
        raise PatchError('retired source tombstone does not match certified carry and checkpoint')
    return True


@contextmanager
def _read_only(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():
        raise PatchError('unsafe or missing GC input')
    db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True,timeout=15)
    db.execute('PRAGMA query_only=ON')
    try:
        db.execute('BEGIN')
        yield db
    finally:
        db.rollback()
        db.close()


def verify_receipt_prefix(db, receipt):
    """Verify the certified immutable handoff prefix once per SQLite witness.

    The signed receipt, trusted local SQLite schema/mutation guard and a
    persisted *local-only* verification witness allow later restarts to skip
    O(N) rehashing of indexed (origin, seq, digest) rows. The original
    signed payloads and anti-entropy index stay available. No remote witness
    or mutable count alone is ever treated as consensus authorization.
    """
    n=receipt.get('total_archived')
    expected=receipt.get('archive_digest')
    if type(n) is not int or n<0 or not isinstance(expected,str) or len(expected)!=64:
        raise PatchError('invalid handoff archive count/digest')
    row=db.execute("SELECT value FROM meta WHERE name='epoch_index_mutation'").fetchone()
    if row is None or row[0]!='0':
        raise PatchError('carry archive index was mutated after its signed append')
    guard=(db.execute('PRAGMA schema_version').fetchone()[0],row[0])
    # Never read old payload bodies here; this is a trusted local index audit.
    # A verified complete rowid prefix 1..N cannot later acquire an INSERT
    # inside it without violating the PK/trigger guard; later inserts >N do
    # not invalidate this durable handoff prefix.
    witness_key='kcia_v96_verified_carry_prefix'
    witness=db.execute('SELECT value FROM meta WHERE name=?',(witness_key,)).fetchone()
    if witness is not None:
        try:
            cached=json.loads(witness[0])
            if (cached=={'digest':expected,'count':n,'schema':guard[0],
                         'mutation':guard[1]}):
                return True
        except (ValueError,TypeError):
            pass
    sha=hashlib.sha256(); seen=0; last=0
    for rid,origin,seq,digest in db.execute(
            'SELECT rowid,origin,seq,digest FROM events WHERE rowid<=? ORDER BY origin,seq', (n,)):
        if type(rid) is not int or not 1<=rid<=n or type(seq) is not int or seq<1:
            raise PatchError('invalid archive row identity')
        sha.update(_canonical_json([origin,seq,digest])+b'\n')
        seen+=1;last=max(last,rid)
    if seen!=n or (n and last!=n) or sha.hexdigest()!=expected:
        raise PatchError('signed carry prefix changed since durable handoff receipt')
    # A rowid gap can otherwise hide a deleted record while leaving max=N.
    if n and db.execute('SELECT MIN(rowid) FROM events WHERE rowid<=?', (n,)).fetchone()[0]!=1:
        raise PatchError('archived carry prefix is not complete')
    witness_data=json.dumps({'digest':expected,'count':n,'schema':guard[0],
                             'mutation':guard[1]},sort_keys=True,separators=(',',':'))
    # The GC reader is read-only, so it can consume a prior witness but must
    # not attempt to create one. On read-only databases without a witness,
    # the full verification above remains authoritative.
    try:
        with db:
            db.execute('INSERT OR REPLACE INTO meta(name,value) VALUES(?,?)',
                       (witness_key,witness_data))
    except sqlite3.OperationalError as exc:
        if 'readonly' not in str(exc).lower() and 'read-only' not in str(exc).lower():
            raise
    return True

def _fsync_marker(path, raw):
    path=Path(path)
    if path.is_symlink() or path.exists():
        raise PatchError('refusing to overwrite existing retirement marker')
    import secrets
    tmp=path.with_name('.'+path.name+'.'+secrets.token_hex(8))
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        with os.fdopen(fd,'wb') as fh:
            fh.write(raw);fh.flush();os.fsync(fh.fileno())
        os.replace(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_unlink(path):
    path=Path(path)
    if path.is_symlink():raise PatchError('GC input became symlink')
    if path.is_file():path.unlink()
    elif path.exists():raise PatchError('unexpected non-file GC input')


def _certified_snapshot(boot, active_pointer):
    pointer=read_durable_epoch_pointer(active_pointer)
    if (pointer is None or pointer['epoch']!=boot.epoch or
            pointer['genesis']!=boot.current_genesis or
            pointer['checkpoint']!=boot.manifest_hash):
        raise PatchError('GC can only retire directly under active certified epoch pointer')
    if not boot.journal_path.is_file() or boot.journal_path.is_symlink():
        raise PatchError('no durable successor journal')
    with _read_only(boot.journal_path) as db:
        row=db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
        if row is None or row[0]!=boot.current_genesis:
            raise PatchError('successor journal identity is not durable')
    if (boot.manifest.is_symlink() or not boot.manifest.is_file()
            or boot.store.is_symlink() or not boot.store.is_dir()):
        raise PatchError('certified replacement model is not available')
    with boot.manifest.open('rb') as fh:
        digest=hashlib.file_digest(fh,'sha256').hexdigest()
    if digest!=boot.manifest_hash:
        raise PatchError('certified replacement manifest hash mismatches')
    meta=read_manifest(boot.manifest)
    if semantic_shard_sha256(meta)!=boot.proposal['semantic']:
        raise PatchError('replacement model semantic hash mismatches quorum')
    verify_shards(meta,boot.store)


def collect_certified_compact_source(boot, *, active_pointer, verify_payloads=True):
    """Retire old source, if and ONLY if the independent compact certificate,
    installed checkpoint, safe old-writer lock, complete carry and source cut
    all validate. Call in a background thread AFTER transport publication.

    ``boot`` MUST come from resolve_certified_boot() using the verified quorum
    ledger, not from an arbitrary network claim. Returns False if already gone.
    """
    old=Path(boot.source_path)
    carry=Path(boot.carry_path)
    tomb=Path(str(old)+'.retired.json')
    _certified_snapshot(boot,active_pointer)
    import fcntl
    lock_file=Path(str(old)+'.train.lock')
    if lock_file.is_symlink():raise PatchError('unsafe old writer lock')
    fd=os.open(lock_file,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (BlockingIOError,PermissionError):
            return False  # still in use by another worker
        receipt=_read_receipt(carry)
        if (receipt.get('format')!='kcia-local-carry-attestation-v1' or
                receipt.get('epoch')!=boot.epoch or
                receipt.get('source_genesis')!=boot.previous_genesis or
                receipt.get('target_genesis')!=boot.current_genesis or
                receipt.get('certified_manifest')!=boot.manifest_hash):
            raise PatchError('carry receipt does not belong to certified epoch')
        with _read_only(carry) as archive:
            archive_gen=archive.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
            if archive_gen is None or archive_gen[0]!=receipt['archive_genesis']:
                raise PatchError('carry archive has different genesis')
            verify_receipt_prefix(archive,receipt)
            if tomb.is_file():
                verify_retirement_marker(boot,receipt)
            else:
                with _read_only(old) as source:
                    row=source.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
                    if row is None or row[0]!=boot.previous_genesis:
                        raise PatchError('source journal belongs to wrong generation')
                    _count,end=verify_certified_cut(SimpleNamespace(db=source),boot.proposal['cutoffs'],boot.proposal['cut'])
                    if end!=receipt.get('source_rowid'):
                        raise PatchError('old source journal continued after certified handoff')
                    # Every original post-cut signed update must have been
                    # archived with unchanged identity and original hash.
                    for origin,seq,digest in source.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
                        if seq<=boot.proposal['cutoffs'].get(origin,0):continue
                        transferred=archive.execute('SELECT digest FROM events WHERE origin=? AND seq=?',(origin,seq)).fetchone()
                        if transferred is None or transferred[0]!=digest:
                            raise PatchError('uncovered old-genesis signed event: refusing GC')
                if verify_payloads:
                    # Verify the actual archive bytes, not only its SQLite
                    # index, before we delete the only other local copy.
                    from .event_archive import read_stored_wire
                    for wire,offset,size,n,digest in archive.execute(
                            'SELECT wire,log_offset,log_size,wire_len,digest FROM events ORDER BY rowid'):
                        read_stored_wire(carry,wire,offset,size,n,digest)
                _fsync_marker(tomb,_canonical_json(marker_for(boot,receipt)))
        # From here onward the durable certified tomb gates every restart.
        # Any process crash halfway through unlink is recoverable.
        for name in (payload_path(old),old,Path(str(old)+'-wal'),Path(str(old)+'-shm')):
            _safe_unlink(name)
        directory=os.open(old.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(directory)
        finally:os.close(directory)
        # Private local fast-resume cache refers to the retired writer only.
        # It can never authenticate another generation's model or events.
        cache=Path(str(old)+'.fast-resume')
        if cache.is_dir() and not cache.is_symlink():
            shutil.rmtree(cache)
        return True
    finally:
        os.close(fd)
