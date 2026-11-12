"""Safe restart from a certified v52 live trainer pointer.

Runs BEFORE opening a new-generation optimizer or accepting peer traffic.
Only locally complete, fully verified data-only checkpoints are accepted.
The old-generation carry archive must have a durable final handoff receipt.
Neither a remote generation-head hint nor a naked pointer is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .core import PatchError
from .live_epoch_switch_v48 import certified_genesis,verify_candidate_files
from .epoch_live_rebase_v52 import verify_certified_cut


@dataclass(frozen=True)
class CertifiedEpochBoot:
    epoch: int
    current_genesis: str
    previous_genesis: str
    manifest_hash: str
    proposal: dict
    manifest: Path
    store: Path
    journal_path: Path
    carry_path: Path
    source_path: Path


def read_durable_epoch_pointer(path: Path):
    path=Path(path)
    if path.is_symlink():
        raise PatchError('unsafe committed epoch pointer symlink')
    if not path.exists():return None
    if not path.is_file() or path.stat().st_size > 4096:
        raise PatchError('unsafe committed epoch pointer')
    try:data=json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise PatchError('corrupt committed epoch pointer') from exc
    def hex64(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
    if (not isinstance(data,dict) or set(data)!={'format','epoch','genesis','checkpoint'}
            or data['format']!='kagami-hot-pointer-v1' or
            type(data['epoch']) is not int or not 1<=data['epoch']<2**32 or
            not hex64(data['genesis']) or not hex64(data['checkpoint'])):
        raise PatchError('invalid committed epoch pointer')
    return data


def resolve_certified_boot(*, root:Path, journal_base:Path, ledger,
                           root_genesis:str, max_state_bytes=2*1024**4):
    """Resolve local bytes for a committed epoch, refusing missing quorum links.

    Expected disk layout is stable across process restart. A staging worker
    stores hardlinked content-addressed checkpoints under
    ``root/checkpoints/views/<manifest-hash>``.
    """
    root=Path(root);journal_base=Path(journal_base)
    pointer=read_durable_epoch_pointer(root/'active-epoch.json')
    if pointer is None:return None
    epoch=pointer['epoch']
    cert=ledger.certified_epoch(epoch)
    if cert is None:
        raise PatchError('active epoch pointer has no locally verified quorum certificate')
    p=cert['proposal']
    if (p.get('epoch')!=epoch or p.get('manifest')!=pointer['checkpoint'] or
            certified_genesis(p)!=pointer['genesis']):
        raise PatchError('pointer does not identify the certified epoch')
    prev_cert=ledger.certified_epoch(epoch-1) if epoch>1 else None
    if epoch>1 and prev_cert is None:
        raise PatchError('missing prior epoch certificate for carry validation')
    previous=certified_genesis(prev_cert['proposal']) if prev_cert else root_genesis
    if p.get('parent') != (ledger.genesis if epoch==1 else __digest(prev_cert['proposal'])):
        raise PatchError('broken certified epoch parent chain')
    view=root/'checkpoints'/'views'/p['manifest']
    manifest=view/'checkpoint.fmanifest'
    verify_candidate_files(SimpleNamespace(manifest=manifest,store=view),p,
                           max_state_bytes=max_state_bytes)
    return CertifiedEpochBoot(epoch,pointer['genesis'],previous,p['manifest'],p,
        manifest,view,root/'journals'/f'epoch-{epoch:08d}.sqlite3',
        root/'carries'/f'epoch-{epoch:08d}.sqlite3',
        journal_base if epoch==1 else root/'journals'/f'epoch-{epoch-1:08d}.sqlite3')


def __digest(p):
    from .online_generation import _digest
    return _digest(p)


def _frozen_source_cut_with_witness(old_journal, boot, receipt):
    """Cache an already-verified certified old-generation cut locally.

    A certified source has a frozen writer at handoff. Its append-only index
    has a durable mutation witness. If schema, mutation counter, or MAX(rowid)
    changes, recompute the full signed digest rather than trusting the cache.
    This is a local speed hint only: the quorum certificate and original
    receipts must still be verified independently on every boot.
    """
    from .core import _canonical_json
    import json
    from .epoch_live_rebase_v52 import verify_certified_cut
    db=old_journal.db
    guard=old_journal.epoch_cut_guard()
    max_rowid=db.execute('SELECT COALESCE(MAX(rowid),0) FROM events').fetchone()[0]
    key='kagami_v96_frozen_source_cut_witness'
    expected={'source_genesis':boot.previous_genesis,
              'target_genesis':boot.current_genesis,
              'manifest':boot.manifest_hash,
              'source_rowid':receipt['source_rowid'],
              'cut':boot.proposal['cut'],
              'cutoffs_digest':hashlib.sha256(_canonical_json(sorted(
                    boot.proposal['cutoffs'].items()))).hexdigest(),
              'schema_guard':list(guard) if guard is not None else None,
              'current_max_rowid':max_rowid}
    if guard is not None:
        prior=db.execute('SELECT value FROM meta WHERE name=?',(key,)).fetchone()
        if prior is not None:
            try:
                if json.loads(prior[0])==expected:
                    return max_rowid
            except (ValueError,TypeError):
                pass
    _,actual_max=verify_certified_cut(old_journal,
        boot.proposal['cutoffs'],boot.proposal['cut'])
    if actual_max!=receipt['source_rowid']:
        raise PatchError('old journal changed after committed epoch handoff')
    if guard is not None:
        with db:
            db.execute('INSERT OR REPLACE INTO meta(name,value) VALUES(?,?)',
                       (key,json.dumps(expected,sort_keys=True,separators=(',',':'))))
    return actual_max


def verify_boot_carry(boot:CertifiedEpochBoot, archive, *, old_journal):
    """Reject restart if the source writer made any update after handoff.

    This is a *local* recovery gate, not a proof that every offline peer has
    migrated. Their old signed events still require carry-lane anti-entropy.
    """
    path=Path(str(archive.journal.path)+'.handoff.json')
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4096:
        raise PatchError('missing durable final carry receipt')
    try:receipt=json.loads(path.read_bytes())
    except (ValueError,UnicodeError) as exc:
        raise PatchError('malformed final carry receipt') from exc
    if (not isinstance(receipt,dict) or receipt.get('format')!='kagami-local-carry-attestation-v1'
            or receipt.get('epoch')!=boot.epoch or
            receipt.get('source_genesis')!=boot.previous_genesis or
            receipt.get('target_genesis')!=boot.current_genesis or
            receipt.get('certified_manifest')!=boot.manifest_hash or
            receipt.get('archive_genesis')!=archive.genesis):
        raise PatchError('durable carry receipt mismatches certified epoch')
    if type(receipt.get('source_rowid')) is not int or receipt['source_rowid']<0:
        raise PatchError('invalid certified old-writer boundary')
    # V88: original handoff archive is an immutable prefix. Signed old-genesis
    # updates that arrive *after* the epoch handoff are appended at later
    # rowids. These must still survive restart and be replayed exactly once.
    from .core import _canonical_json
    from .compact_journal_gc_v88 import verify_receipt_prefix, verify_retirement_marker
    if old_journal is None:
        tomb = Path(str(boot.source_path)+'.retired.json')
        legacy={'epoch':boot.epoch,'manifest':boot.manifest_hash,
                'new_journal':str(boot.journal_path.resolve())}
        if (tomb.is_file() and not tomb.is_symlink() and
                tomb.read_bytes()==_canonical_json(legacy)):
            # Previous version supported retirement only when there were no
            # uncaptured old-genesis deltas. Preserve old safe restarts.
            if boot.source_path.exists() or boot.source_path.is_symlink() or archive.journal.count()!=0:
                raise PatchError('retired source missing trustworthy GC tombstone or carry empty proof')
        else:
            verify_retirement_marker(boot,receipt)
    else:
        old_max=_frozen_source_cut_with_witness(old_journal,boot,receipt)
        if old_max!=receipt['source_rowid']:
            raise PatchError('old journal changed after committed epoch handoff')
    verify_receipt_prefix(archive.journal.db,receipt)
    return True
