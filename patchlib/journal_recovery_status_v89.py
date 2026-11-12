"""Read-only signed journal/automatic epoch readiness status.

No model or journal writes. Never infer a certificate from the mere existence of
snapshot files; the status is explanatory, not an authorization to delete logs.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .disk_health_v86 import _signed_log_quota, free_bytes, GIB


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError, OSError):
        return None


def _sql_count(path, sql):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.5) as db:
            db.execute('PRAGMA query_only=ON')
            return int(db.execute(sql).fetchone()[0])
    except (sqlite3.Error,ValueError,OSError,TypeError):
        return None


def _pending_vote_detail(path, journal):
    """Read-only diagnostic; never authorizes a log retirement."""
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=.5) as db:
            db.execute('PRAGMA query_only=ON')
            row=db.execute('SELECT digest,epoch,wire FROM proposals ORDER BY epoch DESC LIMIT 1').fetchone()
            if row is None:
                return None
            digest,epoch,raw=row
            proposal=json.loads(raw)
            manifest_sha=proposal['manifest']
            if (not isinstance(manifest_sha,str) or len(manifest_sha)!=64 or
                any(c not in '0123456789abcdef' for c in manifest_sha)):
                return None
            total=db.execute('SELECT COUNT(*) FROM votes WHERE digest=?', (digest,)).fetchone()[0]
            pages=db.execute('SELECT COUNT(*) FROM pages WHERE digest=?', (digest,)).fetchone()[0]
            receipt=db.execute('SELECT 1 FROM exporter_receipts WHERE digest=?', (digest,)).fetchone() is not None
            manifest_path=Path(str(journal)+'.snapshots')/'incoming'/(manifest_sha+'.fmanifest')
            return {'epoch':epoch, 'votes':total, 'pages':pages,
                    'needed_pages':proposal['frontier_header']['pages'],
                    'receipt':receipt,'manifest_present':manifest_path.is_file() and not manifest_path.is_symlink()}
    except (sqlite3.Error,OSError,ValueError,KeyError,TypeError):
        return None


def inspect(journal, manifest=None):
    journal=Path(journal).expanduser().absolute()
    if manifest is None:
        # GUI profile path: ROOM/peers/peer-*/events.sqlite3
        if journal.name=='events.sqlite3' and journal.parent.parent.name=='peers':
            manifest=journal.parent.parent.parent/'manifest.json'
    descriptor=_read_json(manifest) if manifest is not None else None
    generation=descriptor.get('generation') if isinstance(descriptor,dict) else None
    owner_pinned=(isinstance(generation,dict) and isinstance(generation.get('owner'),str))
    roster=_read_json(str(journal)+'.generation-auto-v88.json')
    payload=roster.get('payload') if isinstance(roster,dict) else None
    keys=payload.get('keys') if isinstance(payload,dict) else None
    signed_roster=isinstance(keys,dict) and 1<=len(keys)<=3
    pointer=_read_json(str(journal)+'.live-epoch/active-epoch.json')
    stage=Path(str(journal)+'.generation-compact-stage.sqlite3')
    certified=Path(str(journal)+'.generation-compact.sqlite3')
    stats=_signed_log_quota(journal)
    return {
        'journal':str(journal),'manifest':str(manifest) if manifest is not None else None,
        'owner_pinned':owner_pinned,'signed_roster_present':signed_roster,
        'committee_size':len(keys) if isinstance(keys,dict) else 0,
        'committee_quorum':len(keys)//2+1 if signed_roster else None,
        'active_epoch':pointer.get('epoch') if isinstance(pointer,dict) else None,
        'signed_log_usage':stats,
        'certified_epoch_count':_sql_count(certified,'SELECT COUNT(*) FROM commits'),
        'pending_proposals':_sql_count(stage,'SELECT COUNT(*) FROM proposals'),
        'local_voted_proposals':_sql_count(stage,'SELECT COUNT(*) FROM votes'),
        'export_receipts':_sql_count(stage,'SELECT COUNT(*) FROM exporter_receipts'),
        'frontier_pages':_sql_count(stage,'SELECT COUNT(*) FROM pages'),
        'signed_endorsements':_sql_count(stage,'SELECT COUNT(*) FROM endorsements'),
        'physical_free':free_bytes(journal),
        'latest_candidate':_pending_vote_detail(stage,journal),
    }


def report(journal,manifest=None):
    s=inspect(journal,manifest)
    print('KAGAMI JOURNAL / CERTIFIED EPOCH STATUS (read only)')
    print('journal:',s['journal'])
    print(f'physical free: {s["physical_free"]/GIB:.2f} GiB')
    if s['signed_log_usage']:
        used,cap,count=s['signed_log_usage']
        print(f'signed log: {used/GIB:.3f} GiB / '+
              (f'{cap/GIB:.3f} GiB ({100*used/max(1,cap):.1f}%)' if cap else 'no recorded cap')+
              f', {count} indexed events')
    else:
        print('signed log: missing/unreadable')
    print('GUI pinned generation owner:',s['owner_pinned'])
    print('owner-signed 1..3-peer roster:',s['signed_roster_present'],
          f"members={s['committee_size']} required={s['committee_quorum']}")
    print('proposals:',s['pending_proposals'],'local votes:',s['local_voted_proposals'],
          'certificates:',s['certified_epoch_count'],'active epoch:',s['active_epoch'])
    if s['latest_candidate'] is not None:
        c=s['latest_candidate']
        print(f'latest candidate epoch={c["epoch"]}: local recorded votes={c["votes"]}, '
              f'pages={c["pages"]}/{c["needed_pages"]}, '
              f'export receipt={c["receipt"]}, checkpoint manifest present={c["manifest_present"]}')
    print('checkpoint export receipts:',s['export_receipts'],
          'frontier pages:',s['frontier_pages'],
          'signed endorsements:',s['signed_endorsements'])
    if (s['signed_roster_present'] and s['committee_size']==1 and
            s['certified_epoch_count']==0):
        if not s['pending_proposals']:
            print('SOLO STAGE: no durable checkpoint candidate. The exporter may be '
                  'failing before publication; v95 retries even while journal pressure is held.')
        elif not s['export_receipts'] or not s['signed_endorsements']:
            print('SOLO STAGE: incomplete durable export or endorsement. '
                  'No other peer can repair this automatically; inspect V95 export errors.')
        elif not s['local_voted_proposals']:
            print('SOLO STAGE: checkpoint candidate exists but its local signature '
                  'is missing. Inspect compact committee verification logs.')
        else:
            print('SOLO STAGE: local vote exists but certificate not committed. '
                  'Inspect compact vote acceptance and certificate logs.')
    if not s['owner_pinned']:
        print('BLOCKER: this journal is not identifiable as a new owner-pinned v88 GUI room. '
              'Existing room manifests are intentionally not upgraded in place.')
    elif not s['signed_roster_present']:
        print('BLOCKER: no durable owner-signed committee roster. The room creator '
              'must authenticate peers and complete the initial discovery window (solo is supported).')
    elif not s['certified_epoch_count']:
        print('BLOCKER: no confirmed epoch certificate. A signed export, complete '
              "checkpoint transfer, and the pinned committee's real quorum votes are needed (1/1, 2/2, or 2/3). "
              'Examine COMPACT EPOCH CANDIDATE / compact verification logs.')
    elif not s['active_epoch']:
        print('BLOCKER: certificate exists but the verified model/optimizer/carry '
              'activation has not committed. Inspect LIVE EPOCH and prewarm logs.')
    else:
        print('Active epoch pointer exists; if old log remains, inspect V88 COMPACT GC '
              'and recovery messages. Never manually delete a live journal.')
    print('CAUTION: this is advisory evidence only, not a cryptographic verification of '
          'the committee, checkpoint, or certificate. No data was changed.')
    return s
