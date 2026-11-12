"""Pre-certificate snapshot *staging* gate for crash-fault epoch voting.

A quorum cannot sign a candidate without independently verifying its shards.
Request/serve one pending snapshot only if a pinned signer authenticated a
candidate endorsement AND a fenced trainer-export receipt. Neither of these
is a quorum certificate: caller may stage bytes but MUST NOT adopt weights,
change the active generation or GC any journal until full quorum certification.
"""
from __future__ import annotations
import json
from .core import PatchError
from .online_generation import _digest, _verify
from .epoch_autonomous_v56 import PROTOCOL as ENDORSE_PROTOCOL


def verified_pending_candidate(control, ledger, epoch, manifest):
    if control is None or ledger is None or type(epoch) is not int or epoch < 1:
        return None
    if not isinstance(manifest, str) or len(manifest)!=64 or any(c not in '0123456789abcdef' for c in manifest):
        return None
    # No unbounded row traversal: AutonomousPagedEpoch allows one candidate
    # per epoch; verify it still links the certificate ledger's current head.
    row=control.db.execute('SELECT wire FROM proposals WHERE epoch=? LIMIT 1',(epoch,)).fetchone()
    if row is None:return None
    try:
        p=json.loads(row[0])
        if p.get('manifest')!=manifest:return None
        control._validate_proposal(p)
        digest=_digest(p)
        raw=control.db.execute('SELECT wire FROM endorsements WHERE digest=?',(digest,)).fetchone()
        if raw is None:return None
        signed=json.loads(raw[0]);payload=_verify(signed,ledger.trusted)
        if (payload!={'protocol':ENDORSE_PROTOCOL,'kind':'candidate','digest':digest,
                     'epoch':epoch,'room':ledger.room} or
                signed['peer'] not in ledger.committee or signed['peer'] not in p['replicas']):
            return None
        if not control.has_export_receipt(p):return None
        return p
    except (PatchError,ValueError,TypeError,KeyError,AttributeError):
        return None
