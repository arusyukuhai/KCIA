"""V90 pinning and proof regression tests: 1/1, 2/2, 2/3, never transient reconfiguration."""
import copy
import hashlib
import sqlite3

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_public_key

from patchlib.auto_committee_v88 import bootstrap_ready, PROTOCOL, PROTOCOL_V88
from patchlib.core import PatchError
from patchlib.epoch_certificate_v46 import CompactEpochLedger, proposal_v2
from patchlib.frontier_commitment_v44 import build_frontier_pages
from patchlib.generation_committee_v43 import read_generation_committee
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from patchlib.online_generation import GenerationLedger, make_proposal
from patchlib.roomid import id_from_raw_key
from test_auto_committee_v88 import _init, _peer


@pytest.mark.parametrize('n', [1, 2, 3])
def test_new_roster_is_pinned_and_crash_recoverable(tmp_path, n):
    inst, _ = _init(tmp_path)
    manager = inst(tmp_path/'events.db')
    peers = dict(_peer() for _ in range(n))
    myself = min(peers)
    assert manager.try_create(myself=myself, peer_pems=peers)
    assert manager.wire['payload']['protocol'] == (PROTOCOL_V88 if n==3 else PROTOCOL)
    members,pins = read_generation_committee(manager.committee_path)
    assert len(members) == len(pins) == n
    assert inst(tmp_path/'events.db').wire == manager.wire
    assert not manager.try_create(myself=myself, peer_pems=peers)
    # Owner CANNOT silently change the certificate by adding another peer.
    another=dict(peers); another.update([_peer()])
    assert not manager.try_create(myself=myself, peer_pems=another)
    replica_dir=tmp_path/'follower';replica_dir.mkdir()
    replica=inst(tmp_path/'follower.db',replica_dir/'room.json')
    assert replica.accept(manager.wire)
    assert not replica.accept(manager.wire)
    more=copy.deepcopy(manager.wire)
    more['payload']['keys']['tamper']=_peer()[1].decode('ascii')
    with pytest.raises(PatchError):replica.accept(more)


def test_v88_protocol_rejects_two_member_downgrade(tmp_path):
    inst,_=_init(tmp_path)
    mgr=inst(tmp_path/'journal.db')
    peers=dict(_peer() for _ in range(2))
    assert mgr.try_create(myself=next(iter(peers)),peer_pems=peers)
    payload=dict(mgr.wire['payload'],protocol=PROTOCOL_V88)
    malformed={'payload':payload,'signature':mgr.wire['signature']}
    with pytest.raises(PatchError):inst(tmp_path/'other.db').accept(malformed)


def test_bootstrap_settling_and_no_premature_solo():
    assert not bootstrap_ready(online_count=1,elapsed=15,stable=15)
    assert not bootstrap_ready(online_count=2,elapsed=20,stable=0)
    assert not bootstrap_ready(online_count=2,elapsed=25,stable=11)
    assert bootstrap_ready(online_count=1,elapsed=20,stable=12)
    assert bootstrap_ready(online_count=2,elapsed=25,stable=12)
    assert bootstrap_ready(online_count=3,elapsed=0,stable=0)


def _auth_peers(tmp_path,n):
    signers={};verifiers={};members=[]
    for i in range(n):
        prv=tmp_path/f'{i}.pem';pub=tmp_path/f'{i}.pub'
        generate_keypair(prv,pub)
        peer=id_from_raw_key(load_pem_public_key(pub.read_bytes()).public_bytes(Encoding.Raw,PublicFormat.Raw))
        members.append(peer)
        signers[peer]=signer_from_pem(prv)
        verifiers[peer]=verifier_from_pem(pub)
    return sorted(members),signers,verifiers


@pytest.mark.parametrize('n,needed',[(1,1),(2,2),(3,2)])
def test_real_compact_certificate_minimum_quorum(tmp_path,n,needed):
    """The canonical compact signature path must accept exact quorum, reject quorum-1."""
    members,signers,trusted=_auth_peers(tmp_path,n)
    ledgers={peer:CompactEpochLedger(tmp_path/f'c{i}.sqlite3',room='room',peer=peer,
               committee=members,signer=signers[peer],trusted=trusted,genesis='a'*64)
             for i,peer in enumerate(members)}
    header,_=build_frontier_pages({})
    p=proposal_v2(room='room',epoch=1,parent='a'*64,manifest='b'*64,
                  semantic='c'*64,inventory='d'*64,frontier_header=header,
                  replicas=members)
    votes=[]
    from patchlib.online_generation import _sign,_digest
    for peer in members[:needed]:
        votes.append(_sign({'protocol':'kcia-compact-epoch-certificate-v2','kind':'vote',
                            'room':'room','epoch':1,'proposal':_digest(p)},peer,signers[peer]))
    if needed>1:
        with pytest.raises(PatchError):ledgers[members[0]].certificate(p,votes[:-1])
    cert=ledgers[members[0]].certificate(p,votes)
    for ledger in ledgers.values():
        assert ledger.accept(cert)
        assert ledger.certified_epoch(1)==cert
        ledger.close()


@pytest.mark.parametrize('n,needed',[(1,1),(2,2),(3,2)])
def test_v1_certificate_requires_exact_pinned_quorum(tmp_path,n,needed):
    members,signers,trusted=_auth_peers(tmp_path,n)
    ledgers={peer:GenerationLedger(tmp_path/f'v1-{i}.sqlite3',room='room',peer=peer,
                 committee=members,signer=signers[peer],trusted=trusted,genesis='a'*64)
             for i,peer in enumerate(members)}
    p=make_proposal(room='room',epoch=1,parent='a'*64,manifest='b'*64,
                    semantic='c'*64,cut='d'*64,cutoffs={},replicas=members)
    votes=[ledgers[peer].vote(p,verified_snapshot=True,verified_cut=True)
           for peer in members[:needed]]
    if needed>1:
        with pytest.raises(PatchError):ledgers[members[0]].certificate(p,votes[:-1])
    cert=ledgers[members[0]].certificate(p,votes)
    for ledger in ledgers.values():
        assert ledger.accept(cert)
        assert ledger.certified_epoch(1)==cert
        ledger.close()


def test_existing_two_member_committee_must_not_downgrade_when_peer_disappears(tmp_path):
    members,signers,trusted=_auth_peers(tmp_path,2)
    ledger=CompactEpochLedger(tmp_path/'stable.sqlite3',room='room',peer=members[0],
                 committee=members,signer=signers[members[0]],trusted=trusted,genesis='a'*64)
    header,_=build_frontier_pages({})
    proposal=proposal_v2(room='room',epoch=1,parent='a'*64,manifest='b'*64,
                  semantic='c'*64,inventory='d'*64,frontier_header=header,replicas=members)
    from patchlib.online_generation import _sign,_digest
    only_owner=_sign({'protocol':'kcia-compact-epoch-certificate-v2','kind':'vote',
                      'room':'room','epoch':1,'proposal':_digest(proposal)},
                     members[0],signers[members[0]])
    with pytest.raises(PatchError):ledger.certificate(proposal,[only_owner])
    ledger.close()
    # Also cannot reopen the durable ledger with a different configured committee.
    with pytest.raises(PatchError):
        CompactEpochLedger(tmp_path/'stable.sqlite3',room='room',peer=members[0],
              committee=[members[0]],signer=signers[members[0]],trusted=trusted,genesis='a'*64)

@pytest.mark.parametrize('n', [1, 2])
def test_real_snapshot_and_verified_cut_can_certify_small_committee(tmp_path,n):
    """Exercise the exporter and local Merkle/shard rehash, not fake verified flags."""
    import threading
    import torch
    from patchlib.epoch_export_v57 import export_at_step_boundary
    members,signers,trusted=_auth_peers(tmp_path,n)
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT, PRIMARY KEY(origin,seq))')
    db.execute('INSERT INTO events VALUES(?,?,?)',(members[0],1,hashlib.sha256(b'event').hexdigest()))
    class Source:
        def __init__(self,conn):self.db=conn
        def count(self):return self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    model=torch.nn.Linear(3,2)
    from types import SimpleNamespace
    adapter=SimpleNamespace(model_id='tiny',revision='v1',groups=[],lock=threading.RLock())
    p,pages,receipt=export_at_step_boundary(journal=Source(db),model=model,adapter=adapter,
        store=tmp_path/'blobs',incoming=tmp_path/'incoming',room='room',epoch=1,
        parent='e'*64,replicas=members,signer=signers[members[0]],peer=members[0],
        max_state_bytes=1024**2,shard_bytes=65536)
    actual=tmp_path/'incoming'/(p['manifest']+'.fmanifest')
    assert actual.is_file()
    votes=[]; ledgers=[]
    for i,peer in enumerate(members):
        ledger=CompactEpochLedger(tmp_path/f'verified-{i}.sqlite3',room='room',peer=peer,
             committee=members,signer=signers[peer],trusted=trusted,genesis='e'*64)
        votes.append(ledger.vote(p,pages=pages,journal_db=db,manifest_path=actual,shard_store=tmp_path/'blobs'))
        ledgers.append(ledger)
    cert=ledgers[0].certificate(p,votes)
    for ledger in ledgers:
        assert ledger.accept(cert)
        assert ledger.certified_epoch(1) == cert
        ledger.close()
    db.close()

@pytest.mark.parametrize('n', [1, 2, 3])
def test_journal_status_reports_small_signed_roster(tmp_path,n):
    from patchlib.journal_recovery_status_v89 import inspect
    inst,_=_init(tmp_path)
    journal=tmp_path/'root.sqlite3'
    owner=inst(journal)
    peers=dict(_peer() for _ in range(n))
    assert owner.try_create(myself=min(peers),peer_pems=peers)
    status=inspect(journal)
    assert status['signed_roster_present']
    assert status['committee_size']==n
    assert status['committee_quorum']==n//2+1


def test_prevote_snapshot_can_be_requested_from_signed_exporter(tmp_path):
    """2/2 must transfer shards BEFORE the second voter can sign, not after quorum."""
    from patchlib.epoch_export_v57 import RECEIPT_PROTOCOL
    from patchlib.online_generation import _sign,_digest
    from patchlib.pending_snapshot_v90 import verified_pending_candidate
    from tests.test_epoch_v56_autonomy import make_network,deliver
    p,pages,db,old,controls,ledgers,bridges=make_network(tmp_path,origins=2)
    try:
        # One peer exported a snapshot. The second peer has *no* snapshot,
        # which is the real bootstrap scenario. No cert exists yet.
        controls['b'].local_checkpoint=lambda _:None
        bridges['a'].snapshot_available=lambda sha:sha==p['manifest']
        bridges['b'].snapshot_available=lambda sha:False
        controls['a'].offer(p,pages)
        assert ledgers['a'].latest() is None
        assert verified_pending_candidate(controls['a'],ledgers['a'],1,p['manifest']) is None
        receipt=_sign({'protocol':RECEIPT_PROTOCOL,'room':'room','epoch':1,
                       'proposal':_digest(p),'manifest':p['manifest'],
                       'inventory':p['inventory'],'frontier_digest':p['frontier_digest']},
                      'a',old['a'].signer)
        controls['a'].receive_export_receipt(receipt,p)
        assert verified_pending_candidate(controls['a'],ledgers['a'],1,p['manifest'])==p
        from tests.test_v44_low_cost_generation import DummyNode
        node=DummyNode();bridges['a'].tick([('b',node)],now=100)
        assert any(kind=='generation-compact-have' and fields['manifest']==p['manifest']
                   for kind,fields in node.wires)
        for kind,fields in node.wires:
            bridges['b'].receive('a',kind,fields)
        assert ledgers['b'].latest() is None
        pending=DummyNode();bridges['b'].tick([('a',pending)],now=101)
        assert any(kind=='generation-compact-snapshot-request' and
                   fields['manifest']==p['manifest'] for kind,fields in pending.wires)
        assert verified_pending_candidate(controls['b'],ledgers['b'],1,p['manifest'])==p
        assert verified_pending_candidate(controls['b'],ledgers['b'],1,'f'*64) is None
    finally:
        for x in controls.values():x.close()
        for x in ledgers.values():x.close()
        for x in old.values():x.close()
        db.close()
