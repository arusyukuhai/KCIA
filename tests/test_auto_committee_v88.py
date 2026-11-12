from pathlib import Path
import base64
import copy
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from patchlib.auto_committee_v88 import (AutoCommittee, owner_for_local_room,
                                         validate_owner)
from patchlib.core import PatchError
from patchlib.roomid import id_from_raw_key
from patchlib.room_manifest import make_hf_manifest, validate_manifest, manifest_digest
from patchlib.generation_committee_v43 import read_generation_committee


def _peer():
    key=Ed25519PrivateKey.generate().public_key()
    raw=key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return id_from_raw_key(raw), key.public_bytes(Encoding.PEM,PublicFormat.SubjectPublicKeyInfo)


def _init(tmp):
    room_dir=tmp/'room';room_dir.mkdir()
    owner=owner_for_local_room(room_dir/'generation-owner.pem')
    manifest=make_hf_manifest('foo/model','abcd12',[{'repo':'foo/data','revision':'abcd12',
        'config':None,'split':'train','columns':{'text':'text'},'weight':1.}])
    manifest['generation']={'owner':owner}
    validate_manifest(manifest)
    (room_dir/'manifest.json').write_text(json.dumps(manifest))
    journal=tmp/'state.sqlite3'
    def inst(journal_path, manifest_path=room_dir/'manifest.json'):
        return AutoCommittee(room='A'*24,genesis='e'*64,manifest_sha=manifest_digest(manifest),
                             manifest=manifest,journal=journal_path,manifest_path=manifest_path)
    return inst,manifest


def test_owner_pins_one_durable_three_peer_roster(tmp_path):
    inst,_=_init(tmp_path)
    manager=inst(tmp_path/'node1.sqlite3')
    peers=dict(_peer() for _ in range(3))
    myself=sorted(peers)[0]
    assert manager.try_create(myself='not-a-peer',peer_pems=dict(list(peers.items())[:2])) is False
    assert manager.try_create(myself=myself,peer_pems=peers) is True
    group,pins=read_generation_committee(manager.committee_path)
    assert len(group)==3 and len(pins)==3
    assert inst(tmp_path/'node1.sqlite3').wire==manager.wire
    # A different machine, with no owner private key, learns the exact same
    # owner-signed roster from authenticated traffic.
    follower_manifest=tmp_path/'imported';follower_manifest.mkdir()
    follower=inst(tmp_path/'node2.sqlite3',follower_manifest/'manifest.json')
    assert follower.private is None
    assert follower.accept(manager.wire)
    assert follower.accept(manager.wire) is False
    assert follower.wire==manager.wire


def test_reject_tampering_fork_and_wrong_genesis(tmp_path):
    inst,_=_init(tmp_path)
    owner=inst(tmp_path/'one.sqlite3')
    peers=dict(_peer() for _ in range(3))
    assert owner.try_create(myself=next(iter(peers)),peer_pems=peers)
    wire=copy.deepcopy(owner.wire)
    wire['payload']['genesis']='f'*64
    with pytest.raises(PatchError):inst(tmp_path/'two.sqlite3').accept(wire)
    next_peer,_=_peer()
    other=copy.deepcopy(owner.wire)
    other['payload']['keys'][next_peer]=other['payload']['keys'].pop(next(iter(other['payload']['keys'])))
    with pytest.raises(PatchError):inst(tmp_path/'two.sqlite3').accept(other)
    bad=copy.deepcopy(owner.wire);bad['signature']=base64.b64encode(b'!'*64).decode('ascii')
    with pytest.raises(PatchError):inst(tmp_path/'two.sqlite3').accept(bad)
    # Persisted committee cannot be silently altered even when the public
    # roster remains valid.
    owner.committee_path.write_text('{}')
    with pytest.raises(PatchError):inst(tmp_path/'one.sqlite3')


def test_legacy_manifest_does_not_claim_owner():
    x=make_hf_manifest('foo/model','abcd12',[{'repo':'foo/data','revision':'abcd12',
        'config':None,'split':'train','columns':{'text':'text'},'weight':1.}])
    validate_manifest(x)
    with pytest.raises(PatchError):validate_owner(x)


def test_new_gui_room_enables_auto_epoch_but_legacy_room_does_not(tmp_path):
    from patchlib.webview_controller import WebviewController
    from patchlib.gui import DesktopController
    from patchlib.roomid import create_room_id
    ctl=WebviewController(tmp_path/'gui')
    sources=[{'repo':'roneneldan/TinyStories','revision':'main','config':None,
              'split':'train','columns':{'text':'text'},'weight':1.0}]
    model='sshleifer/tiny-gpt2'
    room=create_room_id()
    desc=ctl.save({'room':room,'model':model,'revision':'main',
                   'task':'causal-lm','seed':42,'sources':sources})
    assert 'generation' in desc['manifest']
    key=ctl._room_path(room).parent/'generation-owner.pem'
    assert key.is_file() and (key.stat().st_mode & 0o077)==0
    base=tmp_path/'genesis.fstate';base.write_bytes(b'fake initial path for CLI arg test')
    def args(room,manifest):
        return DesktopController.build_command(
            room=room,model=model,revision='main',base=base,manifest=manifest,
            ws='ws://127.0.0.1:8765/ws',task='causal-lm',kind='loha',rank=16,
            journal=tmp_path/'test.sqlite3')
    assert '--generation-auto-v88' in args(room,ctl._room_path(room))
    legacy_room=create_room_id()
    legacy_path=ctl._room_path(legacy_room)
    legacy_path.parent.mkdir(parents=True)
    legacy=make_hf_manifest(model,'main',sources,architecture='causal-lm',seed=42)
    legacy_path.write_text(json.dumps(legacy))
    assert '--generation-auto-v88' not in args(legacy_room,legacy_path)
    assert not (legacy_path.parent/'generation-owner.pem').exists()
