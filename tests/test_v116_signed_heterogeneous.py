"""Signed small/big rank peers converge without rank padding."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
import torch
from torch import nn
from patchlib.core import UniversalAdapter, decode_patch, encode_patch, PatchError
from patchlib.elastic import ElasticEngine, EventJournal, unpack_event
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem


def test_signed_mixed_physical_ranks_additive_and_replay(tmp_path):
    torch.manual_seed(198)
    base = nn.Sequential(nn.Linear(64,64,bias=False))
    genesis = hashlib.sha256(b'immutable fixture').hexdigest()
    ranks = {'mac':4,'colab':16,'late':8}
    adapters = {p:UniversalAdapter(deepcopy(base),'fixture','sha256:frozen',kind='loha',rank=16,
                  local_rank=rank,target_scope='linear') for p,rank in ranks.items()}
    assert len({ad.layout for ad in adapters.values()})==1
    signer={}; trusted={}
    for peer in ranks:
        secret=tmp_path/(peer+'.key');public=tmp_path/(peer+'.pub')
        generate_keypair(secret, public)
        signer[peer]=signer_from_pem(secret)
        trusted[peer]=verifier_from_pem(public)
    journals={peer:EventJournal(tmp_path/(peer+'.db'),genesis,max_bytes=10**7,min_free_mib=0)
              for peer in ranks}
    engines={peer:ElasticEngine(adapters[peer],peer,genesis,journals[peer],signer=signer[peer],trusted=trusted)
             for peer in ranks}
    try:
        wires=[]
        for peer,value in [('mac',.045),('colab',.032)]:
            with torch.no_grad():
                for g in adapters[peer].groups:
                    for part in g.overlay.local.parts:part.fill_(value)
            wires.append(engines[peer].seal_local(2))
        assert len(wires[0]) < len(wires[1])
        assert unpack_event(wires[0],trusted)[1]['targets'][0]['rank']==4
        assert unpack_event(wires[1],trusted)[1]['targets'][0]['rank']==16
        assert engines['mac'].ingest(wires[1]);assert engines['colab'].ingest(wires[0])
        # A new peer with a third physical rank must accept both old events.
        assert engines['late'].ingest(wires[1])
        assert engines['late'].ingest(wires[0])
        def raw(peer):
            return adapters[peer].model[0].parametrizations.weight.original.detach().clone()
        for peer in ('colab','late'):
            assert torch.allclose(raw(peer),raw('mac'),atol=1e-5)
        assert not engines['late'].ingest(wires[1])
        # Signed wire cannot be modified even if a malicious peer edits rank.
        mutated=bytearray(wires[0]);mutated[-12]^=1
        with pytest.raises(PatchError):engines['late'].ingest(bytes(mutated))
        restart=UniversalAdapter(deepcopy(base),'fixture','sha256:frozen',kind='loha',rank=16,
                                 local_rank=2,target_scope='linear')
        restored=ElasticEngine(restart,'late',genesis,journals['late'],
                               signer=signer['late'],trusted=trusted)
        restored.replay()
        out=restart.model[0].parametrizations.weight.original.detach()
        assert torch.allclose(out,raw('mac'),atol=1e-5)
    finally:
        for j in journals.values():j.close()


def test_invalid_factor_rank_metadata_is_refused():
    a=UniversalAdapter(nn.Sequential(nn.Linear(32,32,bias=False)), 'fixture','revision',
                       kind='loha',rank=16,local_rank=4,target_scope='linear')
    b=UniversalAdapter(nn.Sequential(nn.Linear(32,32,bias=False)), 'fixture','revision',
                       kind='loha',rank=16,local_rank=16,target_scope='linear')
    meta,parts=decode_patch(a.export('source',1))
    meta['targets'][0]['rank']=8
    with pytest.raises(PatchError):b.import_remote(encode_patch(meta,parts))
    meta['targets'][0]['rank']=4
    meta.pop('rank_protocol')
    with pytest.raises(PatchError):b.import_remote(encode_patch(meta,parts))
