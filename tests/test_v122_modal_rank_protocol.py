"""v122 regression: actual continual heterogeneous signed events + keyboard/scrollable editor."""
from copy import deepcopy
import hashlib
import math

import pytest
import torch
from torch import nn

from patchlib.core import UniversalAdapter, decode_patch, FrozenDelta
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal, unpack_event
from patchlib.factor_ledger_v62 import _factor_tile
from patchlib.livefed import _delta_block
from patchlib.webview_gui import compose_html
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem


def test_real_continual_low_rank_signed_protocol_and_replay(tmp_path):
    torch.manual_seed(88)
    base=nn.Sequential(nn.Linear(16,16,bias=False))
    base_a=deepcopy(base);base_b=deepcopy(base)
    small=UniversalAdapter(base_a,'fixture','commit',kind='loha',rank=8,local_rank=2,target_scope='linear')
    big=UniversalAdapter(base_b,'fixture','commit',kind='loha',rank=8,local_rank=8,target_scope='linear')
    private=tmp_path/'mac.key';public=tmp_path/'mac.pub'
    generate_keypair(private,public)
    signer=signer_from_pem(private)
    trusted={'mac':verifier_from_pem(public)}
    ja=EventJournal(tmp_path/'mac.db','a'*64,max_bytes=10_000_000,min_free_mib=0)
    jb=EventJournal(tmp_path/'colab.db','a'*64,max_bytes=10_000_000,min_free_mib=0)
    ea=ContinualElasticEngine(small,'mac','a'*64,ja,unsafe=False,signer=signer,trusted=trusted)
    eb=ContinualElasticEngine(big,'colab','a'*64,jb,unsafe=False,trusted=trusted)
    try:
        with torch.no_grad():
            for group in small.groups:
                for i,p in enumerate(group.overlay.local.parts):
                    p.fill_(.1 if i<2 else .06)
        # This is the actual path used by the GUI's continual signed FedSum,
        # not UniversalAdapter.export's legacy non-continual path.
        wire=ea.seal_local(1)
        _,meta,weights,_=unpack_event(wire,trusted,unsafe=False)
        assert meta['rank_protocol']=='heterogeneous-loha-v1'
        assert meta['targets'][0]['rank']==2
        assert len([k for k in weights if '.f' in k])==4
        assert eb.ingest(wire)
        # Replay the SAME authenticated original wire at the same recipient rank.
        ref=base_b[0].parametrizations.weight.original.detach().clone()
        fresh=UniversalAdapter(deepcopy(base),'fixture','commit',kind='loha',rank=8,local_rank=4,target_scope='linear')
        jc=EventJournal(tmp_path/'late.db','a'*64,max_bytes=10_000_000,min_free_mib=0)
        ec=ContinualElasticEngine(fresh,'late','a'*64,jc,unsafe=False,trusted=trusted)
        try:
            assert ec.ingest(wire)
            assert torch.allclose(ref,fresh.model[0].parametrizations.weight.original,atol=1e-6)
        finally: jc.close()
        # Replay is a startup-only API: reconstruct a pristine adapter and
        # reopen the durable journal rather than replaying twice in one engine.
        restored=UniversalAdapter(deepcopy(base),'fixture','commit',kind='loha',rank=8,local_rank=4,target_scope='linear')
        jd=EventJournal(tmp_path/'late.db','a'*64,max_bytes=10_000_000,min_free_mib=0)
        try:
            ed=ContinualElasticEngine(restored,'late','a'*64,jd,unsafe=False,trusted=trusted)
            ed.replay()
            assert torch.allclose(ref,restored.model[0].parametrizations.weight.original,atol=1e-6)
        finally: jd.close()
    finally: ja.close();jb.close()


def test_sender_rank_controls_all_factor_reconstruction_methods():
    model=nn.Sequential(nn.Linear(12,12,bias=False))
    low=UniversalAdapter(model,'m','rev',kind='loha',rank=8,local_rank=2,target_scope='linear')
    high=UniversalAdapter(nn.Sequential(nn.Linear(12,12,bias=False)),'m','rev',kind='loha',rank=8,local_rank=8,target_scope='linear')
    spec=high.groups[0].overlay.local
    parts=[torch.randn(12,2) for _ in range(4)]
    dense=_delta_block(spec,parts,0,12)
    tiled=_factor_tile(spec,parts,slice(0,12),slice(0,12))
    frozen=FrozenDelta(spec,parts).delta()
    assert torch.allclose(dense,tiled,atol=1e-6)
    assert torch.allclose(dense,frozen,atol=1e-6)


def test_objective_editor_modal_scroll_keyboard_and_fonts():
    from playwright.sync_api import sync_playwright
    html=compose_html()
    assert '"myriad-pro-cond"' in html
    assert 'Noto+Sans+JP' in html
    assert 'https://use.typekit.net/wzz8rse.css' in html
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        for width,height in [(1440,900),(1060,650),(800,510)]:
            page=browser.new_page(viewport={'width':width,'height':height})
            page.route('**/fonts.googleapis.com/**',lambda route:route.abort())
            page.route('**/use.typekit.net/**',lambda route:route.abort())
            exceptions=[]
            page.on('pageerror',lambda e: exceptions.append(str(e)))
            page.set_content(html,wait_until='domcontentloaded')
            assert page.locator('#objectiveDialog').is_hidden()
            page.locator('button[data-tab="new"]').click()
            assert page.locator('#mixedPanel').evaluate('(x)=>x.parentElement.id')=='objectiveDialogBody'
            assert page.locator('#chooseObjectives').is_visible()
            page.locator('#chooseObjectives').click()
            assert page.locator('#objectiveDialog').is_visible()
            page.locator('#mixEnabled').select_option('on')
            box=page.locator('#objectiveDialogBody')
            heights=box.evaluate('(el)=>({client:el.clientHeight,scroll:el.scrollHeight})')
            assert heights['scroll']>heights['client']+100,(width,height,heights)
            box.evaluate('(el)=>{el.scrollTop=el.scrollHeight}')
            assert box.evaluate('(el)=>el.scrollTop')>200
            assert page.locator('#objectiveApply').is_visible()
            page.locator('#objectiveApply').click()
            assert page.locator('#objectiveDialog').is_hidden()
            page.locator('#chooseObjectives').click()
            page.keyboard.press('Escape')
            assert page.locator('#objectiveDialog').is_hidden()
            assert not exceptions,(width,height,exceptions)
            page.close()
        browser.close()
