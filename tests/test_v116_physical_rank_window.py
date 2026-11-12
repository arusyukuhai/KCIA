"""v115: heterogeneous TRAINABLE subranks without wire-layout forks."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.core import UniversalAdapter
from patchlib.webview_gui import KCIABridge, compose_html, run_webview


def make_adapter():
    torch.manual_seed(42)
    return UniversalAdapter(nn.Sequential(nn.Linear(12,12,bias=False)),
                            'test/model','fixed-revision',kind='loha',rank=8,
                            target_scope='linear')


def test_physical_low_rank_updates_wire_and_adamw():
    from patchlib.core import decode_patch
    torch.manual_seed(20)
    a=UniversalAdapter(nn.Sequential(nn.Linear(12,12,bias=False)),
                       'test/model','fixed-revision',kind='loha',rank=8,
                       target_scope='linear',local_rank=2)
    torch.manual_seed(20)
    b=UniversalAdapter(nn.Sequential(nn.Linear(12,12,bias=False)),
                       'test/model','fixed-revision',kind='loha',rank=8,
                       target_scope='linear',local_rank=8)
    assert a.layout==b.layout
    assert a.counts()['patch'] * 4 == b.counts()['patch']
    oa=torch.optim.AdamW(list(a.trainable_parameters()),lr=1e-3)
    ob=torch.optim.AdamW(list(b.trainable_parameters()),lr=1e-3)
    for adapter,opt in ((a,oa),(b,ob)):
        loss=(adapter.model(torch.randn(3,12))**2).mean()
        loss.backward();opt.step()
    assert sum(x.numel() for state in oa.state.values() for x in state.values() if x.ndim) * 4 == sum(
        x.numel() for state in ob.state.values() for x in state.values() if x.ndim)
    wa=a.export('small',1);wb=b.export('big',1)
    assert len(wa)<len(wb)
    m,weights=decode_patch(wa)
    assert m['rank']==8 and m['rank_protocol']=='heterogeneous-loha-v1'
    assert m['targets'][0]['rank']==2
    assert all(p.shape[-1]==2 for p in weights.values())
    b.import_remote(wa);a.import_remote(wb)
    assert torch.isfinite(a.model(torch.randn(2,12))).all()
    assert torch.isfinite(b.model(torch.randn(2,12))).all()
    with pytest.raises(ValueError): a.set_local_training_rank(8)


def test_physical_auto_rank_and_pin(tmp_path):
    from patchlib.local_rank_v116 import choose_physical_rank
    model=nn.Sequential(nn.Linear(12,12,bias=False))
    args=SimpleNamespace(rank=8,kind='loha',model='test/model',model_id=None,
                         revision='fixed-revision',target_scope='linear',
                         wide_lora_rank=4,local_loha_rank='auto',journal=tmp_path/'peer.db')
    low=SimpleNamespace(usable_bytes=128,device='cpu')
    high=SimpleNamespace(usable_bytes=4*1024**3,device='cuda')
    assert choose_physical_rank(model,args,budget=low)['physical_rank']==1
    # Initial choice pinned despite much more memory available on restart.
    assert choose_physical_rank(model,args,budget=high)['physical_rank']==1
    args.local_loha_rank='8'
    with pytest.raises(ValueError,match='cannot resize'):
        choose_physical_rank(model,args,budget=high)


def test_webview_frameless_and_window_controls(tmp_path,monkeypatch):
    class Window:
        def __init__(self):self.actions=[]
        def minimize(self):self.actions.append('minimize')
        def maximize(self):self.actions.append('maximize')
        def restore(self):self.actions.append('restore')
        def destroy(self):self.actions.append('destroy')
    window=Window()
    class Fake:
        windows=[window]
        def create_window(self,title,**kw):
            assert kw['frameless'] is True
            assert kw['easy_drag'] is False
            assert 'windowClose' in kw['html']
            assert 'pywebview-drag-region' in kw['html']
            return window
        def start(self,**kw):return None
    run_webview(workdir=tmp_path,webview_module=Fake())
    bridge=KCIABridge()
    bridge.bind_window(window)
    for act in ('minimize','toggle_maximize','toggle_maximize','close'):
        result=bridge.window_action({'action':act})
        assert result['queued'] is True
    bridge._window_actions.join()
    assert window.actions==['minimize','maximize','restore','destroy']
    with pytest.raises(ValueError):
        bridge.window_action({'action':'eval'})


def test_gui_command_runs_local_subrank_auto(tmp_path):
    from patchlib.gui import DesktopController
    from patchlib.room_manifest import make_hf_manifest
    from patchlib.roomid import create_room_id
    room=create_room_id()
    desc=make_hf_manifest('org/model','fixed-sha',[{
        'repo':'org/dataset','revision':'main','config':None,'split':'train',
        'columns':{'text':'text'},'weight':1.0}])
    manifest=tmp_path/'manifest.json'
    DesktopController.save_manifest(manifest,desc)
    init=tmp_path/'genesis.fstate';init.write_bytes(b'\0')
    journal=tmp_path/'events.sqlite3'
    cmd=DesktopController.build_command(room=room,model='org/model',revision='fixed-sha',
        base=init,manifest=manifest,ws='wss://signal.example.com/ws',
        task='causal-lm',kind='loha',rank=16,journal=journal)
    i=cmd.index('--local-loha-rank')
    assert cmd[i+1]=='auto'
