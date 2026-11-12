import asyncio
from types import SimpleNamespace

import pytest
import torch

from patchlib import mps_acceleration_v108 as acc
from patchlib import mps_launch_v108 as launch


def test_mps_sampling_avoids_probability_cpu_copy_on_native(monkeypatch):
    monkeypatch.delenv('KCIA_MPS_NATIVE_SAMPLING',raising=False)
    distribution=torch.log_softmax(torch.tensor([3.0,1.0,-2.0]),dim=-1)
    called=[]
    old=acc.torch.multinomial
    def fake(tensor,n):
        called.append((str(tensor.device),tensor.shape))
        return old(tensor,n)
    monkeypatch.setattr(acc.torch,'multinomial',fake)
    sample,backend=acc.sample_next(distribution,backend='mps')
    assert backend=='mps-native' and sample.shape==(1,)
    assert len(called)==1


def test_mps_sampling_unsupported_falls_back_to_cpu_once(monkeypatch):
    distribution=torch.log_softmax(torch.tensor([3.0,1.0,-2.0]),dim=-1)
    calls=[]
    original=torch.multinomial
    def patched(tensor,n):
        calls.append(tensor.device.type)
        if len(calls)==1:raise RuntimeError('not implemented for MPS')
        return original(tensor,n)
    monkeypatch.setattr(acc.torch,'multinomial',patched)
    sample,backend=acc.sample_next(distribution,backend='mps')
    assert backend=='cpu-fallback'
    assert sample.numel()==1 and calls==['cpu','cpu']


def test_mps_sampling_driver_errors_are_not_swallowed(monkeypatch):
    def broken(*a,**kw):raise RuntimeError('commit an already committed command buffer')
    monkeypatch.setattr(acc.torch,'multinomial',broken)
    with pytest.raises(RuntimeError,match='already committed'):
        acc.sample_next(torch.log_softmax(torch.randn(6),dim=-1),backend='mps')


def test_mps_sampling_user_can_force_compatible_cpu_fallback(monkeypatch):
    monkeypatch.setenv('KCIA_MPS_NATIVE_SAMPLING','0')
    sample,backend=acc.sample_next(torch.log_softmax(torch.randn(6),dim=-1),backend='mps')
    assert backend=='cpu-fallback' and sample.numel()==1


def test_gae_cpu_recurrence_matches_reference():
    values=torch.tensor([.4,.7,.2,1.5,.1])
    adv,returns=acc.terminal_gae(1.25,values,.99,.95)
    result=[];carry=0
    v=values.tolist()
    for j in range(len(v)-1,-1,-1):
        carry=(1.25 if j==len(v)-1 else 0)+.99*(v[j+1] if j+1<len(v) else 0)-v[j]+.99*.95*carry
        result.append(carry)
    assert torch.allclose(adv,torch.tensor(list(reversed(result))),atol=1e-5)
    assert torch.allclose(returns,adv+values)


def test_gpu_device_opt_in_only_local_and_requested():
    base={'A':'B'}
    assert launch.launch_environment(['--device','cpu'],base)==base
    assert launch.launch_environment(['--device','mps'],base)['KCIA_EXPERIMENTAL_MPS']=='1'
    assert launch.launch_environment(['--device','mps'],base)['PYTORCH_ENABLE_MPS_FALLBACK']=='1'
    assert launch.launch_environment(['--device','cpu','--reward-model-device','mps'],base)['KCIA_EXPERIMENTAL_MPS']=='1'
    assert launch.launch_environment(['--device','cpu','--critic-hf-device','mps'],base)['KCIA_EXPERIMENTAL_MPS']=='1'
    assert launch.launch_environment(['--device','auto'],base)==base
    assert launch.needs_mps(['--device','mps'])


def test_preflight_rejects_unavailable_mps_without_mutating_room(monkeypatch):
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    with pytest.raises(ValueError,match='unavailable'):
        launch.validate_devices('mps')
    with pytest.raises(ValueError,match='unavailable'):
        launch.validate_devices('cpu',{'enabled':True,'ppo_reward':'hf_model',
                                      'reward_model_device':'mps'})
    launch.validate_devices('cpu')


def test_mps_ppo_smoke_is_a_subprocess_with_pinned_worker(monkeypatch):
    from patchlib import mps_ppo_smoke_v108 as probe
    calls=[]
    def fake(cmd,**kwargs):
        calls.append((cmd,kwargs))
        return SimpleNamespace(returncode=-6,stdout='Metal abort')
    monkeypatch.setattr(probe.subprocess,'run',fake)
    assert probe.main()==-6
    cmd,opts=calls[0]
    assert cmd[1:4]==['-m','patchlib.ppo_doctor_v103','--device']
    assert '--mps-worker' in cmd
    assert opts['env']['KCIA_EXPERIMENTAL_MPS']=='1'


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Real Apple GPU required')
def test_real_mps_sampling_gae_and_backward():
    from patchlib.mps_policy_v92 import check_device
    import os
    os.environ['KCIA_EXPERIMENTAL_MPS']='1'
    check_device('mps')
    logits=torch.randn(1,32,device='mps',requires_grad=True)
    sample,kind=acc.sample_next(torch.log_softmax(logits,dim=-1),backend='mps')
    assert kind in ('mps-native','cpu-fallback')
    chosen=torch.log_softmax(logits,dim=-1).gather(-1,sample.to('mps'))
    (-chosen.mean()).backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    advantages,returns=acc.terminal_gae(1.0,torch.tensor([.4,.2,.3],device='mps'),.99,.95)
    assert advantages.device.type=='mps' and returns.shape==(3,)


def test_desktop_command_selected_gpu_remains_peer_local(tmp_path):
    from patchlib.gui import DesktopController
    from patchlib.roomid import create_room_id
    ctl=DesktopController()
    descriptor=ctl.create_manifest('org/model','main','causal-lm',[
        {'repo':'org/data','revision':'main','config':None,'split':'train',
         'columns':{'text':'text'},'weight':1.0}])
    manifest=ctl.save_manifest(tmp_path/'manifest.json',descriptor)
    base=tmp_path/'start.fstate';base.write_bytes(b'test')
    kw=dict(room=create_room_id(),model='org/model',revision='main',base=base,
            manifest=manifest,ws='wss://localhost/ws',task='causal-lm',
            kind='loha',rank=16,journal=tmp_path/'history.sqlite3')
    command=ctl.build_command(**kw,device='mps')
    assert command[command.index('--device')+1]=='mps'
    assert ctl.build_command(**kw)[ctl.build_command(**kw).index('--device')+1]=='auto'
    with pytest.raises(ValueError,match='Unsupported'):
        ctl.build_command(**kw,device='unknown')


def test_webview_launch_passes_mps_and_keeps_signed_room_protocol(tmp_path,monkeypatch):
    from patchlib.webview_controller import WebviewController
    from patchlib.roomid import create_room_id
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    ctl=WebviewController(tmp_path)
    room=create_room_id()
    ctl.save({'room':room,'model':'org/test','revision':'main',
              'task':'causal-lm','initialization':'pretrained','seed':42,
              'sources':[{'repo':'org/data','revision':'main','config':None,
                          'split':'train','columns':{'text':'text'},'weight':1.0}]})
    started=[]
    monkeypatch.setattr(ctl.train,'start',lambda command,log: started.append(command) or 123)
    result=ctl.start({'room':room,'ws':'ws://127.0.0.1:8080/ws',
                      'streaming_base':True,'compute_device':'mps'})
    assert result['pid']==123
    assert started and started[0][started[0].index('--device')+1]=='mps'
    assert not (ctl._room_path(room).parent/'mixed-objectives-v66.json').exists()
    ctl.close()
