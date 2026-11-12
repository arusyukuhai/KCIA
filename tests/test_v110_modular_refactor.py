"""Characterization tests for the v110 behavior-preserving module boundaries."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch


def test_public_entrypoints_keep_their_signatures():
    from patchlib import cli, elasticnet, mixed_learning_v65 as mixed
    from patchlib.webview_controller import WebviewController
    assert str(inspect.signature(cli.load_model)) == "(args, *, use_adapter: 'bool')"
    assert str(inspect.signature(cli.training_batches)) == '(args)'
    assert str(inspect.signature(cli.run_step)) == '(model, adapter, optimizer, batch, args)'
    assert str(inspect.signature(mixed.spec)) == '(args)'
    assert str(inspect.signature(elasticnet.train_elastic)).startswith('(args,')
    assert str(inspect.signature(WebviewController.start)) == "(self, payload: 'dict')"
    assert str(inspect.signature(WebviewController.save)) == "(self, payload: 'dict')"


def test_single_mixed_domain_error_type_and_compatibility_exports():
    from patchlib.mixed_errors_v110 import MixedLearningError
    from patchlib.mixed_learning_v65 import MixedLearningError as LegacyError
    from patchlib.mixed_kernels_v110 import MixedLearningError as KernelError
    from patchlib.mixed_policy_v110 import MixedLearningError as PolicyError
    assert MixedLearningError is LegacyError is KernelError is PolicyError
    from patchlib.cli import load_model, resolve_device, causal_labels_from_attention
    assert load_model.__module__ == 'patchlib.model_loading_v110'
    assert resolve_device('cpu') == torch.device('cpu')
    assert causal_labels_from_attention(torch.tensor([[1, 2]]),torch.tensor([[1, 0]])).tolist()==[[1,-100]]


def test_signed_mixed_policy_bytes_unchanged_from_v109(tmp_path):
    from test_mixed_learning_v65 import make_args
    from patchlib.mixed_learning_v65 import spec
    options=make_args(tmp_path, sft_jsonl=tmp_path/'sft.jsonl', ppo_jsonl=tmp_path/'ppo.jsonl')
    data=spec(options)
    serialized=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    assert hashlib.sha256(serialized).hexdigest() == 'be48822b2b3184dd0042b30de0dba447350aa7767cffdc9da458924e4345ebdb'
    assert 'sft.jsonl' not in serialized.decode()


def test_signed_control_frame_bytes_unchanged_from_v109():
    from patchlib.elasticnet import encode_control, decode_control
    wire=encode_control({'type':'heartbeat'})
    assert hashlib.sha256(wire.encode()).hexdigest() == '92d24da6280c6658416025b276d268024500d2f4c0a8ba219011a0fd594e0c4f'
    assert decode_control(wire,unsafe=True)['type']=='heartbeat'


def test_optimizer_moment_device_migration_does_not_modify_values():
    from patchlib.mixed_kernels_v110 import move_optimizer_moments_to_device
    param=torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optim=torch.optim.AdamW([param],lr=.001)
    optim.state[param]={'exp_avg':torch.tensor([2.0,3.0]),
                        'exp_avg_sq':torch.tensor([4.0,9.0]),
                        'step':torch.tensor(3)}
    prev={k:v.clone() for k,v in optim.state[param].items()}
    move_optimizer_moments_to_device(optim)
    for key in prev:
        torch.testing.assert_close(prev[key],optim.state[param][key])
        assert optim.state[param][key].device.type=='cpu'


def test_main_entrypoints_still_delegate_without_network_start(monkeypatch):
    from patchlib import webview_lifecycle_v110
    from patchlib.webview_controller import WebviewController
    c=WebviewController(Path('/tmp/kcia-v110-inert-controller'))
    try:
        marker=[]
        def fake_save(controller,payload):
            marker.append(('save',payload))
            return {'ok':True}
        def fake_start(controller,payload,*,profile_allocator):
            marker.append(('start',callable(profile_allocator)))
            return {'ok':True}
        monkeypatch.setattr(webview_lifecycle_v110,'save_room',fake_save)
        monkeypatch.setattr(webview_lifecycle_v110,'start_training',fake_start)
        assert c.save({'room':'unit'})=={'ok':True}
        assert c.start({'room':'unit'})=={'ok':True}
        assert marker==[('save',{'room':'unit'}),('start',True)]
    finally:
        c.close()
