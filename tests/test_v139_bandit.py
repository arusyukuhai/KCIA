"""WAN cost-bandit regression tests; no network or model downloads."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.wan_bandit_v139 import WanCostBandit
from patchlib.gui import DesktopController


def finish(bandit, drift=.015, inbound=0, patch_mib=160, *, inject=False):
    steps = bandit.selected
    for i in range(steps):
        bandit.step(5.0 - drift*i/max(steps, 1),
                    inbound_applied=inbound+(1 if inject and i>steps//2 else 0))
    return bandit.observe(steps=steps,signed_bytes=patch_mib*1024*1024,
                          inbound_applied=inbound+(1 if inject else 0))


def test_optimizer_reduces_oversized_event_frequency_without_infinite_changes():
    b=WanCostBandit(baseline=32,ceiling=256,wire_cost=.08)
    assert b.selected==32
    seq=[]
    for _ in range(24):
        before=b.selected
        assert finish(b,drift=.025)
        assert .5 <= b.selected/before <= 2
        seq.append(b.selected)
    assert seq[0]==32
    assert max(seq)>=128
    assert all(32<=x<=256 for x in seq)
    assert b.samples==24
    assert 0 < b.wire_estimate_mib < 200
    assert math.isfinite(b.last_utility)


def test_worsened_loss_prefers_more_frequent_communication_after_probe():
    b=WanCostBandit(baseline=32,ceiling=256)
    seen_128=False
    for _ in range(24):
        arm=b.selected
        if arm >= 128: seen_128=True
        assert finish(b,drift=(-.15 if arm>=128 else .03))
    assert seen_128
    assert b.selected<=64


def test_inbound_updates_and_truncated_blocks_do_not_claim_local_loss_gain():
    b=WanCostBandit(baseline=32,ceiling=256)
    assert finish(b,inject=True) is False
    assert b.samples==0
    for _ in range(8):b.step(4.99,inbound_applied=1)
    assert b.observe(steps=8,signed_bytes=160*1024*1024,inbound_applied=1) is False
    assert b.samples==0
    assert finish(b,inbound=2)
    assert b.samples==1


def test_bad_observations_stay_out_of_regression():
    b=WanCostBandit(baseline=32)
    for i in range(32):
        b.step(float('nan') if i==3 else 5.0,inbound_applied=0)
    assert b.observe(steps=32,signed_bytes=3_000,inbound_applied=0) is False
    assert b.samples==0
    for kwargs in ({'signed_bytes':-1},{'measured_egress_bytes':-1},{'recipients':-1},{'recipients':9}):
        args={'steps':0,'signed_bytes':0,'inbound_applied':0}
        args.update(kwargs)
        with pytest.raises(ValueError):b.observe(**args)
    with pytest.raises(ValueError):WanCostBandit(baseline=32,wire_cost=float('nan'))
    with pytest.raises(ValueError):WanCostBandit(baseline=32,exploration=5.)


def test_actual_network_bytes_raise_cost_over_wire_estimate():
    b=WanCostBandit(baseline=32,wire_cost=.08)
    for i in range(32):b.step(5-i*.001,inbound_applied=0)
    assert b.observe(steps=32,signed_bytes=1*1024**2,
                     measured_egress_bytes=5*1024**2,inbound_applied=0)
    assert b.wire_estimate_mib == 5
    assert b.last_wire_per_step==pytest.approx(5/32)


def test_bandit_persistence_integrity_and_no_signed_room_mutation(tmp_path):
    path=tmp_path/'local.sqlite3.wan-bandit-v139.json'
    b=WanCostBandit(baseline=32,ceiling=128,path=path)
    for _ in range(4):assert finish(b)
    assert (path.stat().st_mode & 0o777)==0o600
    restored=WanCostBandit(baseline=32,ceiling=128,path=path)
    assert restored.selected==b.selected
    assert restored.samples==b.samples
    assert restored.events==b.events
    different_room=WanCostBandit(baseline=64,ceiling=128,path=path)
    assert different_room.samples==0
    different_price=WanCostBandit(baseline=32,ceiling=128,path=path,wire_cost=.24)
    assert different_price.samples==0  # stale decisions from a prior cost preference
    path.write_text('{corrupted')
    assert WanCostBandit(baseline=32,ceiling=128,path=path).samples==0
    # Never write through a user-controlled symbolic link.
    victim=tmp_path/'sensitive';victim.write_text('secret')
    path.unlink()
    path.symlink_to(victim)
    for _ in range(3): finish(WanCostBandit(baseline=32,ceiling=128,path=path))
    assert victim.read_text()=='secret'


def test_online_quad_regression_stable_and_bounded():
    b=WanCostBandit(baseline=32,ceiling=1024)
    for arm in b.actions:
        b.history.append({'arm':arm,'gain':min(2.,max(-2.,(.7-0.2*math.log2(arm/32)**2)))})
    for arm in b.actions:
        assert math.isfinite(b.utility(arm))
    for i in range(40):
        prior=b.selected
        b.choose()
        assert 0.5<=b.selected/prior<=2


def test_cli_flags_and_gui_wan_policy_wiring(tmp_path):
    import subprocess
    import sys
    # Confirm real CLI help exposes all worker tuning flags without starting
    # a network-connected training process.
    help_text=subprocess.run([sys.executable,'-m','patchlib.cli','train-live','--help'],
                             capture_output=True,text=True,check=True).stdout
    assert '--cost-bandit' in help_text and '--bandit-wire-cost' in help_text
    assert '--bandit-exploration' in help_text
    from patchlib.webview_controller import WebviewController
    c=WebviewController(tmp_path)
    prefs=c.training_preferences()
    assert prefs['publish_mode']=='wan'
    assert c.configure_training_preferences({**prefs,'bandit_wire_cost':.2})['bandit_wire_cost']==.2
    assert WebviewController(tmp_path).training_preferences()['bandit_wire_cost']==.2
    with pytest.raises(ValueError):c.configure_training_preferences({**prefs,'bandit_wire_cost':float('inf')})
    js=(Path(__file__).parents[1]/'web'/'desktop_webview.js').read_text()
    html=(Path(__file__).parents[1]/'web'/'desktop_webview.html').read_text()
    assert 'bandit_wire_cost:Number(get(' in js
    assert 'trainBanditStats' in js and 'trainBanditWireCost' in html


def test_wan_gui_launch_passes_cost_to_worker_without_room_manifest_mutation(tmp_path,monkeypatch):
    from patchlib.webview_controller import WebviewController
    from patchlib.roomid import create_room_id
    from patchlib.room_manifest import manifest_digest, load_manifest
    ctrl=WebviewController(tmp_path)
    rid=create_room_id()
    ctrl.save({'room':rid,'model':'sshleifer/tiny-gpt2','revision':'main',
        'task':'causal-lm','initialization':'pretrained',
        'sources':[{'repo':'org/sft','revision':'a'*40,'config':None,
            'split':'train','columns':{'text':'text'},'weight':1.0}]})
    manifest=ctrl._room_path(rid)
    digest=manifest_digest(load_manifest(manifest))
    launched=[]
    monkeypatch.setattr(ctrl.train,'start',lambda cmd,logfile:launched.append(list(cmd)) or 123)
    ctrl.start({'room':rid,'ws':'wss://signal.kcia.tech/ws','auto_model_cadence':True,
                'local_steps':32,'bandit_wire_cost':.18})
    command=launched[-1]
    assert '--cost-bandit' in command
    assert command[command.index('--bandit-wire-cost')+1]=='0.18'
    assert command[command.index('--max-sync-steps')+1]=='512'
    assert manifest_digest(load_manifest(manifest))==digest
    ctrl.start({'room':rid,'ws':'wss://signal.kcia.tech/ws','auto_model_cadence':False,
                'local_steps':64,'bandit_wire_cost':.18})
    assert '--cost-bandit' not in launched[-1]
    for val in (float('nan'),float('inf'),-1,'0.08'):
        with pytest.raises(ValueError):
            ctrl.start({'room':rid,'ws':'wss://signal.kcia.tech/ws',
                        'auto_model_cadence':True,'bandit_wire_cost':val})


def test_bandit_metrics_roundtrip_for_gui(tmp_path):
    from patchlib.live_telemetry import LocalTelemetry, recent_samples
    p=tmp_path/'kcia-metrics.jsonl'
    t=LocalTelemetry(p)
    assert t.record(steps=32, loss=4.6, emitted=1, inbound_applied=0,
                    inbox=0,channels=1,publish_steps=128,
                    bandit_trials=5,bandit_gain_per_1000=-.03,
                    bandit_signed_mib_per_step=1.25,bandit_net_utility=-.2,force=True)
    t.close()
    r=recent_samples(p)
    assert r[0]['publish_steps']==128
    assert r[0]['bandit_trials']==5
    assert r[0]['bandit_gain_per_1000']==pytest.approx(-.03)
    assert r[0]['bandit_signed_mib_per_step']==pytest.approx(1.25)
    assert r[0]['bandit_net_utility']==pytest.approx(-.2)
