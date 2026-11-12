"""Independent local objective training plots. No validation claims or signed changes."""
import json
import math
from pathlib import Path

import pytest
import torch

from patchlib.objective_metrics_v107 import token_top1, reference_token_match, finite_metric
from patchlib.live_telemetry import LocalTelemetry, history_samples
from patchlib.cli import run_step
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
from tests.test_mixed_learning_v65 import ToyTokenizer, make_args, setup_model, write


def test_pt_next_token_shift_and_mask():
    logits=torch.full((1,5,9),-3.)
    # Target is labels[1:], not labels[0:4]. One position masked.
    labels=torch.tensor([[2,5,-100,8,1]])
    for t,tok in enumerate([5,3,8,0]):
        logits[0,t,tok]=9.
    assert token_top1(logits,labels)==pytest.approx(2/3)
    assert token_top1(logits,torch.full_like(labels,-100)) is None
    assert token_top1(torch.randn(1,3,9),labels) is None


def test_sft_only_supervised_response_tokens():
    logits=torch.full((1,6,10),-1.)
    labels=torch.tensor([[-100,-100,-100,4,5,6]])
    logits[0,2,4]=5
    logits[0,3,2]=5
    logits[0,4,6]=5
    assert token_top1(logits,labels)==pytest.approx(2/3)


def test_non_causal_classification():
    logits=torch.tensor([[0.,3.],[5.,0.]])
    labels=torch.tensor([1,0])
    assert token_top1(logits,labels,causal=False)==1.0
    assert token_top1(logits,labels,causal=True) is None


def test_ppo_reference_metrics_not_fabricated():
    assert reference_token_match([1,2,3],[1,7,3,5])==.5
    assert reference_token_match([1,2],[1,2])==1.
    assert reference_token_match([1,2],None) is None
    assert reference_token_match([],[]) is None
    assert finite_metric(float('nan')) is None
    assert finite_metric(float('inf')) is None
    assert finite_metric(-.1,minimum=0) is None


def test_sft_training_records_masked_top1(tmp_path):
    sft=write(tmp_path,'sft.jsonl',[{'prompt':'Hey','response':'Okay'}])
    args=make_args(tmp_path,sft_jsonl=sft,pt_weight=0.,ppo_weight=0.,telemetry_file=str(tmp_path/'samples.jsonl'))
    args._metric_step=0
    model,adapter,opt,batch=setup_model()
    trainer=MixedObjectiveTrainer(args,ToyTokenizer())
    args._v65_mixer=trainer
    try:
        run_step(model,adapter,opt,{},args)
        assert trainer.last_sft_accuracy is not None
        assert 0<=trainer.last_sft_accuracy<=1
        assert getattr(args,'_v107_pt_accuracy',None) is None
    finally:trainer.close()


def test_pt_training_records_real_top1(tmp_path):
    args=make_args(tmp_path,sft_weight=0.,ppo_weight=0.,telemetry_file=str(tmp_path/'telemetry'),pt_weight=1.)
    model,adapter,opt,batch=setup_model()
    args._metric_step=0
    # No SFT/PPO, v65 mixer is disabled.
    run_step(model,adapter,opt,batch,args)
    assert 0<=args._latest_accuracy<=1


def test_ppo_reference_match_present_only_for_labelled(tmp_path):
    ppo=write(tmp_path,'ppo.jsonl',[{'prompt':'Hi','reference':'yes'}])
    args=make_args(tmp_path,ppo_jsonl=ppo,pt_weight=0.,sft_weight=0.,ppo_weight=.2)
    model,adapter,opt,batch=setup_model()
    trainer=MixedObjectiveTrainer(args,ToyTokenizer())
    try:
        _,_,metrics=trainer.ppo_loss(model,torch.device('cpu'))
        assert metrics['reference_match']==pytest.approx(trainer.cached['reward'])
        assert 0<=metrics['reference_match']<=1
    finally:trainer.close()


def test_telemetry_stream_retains_independent_fields_and_backward_compat(tmp_path):
    path=tmp_path/'stats.jsonl'
    writer=LocalTelemetry(path,sample_seconds=0,append=True)
    common={'emitted':0,'inbound_applied':0,'inbox':0,'channels':0}
    writer.record(steps=1,loss=1.,accuracy=.65,pt_accuracy=.4,sft_accuracy=.75,
        sft_preference_accuracy=.5,ppo_reference_match=.25,
        ppo_reward=-.8,ppo_kl=.06,ppo_critic_loss=1.5,**common)
    writer.record(steps=2,loss=1.,accuracy=None,pt_accuracy=float('inf'),
        ppo_reward=float('nan'),ppo_kl=-.1,**common)
    writer.close()
    rows=history_samples(path,window_seconds=0,max_points=50)
    assert rows[0]['pt_accuracy']==.4
    assert rows[0]['sft_accuracy']==.75
    assert rows[0]['ppo_reward']==-.8
    assert rows[0]['ppo_critic_loss']==1.5
    assert rows[1]['pt_accuracy'] is None
    assert rows[1]['ppo_reward'] is None
    assert rows[1]['ppo_kl'] is None
    assert not any('prompt' in row or 'reference' in row for row in rows)


def test_gui_contains_all_objective_plots_and_uses_live_data():
    root=Path(__file__).resolve().parents[1]
    js=(root/'web/desktop_webview.js').read_text()
    html=(root/'web/desktop_webview.html').read_text()
    css=(root/'web/tailwind.cdn.css').read_text()
    for item in ['pt_accuracy','sft_accuracy','sft_preference_accuracy',
                 'ppo_reference_match','ppo_reward','ppo_kl','ppo_critic_loss']:
        assert f"'{item}'" in js
    for name in ['monitorPtAccuracyPlot','monitorSftAccuracyPlot',
                 'monitorPpoReferencePlot','monitorPpoRewardPlot','monitorPpoKlPlot']:
        assert name in html
        assert name in js
    assert 'chartPoints(field)' in js
    assert "field==='ppo_reference_match'" in js
    assert 'grid-auto-rows: 166px' in css
