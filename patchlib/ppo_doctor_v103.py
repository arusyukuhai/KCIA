"""Offline PPO integration probe. No downloads, user data writes or live rooms.

This is deliberately not a certificate of the user's actual HF reward/critic
configuration. It isolates the Actor/LoRA/PPO execution path and backend.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace


def main(argv=None):
    parser=argparse.ArgumentParser(description='Isolated tiny Actor PPO/LoRA weight-update test (no HF downloads)')
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu')
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--mps-worker',action='store_true',help='Build model and train on the dedicated MPS model worker')
    options=parser.parse_args(argv)
    if not 1<=options.steps<=20:
        parser.error('--steps must be 1..20')
    if options.mps_worker:
        # The actual GPU model should be created and trained on the same
        # serialized worker, just like train-elastic. This process is launched
        # by mps-ppo-smoke, so a native Metal abort cannot kill the GUI.
        if options.device!='mps':parser.error('--mps-worker requires --device mps')
        import asyncio
        from .mps_runtime_v91 import model_task
        async def go():
            return await model_task(main,['--device','mps','--steps',str(options.steps)],device='mps')
        return asyncio.run(go())
    import time
    import torch
    from torch import nn
    from .mps_policy_v92 import check_device
    from .core import UniversalAdapter
    from .cli import run_step
    from .mixed_learning_v65 import MixedObjectiveTrainer

    check_device(options.device,field='ppo-doctor device')
    if options.device=='mps' and not torch.backends.mps.is_available():
        print('KAGAMI PPO DOCTOR FAIL: torch.backends.mps.is_available()=False',flush=True)
        return 2
    device=torch.device(options.device)
    class TinyTokenizer:
        eos_token_id=None
        def encode(self,text,add_special_tokens=False):
            return [3+ord(ch)%47 for ch in text]
        def decode(self,ids,skip_special_tokens=False):
            return ''.join(chr(65+int(x)%26) for x in ids)
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=nn.Embedding(64,24)
            self.head=nn.Linear(24,64)
        def forward(self,input_ids,attention_mask=None):
            return {'logits':self.head(self.embed(input_ids))}

    print('KAGAMI PPO DOCTOR platform=' + str(__import__('sys').platform)
          + ' torch=' + str(torch.__version__) + ' device=' + options.device,flush=True)
    # Never modify persistent room state, checkpoint, cache, or user datasets.
    try:
        with tempfile.TemporaryDirectory(prefix='kagami-ppo-doctor-') as directory:
            path=Path(directory)/'reference.jsonl'
            path.write_text(json.dumps({'prompt':'abc','reference':'xyz'})+'\n',encoding='utf8')
            torch.manual_seed(103)
            model=TinyModel().to(device)
            adapter=UniversalAdapter(model,'ppo-doctor','toy-pinned',kind='lora',rank=2,target_scope='linear')
            params=list(adapter.trainable_parameters())
            if not params:raise RuntimeError('Actor has no trainable adapter tensors')
            optimizer=torch.optim.AdamW(params,lr=.03)
            args=SimpleNamespace(
                device=options.device, sft_jsonl=None,ppo_jsonl=path, pt_weight=0.,sft_weight=0.,ppo_weight=1.,
                ppo_clip=.2,ppo_kl_coef=0.,ppo_value_coef=.5,ppo_gamma=.99,
                ppo_gae_lambda=.95,ppo_max_tokens=3,ppo_reuse=1,ppo_reward='exact',
                ppo_kv_cache='off',ppo_adv_norm='rms',ppo_reference_aux_coef=1.,
                max_length=24,hf_task='causal-lm',loss_policy='auto',critic_kind='mlp',
                critic_hidden=32,critic_lr=.001,input_map='',grad_accum=1,
                telemetry_file=None, batch_size=1)
            mixer=MixedObjectiveTrainer(args,TinyTokenizer())
            args._v65_mixer=mixer
            initial=[p.detach().cpu().clone() for p in params]
            try:
                durations=[]
                for step in range(1,options.steps+1):
                    started=time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    loss=run_step(model,adapter,optimizer,{},args)
                    metrics=args._v65_ppo_metrics
                    if not metrics['actor_grad_norm']>0:
                        raise RuntimeError('Actor received no PPO/reference gradient')
                    optimizer.step()
                    # End-of-step timing only in this diagnostic; production
                    # training does not add an extra synchronization.
                    if device.type=='mps':torch.mps.synchronize()
                    elapsed=time.perf_counter()-started
                    durations.append(elapsed)
                    print(f'KAGAMI PPO DOCTOR step={step} seconds={elapsed:.3f} loss={loss:.6g} '
                          f'actor_grad_norm={metrics["actor_grad_norm"]:.6g} '
                          f'reward={metrics["reward"]:.5g}',flush=True)
                total=sum(float((p.detach().cpu()-prev).abs().sum()) for p,prev in zip(params,initial))
                if not total>0:raise RuntimeError('PPO did not update adapter weights')
                print(f'KAGAMI PPO DOCTOR PASS adapter_weight_change_l1={total:.6g} '
                      f'avg_step_seconds={sum(durations)/len(durations):.4f}',flush=True)
                print('This tests local tiny PPO only; HF Reward/Critic, P2P signed epochs and native Metal stability are not certified.',flush=True)
                return 0
            finally:
                mixer.close()
    except Exception as exc:
        print('KAGAMI PPO DOCTOR FAIL: '+str(exc),flush=True)
        traceback.print_exc()
        return 1

if __name__=='__main__':
    raise SystemExit(main())
