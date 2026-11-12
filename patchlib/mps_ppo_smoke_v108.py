"""Run actual tiny Actor/LoRA/PPO on MPS in a kill-isolated subprocess."""
from __future__ import annotations
import os
import subprocess
import sys

def main(steps=4,timeout=120):
    env=os.environ.copy()
    env['KAGAMI_EXPERIMENTAL_MPS']='1'
    args=[sys.executable,'-m','patchlib.ppo_doctor_v103','--device','mps','--steps',str(steps),'--mps-worker']
    print('KAGAMI MPS PPO SMOKE: isolated child, no Hub downloads, no user journals',flush=True)
    try:
        result=subprocess.run(args,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,timeout=timeout,check=False)
    except subprocess.TimeoutExpired:
        print('KAGAMI MPS PPO SMOKE TIMEOUT; use CPU safe mode',flush=True)
        return 124
    print(result.stdout,end='',flush=True)
    if result.returncode:
        print('KAGAMI MPS PPO SMOKE FAILED (child exit=%s); do not use MPS for persistent rooms'%result.returncode,flush=True)
    return result.returncode
