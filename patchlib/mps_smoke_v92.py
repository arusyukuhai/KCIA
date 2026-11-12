"""A standalone subprocess GPU probe; never run in the training process.

Even a pass does not imply training is safe; native abort is hardware/shape-
 and workload-specific. No model files, journals or signatures touched.
"""
from __future__ import annotations
import json
import os
import platform
import subprocess
import sys

_PROBE = r'''import os, platform, sys
import torch
print('torch=',torch.__version__,'python=',sys.version.split()[0],
      'macos=',platform.mac_ver()[0],flush=True)
if not torch.backends.mps.is_available():
    raise SystemExit('MPS unavailable')
a=torch.randn(96,96,device='mps')
b=torch.eye(96,device='mps')
for i in range(32):
    c=torch.relu(a@b+a)
    d=c.detach().to('cpu',non_blocking=False).clone()
    torch.mps.synchronize()
    a=c.detach()
    if i%8==0: print('iteration',i,'sum',float(d.sum()),flush=True)
print('MPS PROBE PASS (not a training stability guarantee)',flush=True)
'''


def main():
    print('This is a separate subprocess. Aborting it will NOT stop Kagami GUI.',flush=True)
    env=os.environ.copy()
    # Probe is intentionally independent from Kagami's default CPU policy.
    env['KAGAMI_EXPERIMENTAL_MPS']='1'
    try:
        cp=subprocess.run([sys.executable,'-c',_PROBE], env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,timeout=90,check=False)
    except subprocess.TimeoutExpired as exc:
        print('MPS PROBE TIMEOUT: backend may be hung',flush=True)
        return 124
    print(cp.stdout, end='',flush=True)
    if cp.returncode:
        print(f'MPS PROBE FAILED: child exit={cp.returncode}. Continue with CPU safe mode.',flush=True)
    return cp.returncode

if __name__=='__main__':
    raise SystemExit(main())
