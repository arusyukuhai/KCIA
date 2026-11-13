#!/usr/bin/env python3
"""KCIA two-peer WAN PT pilot. For live use on Mac and Colab only.

Commands:
  python scripts/start_wan_pt.py prepare
  python scripts/start_wan_pt.py run --role mac --room-id ROOM --revision SHA
  python scripts/start_wan_pt.py run --role colab --room-id ROOM --revision SHA
  python scripts/start_wan_pt.py run --role mac --room-id ROOM --revision SHA --dry-run

The room ID is a bearer secret. Keep it private; use a new one for each pilot.
Room mode creates a separate persistent signing identity beside each journal.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODEL='HuggingFaceTB/SmolLM2-135M'
WS='wss://signal.kcia.tech/ws'
SHA=re.compile(r'^[0-9a-fA-F]{40}$')
ROOM=re.compile(r'^[A-Za-z0-9_-]{24,64}$')

def make_command(a):
    if not ROOM.fullmatch(a.room_id):
        raise ValueError('Invalid room ID; generate via prepare')
    if not SHA.fullmatch(a.revision):
        raise ValueError('Model revision must be a pinned 40-character SHA')
    if a.role not in ('mac','colab'):
        raise ValueError('invalid role')
    state=ROOT/'wan_state'/a.role
    device=a.device or ('cpu' if a.role=='mac' else 'cuda')
    data=ROOT/'examples'/('wan_mac.txt' if a.role=='mac' else 'wan_colab.txt')
    cmd=[sys.executable,'-m','patchlib.cli','train-live',
         '--ws',a.ws,'--room-id',a.room_id,
         '--model',MODEL,'--revision',a.revision.lower(),
         '--auto-state','--state-base',str(state/'genesis.fstate'),
         '--text-file',str(data),'--journal',str(state/'events.sqlite3'),
         '--kind','loha','--rank','32','--local-loha-rank','auto',
         '--reinit-every','20','--local-steps','4',
         '--device',device,'--batch-size','1','--max-length','64',
         '--rounds',str(a.rounds),'--lr','0.0001',
         '--shard-mb',str(a.snapshot_shard_mb)]
    # Mac CPU smoke test must not be silently switched to MPS by adaptive tuning.
    if device=='cpu':cmd+=['--no-auto-tune']
    return cmd

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('prepare',help='Generate fresh room ID and resolve pinned HF model SHA')
    run=sub.add_parser('run',help='Run PT in own peer state/journal')
    run.add_argument('--role',choices=('mac','colab'),required=True)
    run.add_argument('--room-id',required=True)
    run.add_argument('--revision',required=True)
    run.add_argument('--ws',default=WS)
    run.add_argument('--device',choices=('cpu','cuda','mps'),default=None)
    run.add_argument('--rounds',type=int,default=0,help='0 keeps training until manually stopped')
    run.add_argument('--snapshot-shard-mb',type=int,default=2,
                     help='Full-snapshot shard size in MiB (1..64); 2 MiB improves reconnect resume')
    run.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(argv)
    if a.cmd=='prepare':
        from huggingface_hub import model_info
        from patchlib.roomid import create_room_id
        rev=model_info(MODEL).sha
        if not SHA.fullmatch(rev):
            raise RuntimeError('Hub did not supply a full pinned revision SHA')
        print(json.dumps({'model':MODEL,'room_id':create_room_id(),
                          'revision':rev,'ws':WS},indent=2))
        print('COPY room_id AND revision to Colab. Never share signing private keys.')
        return 0
    if a.rounds<0:ap.error('--rounds must be >= 0')
    if not 1<=a.snapshot_shard_mb<=64:ap.error('--snapshot-shard-mb must be 1..64')
    try:cmd=make_command(a)
    except ValueError as e:ap.error(str(e))
    state=ROOT/'wan_state'/a.role
    if a.dry_run:
        print(json.dumps({'cwd':str(ROOT),'role':a.role,'argv':cmd,
             'private_key_will_be_auto_generated_at':str(state/'events.sqlite3.identity.pem')},indent=2))
        return 0
    if a.role=='colab' and (a.device is None or a.device=='cuda'):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Colab CUDA not available. Enable GPU runtime first.')
        except ImportError as e:
            raise RuntimeError('Install requirements-patches.txt first') from e
    state.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(state,0o700)
    print('KCIA PT pilot starting:',a.role,'room:',a.room_id,'device:',a.device or ('cpu' if a.role=='mac' else 'cuda'),flush=True)
    print('Press Ctrl-C after both peers connect, exchange updates, and record loss. Keep your wan_state folder.',flush=True)
    try:
        env=os.environ.copy()
        # Pilot nodes prioritize a joining peer's verified full snapshot over
        # local steps that would fill the journal before the join completes.
        # Production train-live remains unchanged unless explicitly enabled.
        env.setdefault('KCIA_SNAPSHOT_PRIORITY','1')
        return subprocess.call(cmd,cwd=ROOT,env=env)
    except KeyboardInterrupt:
        return 130

if __name__=='__main__':raise SystemExit(main())
