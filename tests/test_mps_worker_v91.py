"""Contract-level regression tests; hardware Metal coverage requires a Mac."""
import asyncio
import contextvars
import threading
import time
from types import SimpleNamespace

import pytest

from patchlib import mps_runtime_v91 as mps
from patchlib.elasticnet import run_step_offloop


def test_serial_gpu_worker_and_synchronize_on_failure(monkeypatch):
    threads=[]
    calls=[]
    executing=0
    lock=threading.Lock()
    context=contextvars.ContextVar('kcia_mps_test',default='none')
    context.set('present')
    monkeypatch.setattr(mps, 'synchronize_mps', lambda: calls.append('sync'))

    def model_step(i):
        nonlocal executing
        with lock:
            assert executing == 0, 'MPS torch work must never overlap across workers'
            executing+=1
        try:
            assert context.get()=='present'
            threads.append(threading.current_thread().name)
            time.sleep(.008)
            if i==3:
                raise ValueError('test model exception')
            return i
        finally:
            with lock:
                executing-=1

    async def run():
        jobs=await asyncio.gather(*(mps.model_task(model_step,i,device='mps')
                                    for i in range(8)),return_exceptions=True)
        return jobs

    results=asyncio.run(run())
    assert len(set(threads))==1
    assert threads[0].startswith('kcia-mps-model')
    assert isinstance(results[3], ValueError)
    assert results[0]==0 and results[7]==7
    assert calls==['sync']*8


def test_cpu_mode_does_not_invoke_mps_sync(monkeypatch):
    monkeypatch.setattr(mps, 'synchronize_mps', lambda: (_ for _ in ()).throw(
        AssertionError('CPU work may not synchronize MPS')))
    assert asyncio.run(mps.model_task(lambda x:x+1,41,device='cpu'))==42


def test_mps_type_detection_without_mac():
    fake=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(
        device=SimpleNamespace(type='mps'))]),buffers=lambda:iter(()))
    assert mps.uses_mps(model=fake)
    assert mps.uses_mps(device='mps:0')
    assert not mps.uses_mps(device='cpu')
    assert mps.execution_device(SimpleNamespace(device='cpu',reward_model_device='mps'))=='mps'
    assert mps.execution_device(SimpleNamespace(device='cpu',critic_hf_device='mps'))=='mps'


def test_hf_network_batch_fetch_does_not_occupy_gpu_worker(monkeypatch):
    events=[]
    started=threading.Event()
    release=threading.Event()
    monkeypatch.setattr(mps, 'synchronize_mps', lambda: None)

    def data():
        started.set()
        assert release.wait(timeout=2.0)
        yield 7

    async def run():
        read=asyncio.create_task(run_step_offloop(
            lambda *_args: _args[-2],None,None,None,iter(data()),
            SimpleNamespace(device='mps',data_wait_report_seconds=.025)))
        assert await asyncio.to_thread(started.wait,1)
        # If HF HTTP fetch held the MPS thread, the second model task would stall.
        result=await asyncio.wait_for(mps.model_task(lambda: 99,device='mps'),timeout=1)
        release.set()
        return result,await read
    assert asyncio.run(run())==(99,7)
