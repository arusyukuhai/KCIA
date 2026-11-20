"""v142: corrupt advisory state and live plots must never abort WAN training."""
import json
from pathlib import Path

import pytest

from patchlib.live_telemetry import LocalTelemetry, recent_samples, history_samples
from patchlib.wan_bandit_v139 import WanCostBandit


@pytest.mark.parametrize('body', ['[]', 'null', '0', 'true', '"a"', '{}',
                                 '{"version": 1, "baseline": 32, "ceiling": 1024, "wire_cost": 0.08, "exploration": 0.04, "selected": 32, "events": 1, "wire_estimate_mib": 1, "history": [null]}'])
def test_invalid_bandit_cache_is_ignored_without_interrupting_training(tmp_path, body):
    p = tmp_path / 'events.sqlite3.wan-bandit-v139.json'
    p.write_text(body)
    b=WanCostBandit(baseline=32,ceiling=1024,path=p)
    assert b.samples == 0
    assert b.selected == 32
    for i in range(32):
        b.step(6-i*0.01,inbound_applied=0)
    assert b.observe(steps=32,signed_bytes=1024,inbound_applied=0)


def test_invalid_stats_are_diagnostic_only_and_graph_remains_parseable(tmp_path):
    p = tmp_path/'metrics.jsonl'
    t=LocalTelemetry(p,sample_seconds=0)
    data={'verify_ms':float('inf'), 'apply_ms':float('nan'),
          'push_mib':float('nan'),'repair_mib':-float('inf'),
          'local_seal_ms':None,'working_save_ms':'not a number',
          'aggregation_saved_mib':float('nan'),'lock_wait_ms':float('inf')}
    assert t.record(steps=1,loss=None,emitted=0,inbound_applied=0,
                    inbox=0,channels=1,**data)
    t.close()
    row=recent_samples(p)[-1]
    assert row['loss'] is None
    for field in data:
        assert row[field]==0.0
    assert history_samples(p,window_seconds=0)[-1]['steps']==1


def test_bad_finished_tail_row_is_skipped_by_latest_history_and_restore(tmp_path):
    p=tmp_path/'metrics.jsonl'
    p.write_text(''.join(json.dumps({'steps':i,'elapsed_s':float(i),
                                      'state':'training','loss':3.0})+'\n'
                         for i in range(10))+'{"loss":"truncated"}\n')
    assert [row['steps'] for row in recent_samples(p,max_samples=1)]==[9]
    assert history_samples(p,window_seconds=0)[-1]['steps']==9
    assert recent_samples(p,max_samples=0)==[]
    # The writer appends after stripping truly partial last records; a valid
    # earlier row is still discoverable when a completed malformed line exists.
    t=LocalTelemetry(p,append=True,sample_seconds=0)
    assert t.steps_offset==10
    assert t.record(steps=0,loss=2.0,emitted=1,inbound_applied=0,inbox=0,channels=0)
    t.close()
    assert recent_samples(p,max_samples=1)[-1]['steps']==10


def test_corrupt_json_roots_inside_full_history_do_not_crash_graph(tmp_path):
    p=tmp_path/'history.jsonl'
    p.write_text(''.join(
        (('null' if i%3==0 else '[]' if i%7==0 else
          json.dumps({'steps':i,'elapsed_s':float(i),
                      'state':'training','loss':2.0})))+'\n'
        for i in range(1000)))
    result=history_samples(p,window_seconds=300,max_points=120)
    assert result
    assert result[-1]['steps']==998
    assert all(isinstance(row,dict) for row in result)
