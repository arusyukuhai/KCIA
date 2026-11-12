import json
from pathlib import Path
import sqlite3

import pytest
from patchlib.journal_pressure_v87 import should_pause_local_training_for_epoch, should_propose_for_pressure
from patchlib.journal_recovery_status_v89 import inspect, report


def test_pressure_preempt_and_hold():
    gib=1024**3
    assert should_propose_for_pressure(2*gib,4*gib,0)
    assert not should_pause_local_training_for_epoch(3*gib,4*gib)
    assert should_pause_local_training_for_epoch(4*gib,4*gib)
    assert not should_pause_local_training_for_epoch(4*gib,0)
    with pytest.raises(ValueError):
        should_pause_local_training_for_epoch(-1,4*gib)


def test_status_detects_legacy_and_pinned_gui(tmp_path, capsys):
    room=tmp_path/'.kcia-gui'/'rooms'/'abc123'
    peer=room/'peers'/'peer-1234567890abcdef'
    peer.mkdir(parents=True)
    journal=peer/'events.sqlite3'
    with sqlite3.connect(journal) as db:
        db.execute('CREATE TABLE meta(name TEXT PRIMARY KEY,value TEXT)')
        db.execute('CREATE TABLE events(origin TEXT,seq INTEGER)')
        db.execute("INSERT INTO meta VALUES('archive_used_bytes','64')")
        db.execute("INSERT INTO meta VALUES('last_local_journal_cap_bytes','128')")
    initial=inspect(journal)
    assert initial['owner_pinned'] is False
    assert initial['signed_log_usage']==(64,128,0)
    (room/'manifest.json').write_text(json.dumps({'generation':{'owner':'example'}}))
    assert inspect(journal)['owner_pinned'] is True
    (Path(str(journal)+'.generation-auto-v88.json')).write_text(json.dumps(
        {'payload':{'keys':{'a':'1','b':'2','c':'3'}},'signature':'test'}))
    result=report(journal)
    assert result['signed_roster_present'] is True
    assert 'no confirmed epoch certificate' in capsys.readouterr().out


def test_status_never_treats_symlink_as_trusted(tmp_path):
    j=tmp_path/'a.sqlite3'
    j.symlink_to(tmp_path/'missing')
    s=inspect(j)
    assert s['signed_log_usage'] is None
