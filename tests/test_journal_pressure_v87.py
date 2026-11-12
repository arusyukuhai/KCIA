import pytest
from patchlib.journal_pressure_v87 import should_propose_for_pressure, cap_message


def test_early_epoch_threshold_and_debounce():
    gib = 1024**3
    assert not should_propose_for_pressure(3*gib, 0, 0)
    assert should_propose_for_pressure(2*gib, 4*gib, 0)
    assert not should_propose_for_pressure(gib, 4*gib, 0)
    assert should_propose_for_pressure(3*gib, 4*gib, 0)
    assert not should_propose_for_pressure(3*gib + 10, 4*gib, 3*gib)
    assert should_propose_for_pressure(3*gib + 128*1024**2, 4*gib, 3*gib)
    with pytest.raises(ValueError):
        should_propose_for_pressure(-1, 1, 0)


def test_quota_diagnostic_distinguishes_physical_disk():
    message=cap_message(4*1024**3,128,4*1024**3)
    assert 'durable update log is full' in message
    assert 'quota' in message
    assert 'disk-report' in message
    assert 'checkpoint/epoch' in message


def test_event_journal_cap_fails_without_side_effect(tmp_path):
    from patchlib.elastic import EventJournal
    from patchlib.core import PatchError
    from patchlib.disk_health_v86 import _signed_log_quota
    journal = EventJournal(tmp_path/'events.db', 'a'*64, max_bytes=100,
                           min_free_mib=0)
    try:
        with pytest.raises(PatchError, match='durable update log is full') as exc:
            journal.append('peer1', 1, b'1'*101)
        assert 'configured cap=' in str(exc.value)
        assert journal.count() == 0
        assert _signed_log_quota(journal.path) == (0, 100, 0)
    finally:
        journal.close()
