"""Reset CLI must never touch the source tree or linked external storage."""
import io
import os
from pathlib import Path

import pytest

from patchlib.reset_local_data import reset_local_data


def test_preview_is_non_destructive(tmp_path):
    path = tmp_path / '.kcia-gui' / 'rooms' / 'room-a'
    path.mkdir(parents=True)
    (path / 'events.sqlite3').write_bytes(b'journal')
    log = io.StringIO()
    assert reset_local_data(root=tmp_path, stream=log) == 0
    assert (path / 'events.sqlite3').read_bytes() == b'journal'
    assert 'DRY RUN' in log.getvalue()


def test_delete_gui_only_preserves_source_and_shared_hf(tmp_path):
    gui = tmp_path / '.kcia-gui'
    nested = gui / 'rooms' / 'room-a' / 'peers'
    nested.mkdir(parents=True)
    (nested / 'events.sqlite3').write_bytes(b'test journal')
    (gui / 'codes').mkdir()
    (gui / 'codes' / 'my_model.py').write_text('sample')
    (gui / 'telemetry').mkdir()
    (gui / 'telemetry' / 'plot.jsonl').write_text('{}')
    (tmp_path / 'kcia.py').write_text('keep')
    (tmp_path / 'hf_cache').mkdir()
    (tmp_path / 'hf_cache' / 'model.safetensors').write_bytes(b'keep')
    assert reset_local_data(root=tmp_path, yes=True, stream=io.StringIO()) == 0
    assert not gui.exists()
    assert (tmp_path / 'kcia.py').read_text() == 'keep'
    assert (tmp_path / 'hf_cache' / 'model.safetensors').read_bytes() == b'keep'
    assert reset_local_data(root=tmp_path, yes=True, stream=io.StringIO()) == 0


def test_delete_does_not_follow_symlink_inside_gui(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    foreign = tmp_path / 'precious'
    foreign.mkdir()
    (foreign / 'data').write_text('keep')
    gui = root / '.kcia-gui'
    gui.mkdir()
    (gui / 'foreign').symlink_to(foreign, target_is_directory=True)
    assert reset_local_data(root=root, yes=True, stream=io.StringIO()) == 0
    assert (foreign / 'data').read_text() == 'keep'
    assert not gui.exists()


def test_refuse_symlink_gui_root(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'data').write_text('keep')
    (root / '.kcia-gui').symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        reset_local_data(root=root, yes=True, stream=io.StringIO())
    assert (other / 'data').read_text() == 'keep'


def test_refuse_dot_kcia_gui_direct_root(tmp_path):
    gui = tmp_path / '.kcia-gui'
    gui.mkdir()
    with pytest.raises(ValueError, match='PARENT'):
        reset_local_data(root=gui, yes=True, stream=io.StringIO())


def test_cli_defaults_explicit_only(tmp_path):
    storage = tmp_path / 'storage'
    storage.mkdir()
    cli = tmp_path / 'source'
    cli.mkdir()
    journal = cli / 'elastic-events.sqlite3'
    journal.write_bytes(b'data')
    (cli / 'elastic-events.sqlite3.fast-resume').mkdir()
    (cli / 'elastic-events.sqlite3.fast-resume' / 'checkpoint').write_bytes(b'data')
    (cli / 'live-checkpoints').mkdir()
    (cli / 'live-checkpoints' / 'checkpoint').write_bytes(b'data')
    (cli / 'other-journal.sqlite3').write_bytes(b'keep')
    reset_local_data(root=storage, yes=True, cli_cwd=cli, stream=io.StringIO())
    assert journal.exists()
    reset_local_data(root=storage, yes=True, include_cli_defaults=True,
                     cli_cwd=cli, stream=io.StringIO())
    assert not journal.exists()
    assert not (cli / 'elastic-events.sqlite3.fast-resume').exists()
    assert not (cli / 'live-checkpoints').exists()
    assert (cli / 'other-journal.sqlite3').read_bytes() == b'keep'


@pytest.mark.skipif(os.name != 'posix', reason='flock protection is POSIX')
def test_active_peer_lock_blocks_reset(tmp_path):
    import fcntl
    gui = tmp_path / '.kcia-gui' / 'rooms' / 'room-a' / 'peers' / 'peer-abc'
    gui.mkdir(parents=True)
    lockfile = gui / 'active.lock'
    lockfile.touch()
    fd = os.open(lockfile, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match='still running'):
            reset_local_data(root=tmp_path, yes=True, stream=io.StringIO())
    finally:
        os.close(fd)
    assert gui.is_dir()


def test_cli_symlink_blocks_all_deletion_before_gui(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    gui = root / '.kcia-gui'
    gui.mkdir()
    (gui / 'important').write_text('still here')
    cli = tmp_path / 'cli'
    cli.mkdir()
    foreign = tmp_path / 'foreign'
    foreign.write_text('protected')
    (cli / 'elastic-events.sqlite3').symlink_to(foreign)
    with pytest.raises(RuntimeError, match='symbolic link'):
        reset_local_data(root=root, yes=True, include_cli_defaults=True,
                         cli_cwd=cli, stream=io.StringIO())
    assert (gui / 'important').read_text() == 'still here'
    assert foreign.read_text() == 'protected'
