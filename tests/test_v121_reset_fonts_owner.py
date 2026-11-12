"""v121: independent confirmation of destructive reset, fonts and orphan-guard."""
import os
import signal
import time
from pathlib import Path

import pytest

from patchlib.webview_gui import KCIABridge, compose_html, run_webview
from patchlib.webview_controller import WebviewController
from patchlib.reset_local_data import preview_local_data
from patchlib import gui_worker_guard


def test_fonts_use_real_local_japanese_fallback():
    content=compose_html()
    assert 'Hiragino Kaku Gothic ProN' in content
    assert 'Yu Gothic' in content
    assert 'Noto Sans JP' in content
    assert '"myriad-pro-cond"' in content
    assert 'Noto+Sans+JP' in content
    assert 'font-synthesis: none;' in content
    assert 'font-family: var(--jp)' in content


def test_full_reset_requires_preview_and_exact_confirmation(tmp_path):
    gui=tmp_path/'.kcia-gui'
    (gui/'rooms'/'room-a').mkdir(parents=True)
    (gui/'rooms'/'room-a'/'weight.safetensors').write_bytes(b'original')
    ctrl=WebviewController(tmp_path)
    bridge=KCIABridge(ctrl)
    with pytest.raises(ValueError,match='confirm'):
        bridge.request_full_reset({'confirmation':'delete','preview_id':'fake'})
    result=bridge.preview_full_reset({'include_cli_defaults':False})
    assert result['total_bytes']>=8
    assert result['targets'][0]['path']==str(gui)
    assert (gui/'rooms'/'room-a'/'weight.safetensors').exists()
    with pytest.raises(ValueError,match='Preview'):
        bridge.request_full_reset({'confirmation':'DELETE KCIA','preview_id':'wrong'})
    armed=bridge.request_full_reset({'confirmation':'DELETE KCIA','preview_id':result['preview_id']})
    assert armed['scheduled']
    assert ctrl._full_reset_armed
    assert bridge._pending_full_reset is False
    assert (gui/'rooms'/'room-a'/'weight.safetensors').exists(), 'must not erase before window closes'
    with pytest.raises(RuntimeError,match='scheduled'):
        ctrl.start({'room':'bad'})


def test_reset_does_not_start_any_worker_and_erases_only_after_exit(tmp_path):
    gui=tmp_path/'.kcia-gui'
    gui.mkdir()
    (gui/'persisted.txt').write_text('test')
    (tmp_path/'kcia.py').write_text('keep')

    class DummyWindow:
        def minimize(self):pass
        def maximize(self):pass
        def restore(self):pass
        def destroy(self):pass
    class FakeWebview:
        settings={}
        def create_window(self, title,**kw):
            self.bridge=kw['js_api']
            return DummyWindow()
        def start(self,**kw):
            # Simulates a GUI user reviewing and explicitly confirming.
            self.start_called=True
            prev=self.bridge.preview_full_reset({'include_cli_defaults':False})
            self.bridge.request_full_reset({'preview_id':prev['preview_id'],
                'confirmation':'DELETE KCIA','include_cli_defaults':False})
            assert (gui/'persisted.txt').is_file()
    view=FakeWebview()
    run_webview(workdir=tmp_path,webview_module=view)
    assert view.start_called
    assert not gui.exists()
    assert (tmp_path/'kcia.py').exists()


def test_reset_cannot_be_armed_while_worker_active(tmp_path,monkeypatch):
    gui=tmp_path/'.kcia-gui';gui.mkdir()
    ctrl=WebviewController(tmp_path)
    monkeypatch.setattr(ctrl.train,'status',lambda:True)
    bridge=KCIABridge(ctrl)
    with pytest.raises(RuntimeError,match='学習'):
        bridge.preview_full_reset()
    assert not ctrl._full_reset_armed


def test_read_only_reset_preview_keeps_cli_data(tmp_path):
    (tmp_path/'.kcia-gui').mkdir()
    cwd=tmp_path/'scripts';cwd.mkdir()
    (cwd/'elastic-events.sqlite3').write_bytes(b'journal')
    preview=preview_local_data(root=tmp_path,include_cli_defaults=True,cli_cwd=cwd)
    assert len(preview['targets'])==2
    assert (cwd/'elastic-events.sqlite3').exists()


def test_parent_guard_no_env_no_thread(monkeypatch):
    monkeypatch.delenv('KCIA_GUI_PARENT_PID',raising=False)
    assert gui_worker_guard.enable_gui_parent_watchdog() is False


def test_parent_guard_rejects_orphaned_trainer_on_start(monkeypatch):
    monkeypatch.setenv('KCIA_GUI_PARENT_PID','99999999')
    with pytest.raises(SystemExit,match='owner already stopped'):
        gui_worker_guard.enable_gui_parent_watchdog()
    assert os.environ.get('KCIA_GUI_PARENT_PID') is None


def test_parent_guard_requests_sigterm_after_gui_exit(monkeypatch):
    state={'parent':4444,'signals':[]}
    monkeypatch.setenv('KCIA_GUI_PARENT_PID','4444')
    monkeypatch.setattr(gui_worker_guard.os,'getppid',lambda:state['parent'])
    monkeypatch.setattr(gui_worker_guard.os,'kill',lambda pid,sig:state['signals'].append((pid,sig)))
    assert gui_worker_guard.enable_gui_parent_watchdog(interval=.01)
    state['parent']=1
    deadline=time.monotonic()+1
    while not state['signals'] and time.monotonic()<deadline:time.sleep(.005)
    assert state['signals']==[(os.getpid(),signal.SIGTERM)]

@pytest.mark.skipif(os.name!='posix',reason='flock only on POSIX')
def test_other_idle_gui_blocks_reset(tmp_path):
    import io
    from patchlib.webview_gui import gui_session_lock
    from patchlib.reset_local_data import reset_local_data
    with gui_session_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another KCIA GUI'):
            reset_local_data(root=tmp_path,yes=True,stream=io.StringIO())
    assert reset_local_data(root=tmp_path,yes=True,stream=io.StringIO())==0
