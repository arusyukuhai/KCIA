"""v117: bridge dispatch must never wait for native Cocoa/GTK window methods."""
import threading
import time
from pathlib import Path
from patchlib.webview_gui import KCIABridge, compose_html
from patchlib.webview_controller import WebviewController


def test_native_window_actions_return_before_os_method(tmp_path):
    gate=threading.Event()
    invoked=threading.Event()
    class FakeWindow:
        def __init__(self):
            self.actions=[]
        def minimize(self):
            invoked.set()
            gate.wait(timeout=2)
            self.actions.append('minimize')
        def maximize(self):self.actions.append('maximize')
        def restore(self):self.actions.append('restore')
        def destroy(self):self.actions.append('destroy')
    win=FakeWindow()
    bridge=KCIABridge(WebviewController(tmp_path))
    bridge.bind_window(win)
    start=time.monotonic()
    assert bridge.window_action({'action':'minimize'})['queued'] is True
    assert time.monotonic()-start < 0.075, 'JS RPC must not wait for the OS event'
    assert invoked.wait(timeout=1)
    # Minimize is still pending, but subsequent JS RPCs are not blocked.
    start=time.monotonic()
    assert bridge.window_action({'action':'toggle_maximize'})['queued'] is True
    assert time.monotonic()-start < 0.075
    gate.set()
    bridge._window_actions.join()
    assert win.actions==['minimize','maximize']
    assert bridge.window_action({'action':'toggle_maximize'})['queued'] is True
    assert bridge.window_action({'action':'close'})['queued'] is True
    bridge._window_actions.join()
    assert win.actions==['minimize','maximize','restore','destroy']


def test_native_window_bridge_has_strict_validation(tmp_path):
    bridge=KCIABridge(WebviewController(tmp_path))
    import pytest
    with pytest.raises(RuntimeError,match='not ready'):
        bridge.window_action({'action':'close'})
    with pytest.raises(ValueError):
        bridge.window_action({'action':'launch_shell'})
    with pytest.raises(ValueError):
        bridge.window_action('close')


def test_brand_and_title_are_native_draggable_but_buttons_are_not():
    html=compose_html()
    assert 'class="brand-logo pywebview-drag-region"' in html
    assert 'class="window-drag pywebview-drag-region"' in html
    assert '.window-drag > * {pointer-events: none' in html
    assert '.kcia-logo {' in html and 'pointer-events: none;' in html
    assert 'justify-content: flex-start' in html
    assert 'id="windowClose"' in html
    assert 'method({action})' in html
    assert 'await bridge.method({action})' not in html
