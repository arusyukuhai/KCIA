"""Stop a GUI-owned training process if its owner abruptly disappears.

Standalone CLI training has no KCIA_GUI_PARENT_PID, so it is unaffected.
The trainer installs its existing SIGTERM handler later during startup; when
that handler is present, this requests its normal graceful shutdown.
"""
from __future__ import annotations
import os
import signal
import sys
import threading


def enable_gui_parent_watchdog(*, interval: float = 2.0) -> bool:
    raw = os.environ.pop('KCIA_GUI_PARENT_PID', None)
    if not raw:
        return False
    try:
        expected = int(raw)
    except (TypeError, ValueError):
        raise SystemExit('Invalid GUI parent PID; refusing unattended training')
    if expected <= 1 or os.getppid() != expected:
        raise SystemExit('KCIA GUI owner already stopped; refusing unattended training')

    def monitor():
        import time
        while True:
            time.sleep(interval)
            if os.getppid() == expected:
                continue
            print('[KCIA] GUI owner disappeared; requesting training shutdown',
                  file=sys.stderr, flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return

    threading.Thread(target=monitor, daemon=True, name='KCIAGUIParentWatchdog').start()
    return True
