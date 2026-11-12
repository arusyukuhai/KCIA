"""Shared URL and cross-process file locking constraints for WebView."""
from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit


def check_signaling_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 512:
        raise ValueError('invalid signaling URL')
    parts = urlsplit(url)
    if (parts.scheme not in ('ws','wss') or not parts.hostname or
        parts.username or parts.password or parts.fragment):
        raise ValueError('invalid signaling URL')
    if parts.scheme == 'ws' and parts.hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('remote signaling requires wss://')
    return url


@contextmanager
def _exclusive_manifest_lock(lockfile: Path):
    """Serialize simultaneous GUI saves of the same room across processes.

    On macOS/Linux the lock belongs to the OS file descriptor and is released
    even if a window crashes. Do not erase this small lock file while running.
    """
    import fcntl
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

