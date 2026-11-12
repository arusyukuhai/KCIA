"""Embedded WebView desktop application (Electron/Tauri-like architecture).

User interface: bundled HTML/CSS/JS rendered by the operating system's WebView.
Backend: a limited pywebview JS-Python bridge, no HTTP socket, no localhost API.
Remote model code is never executed from the UI; only content hash is registered.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import secrets
import queue
import threading
import time
import traceback

from .webview_controller import WebviewController

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / 'web'


class KCIABridge:
    """Minimal JS API; exposed method names are intentionally explicit.

    pywebview calls exposed methods on background threads. WebviewController
    serializes operations needing shared state using its internal RLock.
    """

    def __init__(self, controller: WebviewController | None = None):
        self._controller = controller or WebviewController()
        self._window = None
        self._window_actions = queue.Queue(maxsize=16)
        self._window_worker = None
        self._is_maximized = False
        self._window_closed = False
        self._window_closing_requested = False
        self._window_worker_guard = threading.Lock()
        self._pending_full_reset = None
        self._reset_preview = None

    def bind_window(self, window):
        """Bind exactly the window that owns this bridge (not webview.windows[0])."""
        if self._window is not None:
            raise RuntimeError('window is already bound')
        self._window = window

    def _run_window_actions(self):
        """Keep native Cocoa/GTK calls out of pywebview's synchronous JS RPC.

        Destroying the window before a JS API call returns can strand the
        waiting WebView and hang the UI. A single queue also preserves order
        of rapid minimize / maximize / close clicks.
        """
        while True:
            action, window = self._window_actions.get()
            try:
                # Let the JS API reply return to the renderer before touching
                # the native window (especially for destroy and fullscreen).
                time.sleep(0.09)
                if self._window_closed:
                    continue
                if action == 'minimize':
                    window.minimize()
                elif action == 'toggle_maximize':
                    if self._is_maximized:
                        window.restore()
                    else:
                        window.maximize()
                    self._is_maximized = not self._is_maximized
                elif action == 'close':
                    self._window_closed = True
                    window.destroy()
                    return
            except Exception:
                # The GUI must not silently hang if the OS rejects an action.
                print('[KCIA] Native window action failed:\n' + traceback.format_exc(), file=sys.stderr)
                if action == 'close':
                    self._window_closed = False
                    self._window_closing_requested = False
            finally:
                self._window_actions.task_done()

    def window_action(self, payload):
        """Queue OS window actions without waiting inside the JS-Python RPC."""
        if not isinstance(payload, dict) or payload.get('action') not in ('minimize', 'toggle_maximize', 'close'):
            raise ValueError('unsupported window operation')
        action = payload['action']
        with self._window_worker_guard:
            if self._window is None or self._window_closed or self._window_closing_requested:
                raise RuntimeError('native window is not ready')
            try:
                self._window_actions.put_nowait((action, self._window))
            except queue.Full as exc:
                raise RuntimeError('window action queue is busy') from exc
            if action == 'close':
                self._window_closing_requested = True
            if self._window_worker is None or not self._window_worker.is_alive():
                self._window_worker = threading.Thread(target=self._run_window_actions,
                    name='KCIANativeWindow', daemon=True)
                self._window_worker.start()
        return {'ok': True, 'queued': True, 'action': action}

    def preview_full_reset(self, payload=None):
        """List all KCIA-managed local data; nothing is deleted yet."""
        from .reset_local_data import preview_local_data
        import secrets
        include_cli = bool((payload or {}).get('include_cli_defaults', False))
        with self._controller.lock:
            if self._controller.train.status():
                raise RuntimeError('学習ワーカーを切断して終了するまで、全データ削除はできません。')
            if not self._controller.workdir.is_dir():
                raise RuntimeError('KCIA has no local data to reset yet')
            preview = preview_local_data(root=self._controller.workdir,
                include_cli_defaults=include_cli, cli_cwd=ROOT)
            preview_id = secrets.token_urlsafe(18)
            self._reset_preview = (preview_id, include_cli)
            return {**preview, 'preview_id': preview_id}

    def request_full_reset(self, payload):
        """Arm irreversible reset after a second explicit confirmation."""
        if not isinstance(payload, dict) or payload.get('confirmation') != 'DELETE KCIA':
            raise ValueError('Type DELETE KCIA to confirm permanent deletion')
        with self._controller.lock:
            current = self._reset_preview
            if (not current or payload.get('preview_id') != current[0] or
                    bool(payload.get('include_cli_defaults',False)) != current[1]):
                raise ValueError('Preview the selected deletion targets again first')
            self._controller.arm_full_reset()
            self._pending_full_reset = current[1]
            return {'scheduled':True,'message':'Close KCIA to erase its local data'}

    def ping(self, _payload=None):
        return {'ok': True, 'backend': 'KCIA Python', 'python': sys.executable}

    def defaults(self, _payload=None):
        return self._controller.settings()

    def configure_network(self, payload):
        return self._controller.configure_network(payload)

    def probe_network(self, payload):
        return self._controller.probe_network(payload)

    def new_room(self, _payload=None):
        from .roomid import create_room_id
        return {'room': create_room_id()}

    def select_model(self,payload):
        return self._controller.select_model(payload)

    def model_compatibility(self,payload):
        return self._controller.model_compatibility(payload)

    def register_room(self, payload: dict):
        return self._controller.register_room(payload)

    def list_rooms(self, payload: dict):
        return self._controller.list_rooms(payload)

    def save_room(self, payload: dict):
        return self._controller.save(payload)

    def join_room(self, payload: dict):
        return self._controller.load_room(payload)

    def register_code(self, payload: dict):
        return self._controller.upload_code(payload)

    def pick_training_file(self, payload: dict):
        """Native file selector; never read a large dataset into the WebView."""
        if not isinstance(payload,dict) or payload.get('kind') not in ('sft','ppo','critic'):
            raise ValueError('Invalid training file kind')
        import webview
        if not getattr(webview,'windows',None):
            raise RuntimeError('A desktop window is required for file selection')
        window=webview.windows[0]
        if payload['kind']=='critic':
            mask=('Safetensors (*.safetensors)', 'All files (*.*)')
        else:
            mask=('JSON Lines (*.jsonl)', 'All files (*.*)')
        dialog_type=getattr(webview,'OPEN_DIALOG',None)
        if dialog_type is None:
            dialog_type=webview.FileDialog.OPEN
        result=window.create_file_dialog(dialog_type,allow_multiple=False,file_types=mask)
        return {'path':str(Path(result[0]).expanduser().resolve()) if result else ''}

    def mixed_settings(self, payload: dict):
        return self._controller.mixed_settings(payload)

    def start_training(self, payload: dict):
        return self._controller.start(payload)

    def training_status(self, _payload=None):
        return self._controller.status()

    def training_history(self, payload):
        return self._controller.training_history(payload)

    def storage_paths(self, payload: dict):
        return self._controller.storage_paths(payload)

    def open_storage(self, payload: dict):
        return self._controller.open_storage(payload)

    def copy_logs(self, _payload=None):
        return self._controller.copy_logs()

    def save_logs(self, _payload=None):
        return self._controller.save_logs()

    def start_benchmark(self, payload: dict):
        return self._controller.start_benchmark(payload)

    def benchmark_status(self, _payload=None):
        return self._controller.benchmark_status()

    def stop_training(self, _payload=None):
        return self._controller.pause_training()

    def resume_training(self, _payload=None):
        return self._controller.resume_training()

    def disconnect_training(self, _payload=None):
        return self._controller.stop()

    def local_models(self, _payload=None):
        return self._controller.local_models()

    def device_capabilities(self, _payload=None):
        return self._controller.device_capabilities()

    def inspect_dataset(self, payload):
        return self._controller.inspect_dataset(payload)

    def recommend_training(self, payload):
        return self._controller.recommend_training(payload)

    def pin_mixed_dataset(self, payload):
        return self._controller.pin_mixed_dataset(payload)

    def probe_mixed_dataset(self, payload):
        return self._controller.probe_mixed_dataset(payload)

    def discover_hub_assets(self, payload):
        return self._controller.discover_hub_assets(payload)

    def search_reward_models(self, payload):
        return self._controller.search_reward_models(payload)

    def inspect_reward_model(self, payload):
        return self._controller.inspect_reward_model(payload)

    def search_datasets(self, payload):
        return self._controller.search_datasets(payload)

    def evaluate(self, payload):
        return self._controller.evaluate(payload)

    def available_checkpoints(self, payload):
        return self._controller.available_checkpoints(payload)

    def request_eval_snapshot(self, payload):
        return self._controller.request_eval_snapshot(payload)

    def eval_snapshot_status(self, payload):
        return self._controller.eval_snapshot_status(payload)

    def eval_status(self, _payload=None):
        return self._controller.eval_status()

    def cancel_eval(self, _payload=None):
        return self._controller.cancel_eval()



def compose_html() -> str:
    """Use exclusively bundled GUI CSS/JS; no third-party privileged scripts."""
    template = (WEB / 'desktop_webview.html').read_text(encoding='utf-8')
    css = (WEB / 'tailwind.cdn.css').read_text(encoding='utf-8')
    js = (WEB / 'desktop_webview.js').read_text(encoding='utf-8')
    # Trusted UI assets only; no user-supplied markup is ever interpolated.
    if '</style' in css.lower() or '</script' in js.lower():
        raise ValueError('unexpected closing HTML tag in a bundled resource')
    # pywebview dynamically builds its JS API with `new Function(...)`; it needs
    # 'unsafe-eval'. Without it the HTML loads, but Python methods never appear.
    # No untrusted HTML is interpolated; the page embeds only vetted local assets.
    # JS calls to Python do not require a socket/HTTP connection.
    meta = '''<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline' https://fonts.googleapis.com https://use.typekit.net; font-src data: https://fonts.gstatic.com https://use.typekit.net; img-src data: https://p.typekit.net; connect-src 'none'; worker-src 'none'; form-action 'none'; base-uri 'none'; object-src 'none'; frame-src 'none'">'''
    template = template.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n'+meta, 1)
    template = template.replace('</head>', f'<style>\n{css}\n</style>\n</head>', 1)
    template = template.replace('</body>', f'<script>\n{js}\n</script>\n</body>', 1)
    # Adobe Fonts' CSS kit stays remote (licensed service); no font assets are
    # bundled. Load after page paint so a dead network never hangs WebView.
    # IMPORTANT: the font-src data: and img-src p.typekit.net CSP allowances
    # are required by Adobe Fonts; link.onload by itself does NOT prove that
    # a requested family/weight has been loaded.
    font_loader = """<script>
(function () {
  'use strict';
  const KIT = 'https://use.typekit.net/wzz8rse.css';
  const FAMILY = 'myriad-pro-cond';
  const TEST_GLYPHS = 'KCIA abcdefghijklmnopqrstuvwxyz 0123456789';
  let stylesheet = null;
  let generation = 0;
  const status = {phase:'pending', message:'Waiting to load Adobe Fonts', faces:0,
    stylesheet:false, family:FAMILY, weights:[], url:KIT,
    origin:location.origin, stage:'init'};
  window.__kciaFontStatus = status;
  const labelFor = {
    active: 'FONT ✓', partial: 'FONT WGT?', loading: 'FONT …',
    missing: 'FONT KIT?', error: 'FONT CDN!', unsupported: 'FONT API!'
  };
  function report(phase, message, faces) {
    status.phase = phase;
    status.message = message;
    if (typeof faces === 'number') status.faces = faces;
    const control = document.getElementById('fontStatus');
    if (control) {
      control.textContent = labelFor[phase] || 'FONT !';
      control.className = 'font-status font-' + phase;
      control.title = message + ' — click to retry';
      control.setAttribute('aria-label', 'Adobe Fonts: ' + message + '. Click to retry.');
    }
  }
  function matchingFaces() {
    return Array.from(document.fonts || []).filter(face =>
      String(face.family || '').replace(/["']/g, '').trim().toLowerCase() === FAMILY);
  }
  function inspectFont(token, retry) {
    if (token !== generation) return;
    status.stage='face';
    if (!document.fonts || !document.fonts.load) {
      report('unsupported', 'This WebView does not support the CSS Font Loading API');
      return;
    }
    // FontFaceSet may lag behind link.onload (especially on WKWebView).
    // Wait for style application, and retry discovery briefly rather than
    // wrongly reporting that a valid kit has no matching face.
    const declared = matchingFaces();
    if (!declared.length) {
      if (retry < 4) {
        setTimeout(() => inspectFont(token, retry + 1), [150,350,700,1200][retry]);
        return;
      }
      const known = Array.from(document.fonts || []).map(f => f.family).filter(Boolean);
      status.weights = [];
      report('missing', 'Adobe CSS responded, but no '+FAMILY+' @font-face was registered. '
        + 'Check kit wzz8rse (including the style), or WKWebView CSS loading. '
        + (known.length ? 'Other families: '+Array.from(new Set(known)).slice(0,8).join(', ') : 'No remote faces registered.'), 0);
      return;
    }
    const weights = Array.from(new Set(declared.map(f => String(f.weight || '400'))));
    status.weights = weights;
    const regular = weights.some(w => w === '400' || w === 'normal' || w === '100 900');
    const weight = regular ? '400' : (/^\\d{3}$/.test(weights[0]) ? weights[0] : '400');
    status.stage='font';
    let timeout = null;
    Promise.race([
      document.fonts.load(weight + ' 16px "' + FAMILY + '"', TEST_GLYPHS),
      new Promise((_, reject) => {timeout = setTimeout(() => reject(new Error('Font download timeout (12 s)')), 12000);})
    ]).then(faces => {
      if (token !== generation) return;
      status.stage='font';
      const loaded = Array.from(faces || []).filter(f =>
        String(f.family).replace(/["']/g, '').toLowerCase() === FAMILY && f.status === 'loaded');
      if (!loaded.length) {
        report('error', FAMILY+' was declared, but its font file did not load. Check Adobe Fonts network access and the Web Project.',0);
      } else if (!regular) {
        report('partial', FAMILY+' loaded with weight(s): '+weights.join(', ')+'. Regular 400 is missing from the kit; add it for normal text.',loaded.length);
      } else {
        report('active', FAMILY+' loaded from Adobe Fonts ('+loaded.length+' face(s); weight '+weight+')',loaded.length);
      }
    }).catch(err => {
      status.stage='font';
      if (token === generation) report('error', FAMILY+' font file could not be loaded: '+String(err.message || err));
    }).finally(() => {if (timeout !== null) clearTimeout(timeout);});
  }
  function loadAdobeFont() {
    const token = ++generation;
    if (stylesheet) stylesheet.remove();
    status.stylesheet = false;
    status.faces = 0;
    status.weights = [];
    status.stage='css';
    report('loading', 'Loading Adobe Fonts stylesheet: '+KIT+' (page origin: '+location.origin+')');
    stylesheet = document.createElement('link');
    stylesheet.rel = 'stylesheet';
    stylesheet.href = KIT;
    // Dynamically appended after initial render: already non-render-blocking.
    // Do NOT set media=print here; on WebKit a media switch after onload may
    // expose @font-face *after* the following JavaScript inspection.
    stylesheet.onload = function () {
      if (token !== generation) return;
      status.stylesheet = true;
      requestAnimationFrame(() => requestAnimationFrame(() => inspectFont(token, 0)));
    };
    stylesheet.onerror = function () {
      if (token === generation) report('error', 'Adobe Fonts CSS could not be downloaded or was blocked: '+KIT+' (page origin: '+location.origin+'). Try: curl -IL '+KIT);
    };
    document.head.appendChild(stylesheet);
  }
  function ready() {
    const control = document.getElementById('fontStatus');
    if (control) control.addEventListener('click', loadAdobeFont);
    loadAdobeFont();
    const noto = document.createElement('link');
    noto.rel = 'stylesheet';
    noto.href = 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;600;700&display=swap';
    document.head.appendChild(noto);
  }
  if (document.readyState === 'complete') ready();
  else window.addEventListener('load', ready, {once:true});
})();
</script>"""

    return template.replace('</body>', font_loader+'\n</body>', 1)



@contextlib.contextmanager
def gui_session_lock(workdir: Path):
    """Shared flock so an all-data reset rejects another open v121 GUI."""
    if os.name != 'posix':
        yield
        return
    import fcntl
    directory = Path(workdir) / '.kcia-gui'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / 'app.lock'
    fd = os.open(path, os.O_RDWR|os.O_CREAT|getattr(os, 'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def serve_gui_origin(html: str):
    """Serve only the current GUI page from loopback, no API or filesystem.

    WKWebView treats HTML loaded as an in-memory document differently from
    a normal HTTP page. A loopback origin gives remote font resources an
    ordinary page origin without hosting/redistributing font files. The
    privileged pywebview bridge is *not* exposed by this HTTP server.
    """
    body = html.encode('utf-8')
    secret_path = '/' + secrets.token_urlsafe(28)

    class PageHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            expected_host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != expected_host:
                self.send_error(421, 'Invalid Host')
                return
            if self.path != secret_path:
                self.send_error(404, 'Not Found')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.send_error(405, 'Method Not Allowed')

        def log_message(self, *args):
            pass  # Never log the random session URL.

    server = ThreadingHTTPServer(('127.0.0.1', 0), PageHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever,
        name='KCIA-local-gui-page', daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}{secret_path}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_webview(*, workdir: Path | None = None, webview_module=None):
    """Launch native window. Import pywebview lazily; unit tests can use a stub."""
    if webview_module is None:
        try:
            import webview as webview_module
        except ImportError as exc:
            raise RuntimeError(
                'GUI requires pywebview. Install: python -m pip install "pywebview>=6.1,<7"'
            ) from exc
    controller = WebviewController(workdir)
    bridge = KCIABridge(controller)
    # Explicit drag-region selector for WebKit/GTK/Qt backends. Dragging must
    # never trigger on controls or scrollable content.
    if isinstance(getattr(webview_module, 'settings', None), dict):
        webview_module.settings['DRAG_REGION_SELECTOR'] = '.pywebview-drag-region'
        webview_module.settings['DRAG_REGION_DIRECT_TARGET_ONLY'] = False
    safe_mode = os.environ.get('KCIA_GUI_SAFE', '').strip() == '1'
    # On macOS, use a real loopback HTTP page origin instead of loading an
    # HTML string into WKWebView. This avoids WebKit about:blank/local-origin
    # quirks when fetching external Adobe Fonts; all app API calls still use
    # the native pywebview bridge, never HTTP. Other renderers retain HTML mode.
    use_http_origin = (sys.platform == 'darwin' and
                       getattr(webview_module, '__name__', '') == 'webview')
    preference = os.environ.get('KCIA_GUI_HTTP_ORIGIN', '').strip()
    if preference == '1':
        use_http_origin = True
    elif preference == '0':
        use_http_origin = False
    page_html = compose_html()
    if safe_mode:
        page_html = page_html.replace('<body>', '<body class="native-titlebar">', 1)

    @contextlib.contextmanager
    def page_source():
        if not use_http_origin:
            yield {'html': page_html}
            return
        with contextlib.ExitStack() as stack:
            try:
                url = stack.enter_context(serve_gui_origin(page_html))
            except OSError as exc:
                # If loopback cannot bind, keep the GUI available; do not hide
                # WebView launch errors or accidentally yield twice.
                print('[KCIA GUI] Local origin unavailable:', exc, file=sys.stderr)
                yield {'html': page_html}
            else:
                yield {'url': url}

    with page_source() as source:
        window = webview_module.create_window(
            'KCIA — Distributed Learning',
            **source,
            js_api=bridge,
            width=1380,
            height=860,
            min_size=(960, 620),
            resizable=True,
            focus=True,
            background_color='#080e0b',
            frameless=not safe_mode,
            easy_drag=False,  # Only the custom header may drag the window.
        )
        bridge.bind_window(window)
        try:
            debug = os.environ.get('KCIA_GUI_DEBUG', '').strip() == '1'
            if debug:
                print('[KCIA GUI] Python:', sys.executable, file=sys.stderr)
                print('[KCIA GUI] Local HTTP origin:', use_http_origin, file=sys.stderr)
                print('[KCIA GUI] Starting pywebview in debug mode', file=sys.stderr)
            with gui_session_lock(controller.workdir):
                webview_module.start(debug=debug)
        finally:
            # Release live journal/profile locks before the irreversible action.
            controller.close()
            if bridge._pending_full_reset is not None:
                if controller.train.status():
                    raise RuntimeError('Reset aborted: training worker has not exited')
                from .reset_local_data import reset_local_data
                # The reset helper checks for another GUI holding a peer lock.
                # It preserves source code, custom paths and shared HF caches.
                reset_local_data(root=controller.workdir, yes=True,
                    include_cli_defaults=bridge._pending_full_reset, cli_cwd=ROOT)
                print('[KCIA] Local data reset complete. Restart the GUI to create new rooms.',flush=True)
    return window
