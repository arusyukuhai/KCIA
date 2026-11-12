"""KCIA v127: a normal localhost origin for WKWebView, no API or font hosting."""
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

from patchlib.webview_gui import serve_gui_origin, compose_html, run_webview


def test_local_page_serves_exact_html_only():
    html = compose_html()
    with serve_gui_origin(html) as url:
        parsed = urlsplit(url)
        assert parsed.hostname == '127.0.0.1'
        assert parsed.path.count('/') == 1
        with urllib.request.urlopen(url, timeout=2) as response:
            assert response.status == 200
            assert response.headers['Cache-Control'] == 'no-store'
            assert response.headers['Referrer-Policy'] == 'no-referrer'
            assert response.read().decode('utf-8') == html
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(url + '/other', timeout=2)
        assert err.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(urllib.request.Request(url, data=b'attack'), timeout=2)
        assert err.value.code == 405
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(urllib.request.Request(url, headers={'Host':'evil.example'}),timeout=2)
        assert err.value.code == 421
    with pytest.raises((urllib.error.URLError, OSError)):
        urllib.request.urlopen(url, timeout=.3)


def test_page_uses_loopback_on_explicit_preference_and_native_bridge_unchanged(tmp_path,monkeypatch):
    monkeypatch.setenv('KCIA_GUI_HTTP_ORIGIN','1')
    class Native:
        settings = {}
        def create_window(self, title, **kwargs):
            assert 'url' in kwargs and 'html' not in kwargs
            with urllib.request.urlopen(kwargs['url'], timeout=2) as response:
                rendered=response.read().decode('utf-8')
                assert 'KCIA' in rendered
                assert 'https://use.typekit.net/wzz8rse.css' in rendered
            assert callable(kwargs['js_api'].new_room)
            self.url=kwargs['url']
            self.window=object()
            return self.window
        def start(self, **kwargs):
            assert kwargs['debug'] is False
            with urllib.request.urlopen(self.url,timeout=2) as response:
                assert response.status==200
    native=Native()
    assert run_webview(workdir=tmp_path,webview_module=native) is native.window
    with pytest.raises((urllib.error.URLError, OSError)):
        urllib.request.urlopen(native.url,timeout=.3)


def test_old_in_memory_mode_remains_available(tmp_path,monkeypatch):
    monkeypatch.setenv('KCIA_GUI_HTTP_ORIGIN','0')
    class Stub:
        settings={}
        def create_window(self,title,**kw):
            assert 'html' in kw and 'url' not in kw
            return object()
        def start(self,**kw):pass
    assert run_webview(workdir=tmp_path,webview_module=Stub()) is not None


def test_chromium_remote_font_stylesheet_from_loopback():
    from playwright.sync_api import sync_playwright
    KIT = 'https://use.typekit.net/wzz8rse.css'
    with serve_gui_origin(compose_html()) as url:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
            page=browser.new_page()
            page.route('https://fonts.googleapis.com/**',lambda r:r.abort())
            page.route(KIT,lambda r:r.fulfill(status=200,content_type='text/css',headers={'Access-Control-Allow-Origin':'*'},body='''
                @font-face {font-family:"myriad-pro-cond";src:local("DejaVu Sans");font-weight:400}
            '''))
            # The CI browser blocks localhost navigation, even when routed.
            # An isolated .test origin tests normal HTTP document semantics.
            test_url='http://kcia-ui.test/gui'
            page.route(test_url,lambda r:r.fulfill(status=200,content_type='text/html',body=compose_html()))
            try:
                page.goto(test_url,wait_until='domcontentloaded')
            except Exception as exc:
                if 'ERR_BLOCKED_BY_ADMINISTRATOR' in str(exc):
                    pytest.skip('Browser CI does not allow HTTP navigation')
                raise
            page.wait_for_function('window.__kciaFontStatus?.phase === "active"',timeout=10000)
            assert page.evaluate('location.origin === "http://kcia-ui.test"')
            assert page.locator('#fontStatus').inner_text()=='FONT ✓'
            assert 'myriad-pro-cond' in page.evaluate('getComputedStyle(document.body).fontFamily')
            browser.close()


def test_local_port_unavailable_falls_back_to_html(tmp_path,monkeypatch):
    import patchlib.webview_gui as wg
    monkeypatch.setenv('KCIA_GUI_HTTP_ORIGIN','1')
    from contextlib import contextmanager
    @contextmanager
    def blocked(html):
        raise OSError('cannot bind loopback')
        yield 'unused'
    monkeypatch.setattr(wg,'serve_gui_origin',blocked)
    class Native:
        settings={}
        def create_window(self,title,**kw):
            assert 'html' in kw and 'url' not in kw
            return 'window'
        def start(self,**kw):pass
    assert run_webview(workdir=tmp_path,webview_module=Native())=='window'
