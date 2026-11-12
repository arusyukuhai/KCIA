"""v125 regression: verify actual FontFace loading, not just CSS declarations.

No Adobe font files are included; a freely installed local test face mocks the
remote CSS so the test can run offline. A real Mac+Adobe CDN probe remains
necessary for validation of the user's personal Adobe Web Project.
"""
from patchlib.webview_gui import compose_html

KIT = 'https://use.typekit.net/wzz8rse.css'


def test_adobe_csp_allows_font_data_uris_and_usage_tracking():
    html = compose_html()
    assert "font-src data: https://fonts.gstatic.com https://use.typekit.net" in html
    assert "img-src data: https://p.typekit.net" in html
    assert 'id="fontStatus"' in html
    assert 'window.__kciaFontStatus' in html
    assert 'document.fonts.load(weight +' in html
    assert KIT in html
    assert '"Myriad Pro Condensed"' in html


def test_kit_missing_face_reports_failure_instead_of_fake_success():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
        page = browser.new_page()
        page.route('https://fonts.googleapis.com/**', lambda route: route.abort())
        hits = []
        def kit(route):
            hits.append(route.request.url)
            route.fulfill(status=200, content_type='text/css', body=':root {--kcia-adobe-mock: 1}')
        page.route(KIT, kit)
        page.set_content(compose_html(), wait_until='load')
        page.wait_for_function("window.__kciaFontStatus?.phase === 'missing'", timeout=8000)
        state = page.evaluate("window.__kciaFontStatus")
        assert state['stylesheet'] is True
        assert state['faces'] == 0
        assert 'no myriad-pro-cond @font-face' in state['message']
        assert hits == [KIT]
        browser.close()


def test_mocked_local_font_face_actually_loads_and_ui_reports_success():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
        page = browser.new_page()
        page.route('https://fonts.googleapis.com/**', lambda route: route.abort())
        page.route(KIT, lambda route: route.fulfill(status=200, content_type='text/css', body='''
          @font-face { font-family: "myriad-pro-cond";
            src: local("DejaVu Sans"); font-weight: 400; font-style: normal; }
        '''))
        page.set_content(compose_html(), wait_until='load')
        page.wait_for_function("window.__kciaFontStatus?.phase === 'active'", timeout=8000)
        assert page.locator('#fontStatus').inner_text() == 'FONT ✓'
        assert page.evaluate('window.__kciaFontStatus.faces') >= 1
        assert 'myriad-pro-cond' in page.locator('body').evaluate('(el) => getComputedStyle(el).fontFamily')
        browser.close()


def test_font_status_click_retries_after_network_failure():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
        page = browser.new_page()
        page.route('https://fonts.googleapis.com/**', lambda route: route.abort())
        requests = {'count': 0}
        def kit(route):
            requests['count'] += 1
            if requests['count'] == 1:
                route.abort()
            else:
                route.fulfill(status=200, content_type='text/css', body='''
                  @font-face { font-family: "myriad-pro-cond";
                    src: local("DejaVu Sans"); font-weight:400; }
                ''')
        page.route(KIT, kit)
        page.set_content(compose_html(), wait_until='load')
        page.wait_for_function("window.__kciaFontStatus?.phase === 'error'", timeout=8000)
        page.locator('#fontStatus').click()
        page.wait_for_function("window.__kciaFontStatus?.phase === 'active'", timeout=8000)
        assert requests['count'] >= 2
        browser.close()
