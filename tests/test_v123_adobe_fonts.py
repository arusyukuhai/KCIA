"""Contract tests for the user's Adobe Fonts kit and native fallback."""
from patchlib.webview_gui import compose_html

KIT = "https://use.typekit.net/wzz8rse.css"

def test_adobe_kit_is_loaded_without_blocking_gui():
    html = compose_html()
    assert KIT in html
    assert "fonts.cdnfonts.com" not in html
    assert "style-src 'unsafe-inline' https://fonts.googleapis.com https://use.typekit.net" in html
    assert "font-src data: https://fonts.gstatic.com https://use.typekit.net" in html
    assert "window.addEventListener('load'" in html
    assert '"myriad-pro-cond"' in html
    assert '"Noto Sans JP"' in html

def test_adobe_kit_requested_and_family_applied_in_chromium():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page()
        requested=[]
        def stylesheet(route):
            requested.append(route.request.url)
            route.fulfill(status=200, content_type='text/css',body=':root{--adobe-kit-received:1;}')
        page.route(KIT,stylesheet)
        page.route('https://fonts.googleapis.com/**',lambda route:route.abort())
        page.set_content(compose_html(),wait_until='load')
        page.wait_for_function("document.querySelector(\"link[href=\'"+KIT+"\']\") !== null")
        page.wait_for_function('getComputedStyle(document.documentElement).getPropertyValue("--adobe-kit-received").trim() === "1"')
        result=page.evaluate('''() => ({
            root:getComputedStyle(document.documentElement).getPropertyValue('--jp').trim(),
            body:getComputedStyle(document.body).fontFamily,
            input:getComputedStyle(document.querySelector('input')).fontFamily,
            csp:document.querySelector('meta[http-equiv="Content-Security-Policy"]').content
        })''')
        assert requested and all(url == KIT for url in requested)
        for key in ('root','body','input'):
            assert 'myriad-pro-cond' in result[key]
        assert 'use.typekit.net' in result['csp']
        browser.close()
