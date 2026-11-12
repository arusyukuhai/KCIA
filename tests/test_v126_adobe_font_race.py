"""Regression tests for font status: no CSS/face/partial/loaded distinction.

Offline fonts use an installed test face; real Adobe licensing content is not bundled.
"""
import pytest
from patchlib.webview_gui import compose_html

KIT='https://use.typekit.net/wzz8rse.css'

@pytest.fixture()
def browser_page():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page()
        page.route('https://fonts.googleapis.com/**',lambda r:r.abort())
        yield page
        browser.close()

def wait(page,phase):
    page.wait_for_function('(phase)=>window.__kciaFontStatus?.phase===phase',arg=phase,timeout=10000)

def test_condensed_load_from_printless_stylesheet(browser_page):
    page=browser_page
    page.route(KIT,lambda r:r.fulfill(status=200,content_type='text/css',body='''
        @font-face {font-family:"myriad-pro-cond";src:local("DejaVu Sans");font-weight:400}
    '''))
    page.set_content(compose_html(),wait_until='load')
    wait(page,'active')
    assert page.locator('#fontStatus').inner_text()=='FONT ✓'
    assert page.locator('link[href="'+KIT+'"]').get_attribute('media') is None
    assert page.evaluate('window.__kciaFontStatus.weights') == ['400']

def test_kit_missing_family_status_is_explanatory(browser_page):
    page=browser_page
    page.route(KIT,lambda r:r.fulfill(status=200,content_type='text/css',body='''
        @font-face {font-family:"not-the-requested-family";src:local("DejaVu Sans")}
    '''))
    page.set_content(compose_html(),wait_until='load')
    wait(page,'missing')
    assert page.locator('#fontStatus').inner_text()=='FONT KIT?'
    assert 'no myriad-pro-cond @font-face' in page.locator('#fontStatus').get_attribute('title')

def test_if_kit_has_only_bold_face_reports_partial(browser_page):
    page=browser_page
    page.route(KIT,lambda r:r.fulfill(status=200,content_type='text/css',body='''
        @font-face {font-family:"myriad-pro-cond";src:local("DejaVu Sans");font-weight:700}
    '''))
    page.set_content(compose_html(),wait_until='load')
    wait(page,'partial')
    assert page.locator('#fontStatus').inner_text()=='FONT WGT?'
    assert 'Regular 400 is missing' in page.locator('#fontStatus').get_attribute('title')

def test_css_network_failure_status(browser_page):
    page=browser_page
    page.route(KIT,lambda r:r.abort())
    page.set_content(compose_html(),wait_until='load')
    wait(page,'error')
    assert page.locator('#fontStatus').inner_text()=='FONT CDN!'
    assert 'CSS could not be downloaded' in page.locator('#fontStatus').get_attribute('title')

def test_retry_after_css_network_failure(browser_page):
    page=browser_page
    calls={'n':0}
    def reply(route):
        calls['n']+=1
        if calls['n']==1: route.abort()
        else: route.fulfill(status=200,content_type='text/css',body='''
          @font-face {font-family:"myriad-pro-cond";src:local("DejaVu Sans");font-weight:400}
        ''')
    page.route(KIT,reply)
    page.set_content(compose_html(),wait_until='load')
    wait(page,'error')
    page.locator('#fontStatus').click()
    wait(page,'active')
    assert calls['n']==2
