"""KCIA logo should have painted content flush-left (not SVG letterboxing)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchlib.webview_gui import compose_html


def test_logo_css_has_intrinsic_ratio_and_no_height_clamp():
    html = compose_html()
    assert 'class="kcia-logo"' in html
    assert 'max-height: none' in html
    assert 'width: 138px' in html


def test_logo_browser_alignment():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                                    args=['--no-sandbox'])
        for width, height in [(1380,860),(1120,800),(900,720),(640,640)]:
            page = browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
            page.set_content(compose_html(), wait_until='domcontentloaded')
            measurement = page.evaluate('''() => {
              const sidebar=document.querySelector('.sidebar');
              const brand=document.querySelector('.brand-logo');
              const svg=document.querySelector('.kcia-logo');
              const leftMost=svg.querySelector('path').getBoundingClientRect().left;
              const sideRect=sidebar.getBoundingClientRect();
              const style=getComputedStyle(sidebar);
              const sidebarContentLeft=sideRect.left+parseFloat(style.paddingLeft);
              const svgRect=svg.getBoundingClientRect();
              return {leftMost,sidebarContentLeft,brandLeft:brand.getBoundingClientRect().left,
                      svgLeft:svgRect.left,svgWidth:svgRect.width,svgHeight:svgRect.height,
                      ratio:svgRect.width/svgRect.height};
            }''')
            assert abs(measurement['leftMost']-(measurement['sidebarContentLeft']+10)) < 1.5,(width,measurement)
            assert abs(measurement['leftMost']-measurement['svgLeft']) < 1.5,(width,measurement)
            assert abs(measurement['ratio'] - (1381/492)) < 0.05,(width,measurement)
            if width==1380:
                page.screenshot(path='/mnt/data/kcia_v118_logo_left_aligned.png', full_page=True)
            page.close()
        browser.close()
