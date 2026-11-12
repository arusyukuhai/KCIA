"""Adobe Fonts kit wzz8rse exposes Myriad Pro Condensed as myriad-pro-cond."""
from patchlib.webview_gui import compose_html


def test_font_family_consistent_between_css_and_runtime():
    html = compose_html()
    assert "https://use.typekit.net/wzz8rse.css" in html
    assert "const FAMILY = 'myriad-pro-cond';" in html
    assert '"myriad-pro-cond"' in html
    assert 'myriad-pro-condensed' not in html
    assert "String(face.family || '')" in html


def test_font_status_remains_diagnostic():
    html = compose_html()
    assert "FONT CDN!" in html
    assert "FONT KIT?" in html
    assert "FONT ✓" in html
