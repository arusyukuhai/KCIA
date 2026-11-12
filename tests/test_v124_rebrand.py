"""KCIA-only brand, URI scheme and SVG navigation smoke contracts."""
from pathlib import Path
from patchlib.webview_gui import compose_html
from patchlib.storage_location import _config_file, choose_data_root

ROOT=Path(__file__).resolve().parents[1]

def test_brand_gui_and_uri():
    html=compose_html()
    assert '<title>KCIA — Distributed Intelligence</title>' in html
    assert 'kcia://join' in html
    assert "url.protocol!=='kcia:'" in html
    assert 'kagami://' not in html.lower()
    assert 'DELETE KCIA' in html
    assert 'https://use.typekit.net/wzz8rse.css' in html
    assert 'myriad-pro-cond' in html
    assert 'myriad-pro-semi-condensed' not in html

def test_six_svg_navigation_symbols():
    html=compose_html()
    assert html.count('class="navicon"') == 6
    assert html.count('class="navicon" aria-hidden="true"><svg') == 6
    for tab in ('models', 'training', 'datasets','eval','network','new'):
        assert 'data-tab="'+tab+'"' in html

def test_storage_and_new_entrypoint(tmp_path):
    assert _config_file(tmp_path).parts[-3:]==('.config','kcia','data-root.txt')
    assert (ROOT/'kcia.py').is_file()
    assert not (ROOT/'kagami.py').exists()
    home=tmp_path/'home';home.mkdir()
    cwd=tmp_path/'cwd';cwd.mkdir()
    result=choose_data_root(cwd=cwd,home=home,environment={})
    assert 'kcia' in str(result).lower()
