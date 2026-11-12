"""Offline deployment safeguards; not a substitute for a VPS/Docker/WAN test."""
from pathlib import Path
import ast
import importlib.util
import ipaddress
import json
import re
import pytest

ROOT=Path(__file__).resolve().parents[1]


def _setup():
    spec=importlib.util.spec_from_file_location('kcia_setup_env',ROOT/'deploy/vps/setup_env.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_separate_domains_and_fresh_secrets():
    config=_setup().configuration
    x=config('signal.example.com','turn.example.com','8.8.8.8')
    y=config('signal.example.com','turn.example.com','8.8.8.8')
    a=dict(line.split('=',1) for line in x.splitlines())
    b=dict(line.split('=',1) for line in y.splitlines())
    assert a['SIGNAL_HOST']=='signal.example.com'
    assert a['TURN_HOST']=='turn.example.com'
    assert a['TURN_REALM']=='turn.example.com'
    assert a['CF_TUNNEL_TOKEN']=='PASTE_CLOUDFLARE_TUNNEL_TOKEN_HERE'
    for key in ('TURN_SECRET',):
        assert re.fullmatch('[0-9a-f]{64}',a[key]);assert a[key]!=b[key]


def test_invalid_network_configs_fail_closed():
    config=_setup().configuration
    with pytest.raises(ValueError,match='differ'):
        config('same.example.com','same.example.com','8.8.8.8')
    for ip in ('127.0.0.1','192.0.2.1','10.0.0.1','not-ip'):
        with pytest.raises(ValueError):config('signal.example.com','turn.example.com',ip)
    with pytest.raises(ValueError): config('bad','turn.example.com','8.8.8.8')


def test_compose_only_tunnel_and_turn_public():
    yaml_path=ROOT/'deploy/vps/compose.yml'
    text=yaml_path.read_text()
    assert 'caddy:' not in text.lower()
    assert 'cloudflared:' in text
    assert 'TUNNEL_TOKEN:' in text
    assert 'network_mode: host' in text
    assert 'ports:' not in text  # Prevent accidental public 8080 origin binding.
    assert '49160' in text and '49200' in text
    try:
        import yaml
    except ImportError:
        return
    doc=yaml.safe_load(text)
    assert set(doc['services']) == {'signal','cloudflared','coturn'}
    assert 'ports' not in doc['services']['signal']
    assert doc['services']['cloudflared']['environment']['TUNNEL_TOKEN'].startswith('${CF_TUNNEL_TOKEN:')
    assert doc['services']['coturn']['network_mode']=='host'


def test_turn_only_filters_both_directions():
    from patchlib.p2p import _relay_only_sdp
    sdp='v=0\r\na=candidate:1 1 udp 1 10.0.0.1 1234 typ host\r\na=candidate:2 1 udp 1 8.8.8.8 9876 typ relay\r\na=end-of-candidates\r\n'
    assert 'typ host' not in _relay_only_sdp(sdp)
    assert 'typ relay' in _relay_only_sdp(sdp)
    with pytest.raises(ConnectionError,match='no relay'):
        _relay_only_sdp('v=0\r\na=candidate:1 1 udp 1 10.0.0.1 1234 typ host\r\n')
    source=(ROOT/'patchlib/p2p.py').read_text()
    assert "'type':'offer','sdp':(_relay_only_sdp" in source
    assert "'type':'answer', 'sdp':(_relay_only_sdp" in source


def test_notebook_source_parses_and_uses_no_hardcoded_tokens():
    nb=json.loads((ROOT/'notebooks/KCIA_Cloudflare_Mac_Colab.ipynb').read_text())
    assert nb['nbformat']==4
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        if source.startswith('%'):continue
        ast.parse(source)
    text=json.dumps(nb)
    assert 'app_token' not in text and 'getpass' not in text
    assert 'APP_TOKEN' not in text and 'ROOM_PUBLISH_TOKEN' not in text
    assert 'await connect(args)' in text
