"""KCIA pilot release correctness without external services or Docker."""
from pathlib import Path
import json
import pytest
from scripts.wan_preflight import signal_host, local_report

ROOT = Path(__file__).resolve().parents[1]


def test_vps_image_and_compose_use_same_persistent_data_dir_and_user():
    df=(ROOT/'deploy/vps/Dockerfile').read_text()
    compose=(ROOT/'deploy/vps/compose.yml').read_text()
    assert 'mkdir -p /var/lib/kcia/rooms' in df
    assert 'chown -R 10001:10001 /var/lib/kcia' in df
    assert 'ROOM_DATA_DIR=/var/lib/kcia/rooms' in df
    assert 'ROOM_DATA_DIR: /var/lib/kcia/rooms' in compose
    assert 'rooms:/var/lib/kcia/rooms' in compose
    assert '/var/lib/kagami' not in df
    assert '/build/kcia-signal' in df
    assert 'no-new-privileges:true' in compose


def test_public_wan_service_not_listening_on_public_host_port():
    compose=(ROOT/'deploy/vps/compose.yml').read_text()
    assert 'TUNNEL_TOKEN:' in compose
    assert 'TURN_SECRET:' in compose
    assert 'ports:' not in compose.split('cloudflared:')[0]
    assert 'APP_TOKEN' not in compose
    assert 'ROOM_PUBLISH_TOKEN' not in compose


def test_wss_preflight_does_not_accept_credentials_or_insecure_urls():
    assert signal_host('wss://signal.example.net/ws')=='signal.example.net'
    for bad in ('ws://signal.example.net/ws','https://signal.example.net/ws',
                'wss://signal.example.net/', 'wss://localhost/ws',
                'wss://user:pass@signal.example.net/ws', 'wss://signal.example.net/ws?token=secret'):
        with pytest.raises(ValueError): signal_host(bad)


def test_local_report_has_no_secrets_and_detects_source_files():
    r=local_report('vps')
    assert 'turn_secret' not in str(r).lower()
    assert all(r['project_files'].values())
    assert r['role']=='vps'


def test_colab_notebook_sends_real_max_rank_and_distinct_colab_state():
    nb=json.loads((ROOT/'notebooks/KCIA_Cloudflare_Mac_Colab.ipynb').read_text())
    body='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
    # v130 delegates the signed room-mode trainer to the same tested launcher
    # on both devices, rather than embedding an outdated argv in the notebook.
    assert 'start_wan_pt.py' in body
    assert 'wan_preflight.py' in body
    assert "private/colab.pem" in body
    launcher=(ROOT/'scripts/start_wan_pt.py').read_text()
    assert "'--rank','32','--local-loha-rank','auto'" in launcher
    assert "'--reinit-every','20','--local-steps','4'" in launcher
    assert "'--signing-key'" not in launcher


def test_pilot_readme_separates_discovery_from_real_training_and_respects_20_4():
    s=(ROOT/'README_WAN_PILOT_JA.md').read_text()
    assert '--reinit-every 20 --local-steps 4' in s
    assert 'check_wan_p2p.py connect' in s
    assert 'README_WAN_PILOT_JA' not in s
    assert 'Colab' in s and 'Cloudflare' in s and 'Mac' in s
