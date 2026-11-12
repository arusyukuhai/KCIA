"""Offline validation of the KCIA WAN deployment additions."""
import pathlib
import pytest
from deploy.vps.setup_env import configuration
from patchlib.p2p import _relay_only_sdp
from scripts.check_signaling_vps import _health_url


def test_secret_config_and_public_ip():
    content=configuration('signal.example.org','turn.example.org','8.8.8.8')
    assert 'TURN_HOST=turn.example.org' in content
    entries=dict(line.split('=',1) for line in content.splitlines())
    assert 'APP_TOKEN' not in entries and 'ROOM_PUBLISH_TOKEN' not in entries
    assert len(entries['TURN_SECRET'])==64
    with pytest.raises(ValueError): configuration('localhost','turn.example.org','8.8.8.8')
    with pytest.raises(ValueError): configuration('signal.example.org','turn.example.org','127.0.0.1')
    with pytest.raises(ValueError): configuration('signal.example.org','turn.example.org','203.0.113.10')


def test_relay_only_fails_closed():
    sdp='v=0\r\na=candidate:1 1 udp 3 192.168.1.2 4321 typ host\r\na=candidate:2 1 udp 3 1.2.3.4 5321 typ relay\r\na=end-of-candidates\r\n'
    filtered=_relay_only_sdp(sdp)
    assert 'typ relay' in filtered and 'typ host' not in filtered
    assert 'a=end-of-candidates' in filtered
    with pytest.raises(ConnectionError,match='TURN-only'):
        _relay_only_sdp('v=0\r\na=candidate:1 1 udp 3 127.0.0.1 2345 typ host\r\n')


def test_health_url_requires_correct_ws_path():
    assert _health_url('wss://kcia.example.org/ws')=='https://kcia.example.org/healthz'
    with pytest.raises(ValueError): _health_url('https://kcia.example.org/ws')


def test_compose_and_dockerfile_have_no_public_signaling_port():
    root=pathlib.Path(__file__).resolve().parents[1]
    compose=(root/'deploy/vps/compose.yml').read_text()
    assert 'APP_TOKEN' not in compose and 'ROOM_PUBLISH_TOKEN' not in compose
    assert 'TURN_SECRET: ${TURN_SECRET:?' in compose
    assert '"8080:8080"' not in compose
    assert 'network_mode: host' in compose
    assert '--no-cli' not in compose and '--no-dtls' not in compose
    assert 'dockerfile: deploy/vps/Dockerfile' in compose
