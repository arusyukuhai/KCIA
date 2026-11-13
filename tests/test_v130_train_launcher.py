"""Offline consistency checks for Mac/Colab training launch."""
import argparse
from pathlib import Path
import importlib.util
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('launch',ROOT/'scripts/start_wan_pt.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.mark.parametrize('role,expected_device', [('mac','cpu'),('colab','cuda')])
def test_role_training_args_are_distinct_and_signed(role,expected_device):
    a=argparse.Namespace(room_id='abcdeFGHIJKLMNOPQRSTU123456',revision='1'*40,role=role,
        ws='wss://signal.kcia.tech/ws',device=None,rounds=0)
    argv=module.make_command(a)
    assert '--signing-key' not in argv
    assert argv[argv.index('--device')+1]==expected_device
    assert argv[argv.index('--rank')+1]=='32'
    assert argv[argv.index('--local-loha-rank')+1]=='auto'
    assert argv[argv.index('--local-steps')+1]=='4'
    assert argv[argv.index('--reinit-every')+1]=='20'
    assert argv[argv.index('--room-id')+1]==a.room_id
    assert argv[argv.index('--state-base')+1].endswith('/'+role+'/genesis.fstate')
    assert argv[argv.index('--journal')+1].endswith('/'+role+'/events.sqlite3')
    assert argv[argv.index('--text-file')+1].endswith('wan_'+role+'.txt')
    assert ('--no-auto-tune' in argv)==(role=='mac')

def test_bad_revision_rejected():
    a=argparse.Namespace(room_id='123456789012345678901234',revision='main',role='mac',ws='wss://x/ws',device=None,rounds=0)
    with pytest.raises(ValueError):module.make_command(a)

def test_default_ua_used_on_real_paths():
    for file in ['patchlib/roomid.py','patchlib/p2p.py','patchlib/webview_controller.py']:
        text=(ROOT/file).read_text()
        assert 'KCIA-WAN-Client/1.0' in text
        assert 'user_agent_header=' in text
    assert 'KCIA-WAN-Pilot/1.0' in (ROOT/'scripts/check_signaling_vps.py').read_text()

def test_notebook_valid_and_uses_launcher():
    notebook=json.loads((ROOT/'notebooks/KCIA_Cloudflare_Mac_Colab.ipynb').read_text())
    assert notebook['nbformat']==4
    code='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'start_wan_pt.py' in code
    assert 'train_room' in code
