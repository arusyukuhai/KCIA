"""v138 WAN scaling, independent objective selection, private peer datasets."""
from pathlib import Path
import json
import subprocess
from types import SimpleNamespace

import pytest
from patchlib.wan_cadence_v138 import recommend_steps
from patchlib.roomid import create_room_id
from patchlib.webview_controller import WebviewController
from patchlib.room_manifest import manifest_digest, load_manifest
from patchlib.room_training_v119 import for_local_peer, canonical_training, shared_policy
from patchlib.mix_gui_v66 import normalize, objective_policy, atomic_json
from patchlib.local_training_v110 import training_batches


SRC={'repo':'org/text-dataset','revision':'a'*40,'config':None,
     'split':'train','columns':{'text':'text'},'weight':1.0}
SRC2={'repo':'org/another-dataset','revision':'b'*40,'config':None,
      'split':'train','columns':{'text':'text'},'weight':1.0}


def new_room(c):
    room=create_room_id()
    c.save({'room':room,'model':'sshleifer/tiny-gpt2','revision':'main',
            'task':'causal-lm','initialization':'pretrained','sources':[SRC]})
    return room


def test_cadence_model_monotonic_and_wire_rank_sensitive():
    assert recommend_steps(parameters=0) == 32
    sizes=[100_000_000,500_000_000,1_500_000_000,7_000_000_000,70_000_000_000,700_000_000_000]
    vals=[recommend_steps(parameters=x) for x in sizes]
    assert vals == sorted(vals)
    assert vals[0] == 32
    assert vals[-1] == 512
    tiny=recommend_steps(parameters=135_000_000,observed_patch_bytes=11_000_000)
    large=recommend_steps(parameters=135_000_000,observed_patch_bytes=158_000_000)
    assert tiny == 32 and large>=128
    assert recommend_steps(parameters=5_000_000_000,baseline=64)>=64
    assert recommend_steps(parameters=100_000_000_000,maximum=128)==128
    for kw in ({'parameters':-1},{'parameters':100,'observed_patch_bytes':-20},
               {'parameters':100,'maximum':0}):
        with pytest.raises(ValueError):recommend_steps(**kw)


def test_private_peer_pt_sources_survive_restart_and_never_change_manifest(tmp_path,monkeypatch):
    c=WebviewController(tmp_path)
    room=new_room(c)
    before=load_manifest(c._room_path(room));digest=manifest_digest(before)
    commands=[]
    monkeypatch.setattr(c.train,'start',lambda command,logfile:commands.append(command) or 531)
    c.start({'room':room,'ws':'wss://signal.kcia.tech/ws','kind':'loha',
             'rank':32,'auto_model_cadence':True,'local_steps':32,
             'peer_pt_sources':[SRC2]})
    cmd=commands[-1]
    assert '--auto-model-cadence' in cmd
    assert cmd[cmd.index('--max-sync-steps')+1]=='512'
    private=Path(cmd[cmd.index('--peer-pt-sources-file')+1])
    assert private.parent==c._room_path(room).parent
    assert json.loads(private.read_text())==[SRC2]
    assert private.stat().st_mode & 0o077 == 0
    assert manifest_digest(load_manifest(c._room_path(room)))==digest
    assert 'org/another-dataset' not in c._room_path(room).read_text()
    c.start({'room':room,'ws':'wss://signal.kcia.tech/ws','peer_pt_sources':[],
             'local_steps':32})
    assert '--peer-pt-sources-file' not in commands[-1]
    assert '--auto-model-cadence' not in commands[-1] # manual mode remains intact
    # Closing the first GUI releases its exclusive peer journal lock.  A
    # second *concurrent* controller intentionally uses a separate identity.
    c.profiles[room].close()
    # A real GUI restart releases its exclusive peer-profile lock first.
    # Two *simultaneous* GUIs deliberately get different peer identities and
    # must not import one another's private dataset choices.
    c.profiles[room].close()
    del c.profiles[room]
    other=WebviewController(tmp_path)
    assert other.peer_pt_settings({'room':room})['sources']==[SRC2]


def test_private_pt_source_task_validation(tmp_path,monkeypatch):
    c=WebviewController(tmp_path)
    room=new_room(c)
    wrong={**SRC2,'columns':{'input':'q','target':'a'}}
    with pytest.raises(ValueError,match='column mapping'):
        c.start({'room':room,'ws':'wss://signal.kcia.tech/ws','peer_pt_sources':[wrong]})
    with pytest.raises(ValueError,match='source|dataset|Hugging'):
        c.start({'room':room,'ws':'wss://signal.kcia.tech/ws','peer_pt_sources':[{'repo':'bad'}]})


def test_room_pt_batch_iter_receives_peer_sources_without_changing_signed_genesis(tmp_path, monkeypatch):
    c=WebviewController(tmp_path);room=new_room(c)
    manifest=c._room_path(room)
    local=tmp_path/'sources.json';local.write_text(json.dumps([SRC2]))
    seen=[]
    def fake_batches(args, descriptor):
        seen.append(descriptor)
        yield {'sentinel':True}
    monkeypatch.setattr('patchlib.hf_datasets.batches_from_room',fake_batches)
    args=SimpleNamespace(room_manifest=manifest,batches_dir=None,text_file=None,jsonl=None,
                         pt_weight=1.,sft_jsonl='',ppo_jsonl='',sft_hf_repo='',ppo_hf_repo='',
                         peer_pt_sources_file=local,peer_id='peer123')
    assert next(training_batches(args))=={'sentinel':True}
    assert seen[0]['datasets']['sources']==[SRC2]
    assert load_manifest(manifest)['datasets']['sources']==[SRC]


def test_same_pinned_sft_policy_allows_per_peer_hf_corpus():
    from patchlib.room_manifest import make_hf_manifest
    m=make_hf_manifest('sshleifer/tiny-gpt2','main',[SRC])
    m['training']=canonical_training({'enabled':True,'sft_hf_repo':'org/sft-a',
             'sft_hf_revision':'1'*40,'ppo_weight':0.0,'pt_weight':0.0})
    signed=shared_policy(m)
    data=for_local_peer(m,{'enabled':True,'sft_hf_repo':'org/sft-b',
              'sft_hf_revision':'2'*40})
    normalized=normalize(data,task='causal-lm',allow_pt_only=True)
    assert normalized['pt_weight']==0
    assert normalized['sft_hf_repo']=='org/sft-b'
    # Room-wide scoring and network protocol never include the private repo.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        f=Path(d)/'policy.json';atomic_json(f,signed)
        assert objective_policy(normalized,pinned=f)==signed
    assert 'org/sft-b' not in json.dumps(m)


def test_objective_switches_exist_and_validate_payload_via_browser():
    from patchlib.webview_gui import compose_html
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                                   args=['--no-sandbox'])
        try:
            page=browser.new_page()
            page.set_content(compose_html())
            output=page.evaluate('''()=>{
              const ids=['mixPtEnabled','mixSftEnabled','mixPpoEnabled','peerPtUseDraft','peerPtMode'];
              const available=ids.every(id=>!!document.getElementById(id));
              $('mixSftEnabled').value='on';$('mixSftEnabled').dispatchEvent(new Event('change'));
              $('mixSftHfRepo').value='org/my-sft';$('mixSftHfRevision').value='a'.repeat(40);
              $('mixSftEnabled').value='on';$('mixPpoEnabled').value='off';
              const sft=mixedPayload();
              $('mixPtEnabled').value='off';
              const noPT=mixedPayload();
              $('mixPpoEnabled').value='on';$('mixPpoEnabled').dispatchEvent(new Event('change'));
              $('mixPpoHfRepo').value='org/my-ppo';$('mixPpoHfRevision').value='b'.repeat(40);
              const ppo=mixedPayload();
              return {available,sftWeight:sft.sft_weight,ppoWeight:sft.ppo_weight,
                ptWeight:noPT.pt_weight,sftOn:!!ppo.sft_hf_repo,ppoOn:!!ppo.ppo_hf_repo};
            }''')
            assert output=={'available':True,'sftWeight':1,'ppoWeight':0,
                            'ptWeight':0,'sftOn':True,'ppoOn':True}
        finally:
            browser.close()
