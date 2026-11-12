"""GUI room-manifest saves and safe training launches.

All compatibility, journal pinning, model-device preflight, and atomic policy
checks run in the same order as before; WebviewController is still the owner of
rooms, process handles and per-window state.
"""
from __future__ import annotations
import datetime
import json
import os
import secrets
from pathlib import Path
from .room_manifest import (
    make_hf_manifest, manifest_digest, validate_manifest,
    validate_hf_sources, load_manifest,
)
from .roomid import validate_room
from .webview_validation_v110 import check_signaling_url, _exclusive_manifest_lock


def save_room(self, payload: dict):
    room = validate_room(payload['room'])
    source = validate_hf_sources(payload['sources'])
    model, revision, task = payload['model'],payload['revision'],payload['task']
    from .live_initialization import (room_initialization,
                                      RANDOM_MANIFEST_DESCRIPTION)
    initialization=payload.get('initialization',
                               'random' if model=='roneneldan/TinyStories-1M' else 'pretrained')
    if initialization not in ('random','pretrained','config'):
        raise ValueError('invalid genesis initialization')
    creation_mode=payload.get('creation_mode','weights')
    if creation_mode not in ('weights','architecture','code'):
        raise ValueError('unknown model creation mode')
    if creation_mode in ('architecture','code'):
        initialization='config'
    if initialization=='random' and model!='roneneldan/TinyStories-1M':
        raise ValueError('Random TinyStories initialization cannot be used for Llama/DeepSeek/Qwen. Select pretrained weights and create a new room.')
    if initialization=='random' and creation_mode!='weights':
        raise ValueError('legacy random TinyStories cannot be combined with a new architecture mode')
    from .model_design import DESCRIPTION, extract_config
    description=(RANDOM_MANIFEST_DESCRIPTION if initialization=='random' else
                 DESCRIPTION[creation_mode])

    code_path = payload.get('code_path') or ''
    code_url = payload.get('code_url') or ''
    if code_path:
        file = Path(code_path).resolve()
        # Only code already vetted by the explicit upload endpoint may be registered.
        allowed = (self.workdir / '.fedpara-gui' / 'codes').resolve()
        if file.parent != allowed or not file.is_file():
            raise PermissionError('model code must first be uploaded through this dashboard')
    else:
        file = None
    manifest = make_hf_manifest(model,revision,source,architecture=task,
                                seed=payload.get('seed',42),description=description,
                                code_path=file,code_url=code_url)
    if creation_mode=='architecture':
        manifest['design']={'mode':'architecture'}
    elif creation_mode=='code':
        if file is None:
            existing=self._room_path(room)
            if existing.is_file():
                previous=load_manifest(existing)
                if (previous['model']['id']==model and previous['model']['revision']==revision
                    and previous['model']['architecture']==task and
                    previous['datasets']==manifest['datasets'] and
                    previous['model']['description']==description and
                    previous.get('design',{}).get('mode')=='code'):
                    return {'path':str(existing),'sha256':manifest_digest(previous),'manifest':previous}
            # A joining peer must NOT have to download and execute the
            # author's Python. The signaling manifest pins the equivalent
            # DATA-ONLY HF Config. Accept it only after verifying every
            # user-visible immutable creation field and the code origin.
            with self.lock:
                imported=self.imported.get(room)
            if (imported and imported.get('design',{}).get('mode')=='code'
                and imported['model']['id']==model
                and imported['model']['revision']==revision
                and imported['model']['architecture']==task
                and imported['model']['description']==description
                and imported['datasets']==manifest['datasets']
                and (not code_url or code_url==imported['code']['source'])):
                manifest=validate_manifest(imported)
            else:
                raise ValueError('Python architecture requires an uploaded .py or an authenticated, pinned model descriptor')
        else:
            if file.suffix!='.py':raise ValueError('Python architecture source must be .py')
            manifest['design']=extract_config(file.read_text(encoding='utf8'))
            tok_id=(payload.get('tokenizer_model') or model).strip()
            tok_revision=(payload.get('tokenizer_revision') or revision).strip()
            manifest['design']['tokenizer']={'id':tok_id,'revision':tok_revision}
        # A portable, strictly data-only config is network-shared; never execute source on peers.
    # Pinned room-global adapter layout: selecting a different rank on
    # another device would silently fork the genesis and Gossip stream.
    # Imported descriptors remain immutable, including legacy unpinned ones.
    # A remote room is an immutable shared object. When a peer joins,
    # preserve the ENTIRE server-confirmed descriptor, including adapter
    # layout and custom tokenizer fingerprints. Reconstructing only its
    # basic model/dataset fields silently drops the network identity.
    with self.lock:
        imported_descriptor = self.imported.get(room)
    if imported_descriptor is not None:
        matching = (imported_descriptor['model'] == manifest['model'] and
                    imported_descriptor['datasets'] == manifest['datasets'] and
                    imported_descriptor.get('design', {}).get('mode', 'weights') == creation_mode)
        if not matching or file is not None:
            raise ValueError('Remote model configuration is immutable; use a fresh model ID for edits')
        manifest = validate_manifest(dict(imported_descriptor))
    if 'adapter' not in manifest and imported_descriptor is None:
        from .auto_policy import estimate_adapter_rank
        config = dict(manifest.get('design',{}).get('kwargs') or {})
        if not config:
            try:
                from .hf_config_v83 import load_hf_config
                config = load_hf_config(model,revision=revision,
                          local_files_only=True,trust_remote_code=False).to_dict()
            except (ImportError,OSError,ValueError,RuntimeError,TypeError,KeyError):
                pass
        resources=self.device_capabilities()
        kind=payload.get('adapter_kind','loha')
        recommendation=estimate_adapter_rank(config=config,
            available_ram_bytes=resources.get('ram_bytes') or 0,
            available_vram_bytes=resources.get('vram_bytes') or 0,
            kind=kind)
        adapter_scope=payload.get('adapter_scope','auto')
        if adapter_scope not in ('auto','universal'):
            raise ValueError('invalid universal adapter coverage setting')
        manifest['adapter']={'kind':kind, 'rank':recommendation['rank'],
                             'wide_lora_rank':4}
        if adapter_scope=='universal':
            manifest['adapter']['target_scope']='universal'
    validate_manifest(manifest)
    room_initialization(manifest)  # fail on non-pinned random models before writing
    # Preserve an imported code fingerprint if the user has not uploaded new code.
    with self.lock:
        old = self.imported.get(room)
        if old and not file and old['code']['sha256'] and old['code']['source']==code_url:
            manifest['code'] = dict(old['code'])
            validate_manifest(manifest)
        path = self._room_path(room)
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.chmod(path.parent,0o700)
        # GUI creation should not silently fork a second room with exactly
        # the same immutable model definition. Separate IDs mean separate
        # training histories, so we never delete or merge them implicitly.
        # A catalog-wide lock makes the duplicate test + commit atomic even
        # across two GUI processes creating different IDs concurrently.
        # Legacy/programmatic APIs retain their old semantics unless they
        # explicitly request duplicate prevention.
        from contextlib import nullcontext
        protect = payload.get('prevent_duplicate') is True
        catalog_dir = path.parent.parent
        catalog_lock = (_exclusive_manifest_lock(catalog_dir / '.catalog.lock')
                        if protect else nullcontext())
        with catalog_lock:
            if protect and not path.is_file() and payload.get('allow_duplicate') is not True:
                candidate_digest = manifest_digest(manifest)
                for existing in catalog_dir.glob('*/manifest.json'):
                    if existing.parent.name == room or existing.is_symlink():
                        continue
                    try:
                        prior=dict(load_manifest(existing))
                        prior.pop('generation',None)
                        previous_digest = manifest_digest(prior)
                    except (OSError, ValueError, KeyError):
                        # Damaged/unreadable entries must not block a new model.
                        continue
                    if previous_digest == candidate_digest:
                        raise ValueError(
                            'Identical model already saved under room '
                            + existing.parent.name
                            + '. Select it in MODELS, or enable '
                            '"Create independent experiment" in NEW MODEL.')
            # Two WebViews share this pathname. An unchecked Save/Start could
            # replace a running peer's pinned genesis descriptor and induce
            # endless JOIN WAIT / genesis mismatches in another window.
            with _exclusive_manifest_lock(path.parent / '.manifest.lock'):
                if path.is_symlink():
                    raise PermissionError('symlink cannot be used as a room manifest')
                if path.is_file():
                    current=load_manifest(path)
                    # Preserve the previously pinned owner for a second local GUI.
                    comparable=dict(current)
                    if imported_descriptor is None and 'generation' not in manifest:
                        comparable.pop('generation',None)
                    if manifest_digest(comparable) != manifest_digest(manifest):
                        raise ValueError('このルームは既に別のモデル/データセット設定で固定されています。'
                                         '参加する場合は「設定を取得」で元の設定を読み込み、'
                                         '新しい構成に変える場合は「新規ルーム」を作成してください。')
                    return {'path':str(path),'sha256':manifest_digest(current),
                            'manifest':current}
                # V88: new GUI rooms pin a room-owner PUBLIC key immutably.
                # The owner's private key stays local. Imported/legacy room
                # manifests are NEVER rewritten to invent an authority.
                if imported_descriptor is None and 'generation' not in manifest:
                    from .auto_committee_v88 import owner_for_local_room
                    owner_file = path.parent / 'generation-owner.pem'
                    if owner_file.exists() or owner_file.is_symlink():
                        raise PermissionError('room owner private key exists without a room manifest; manual recovery required')
                    manifest['generation']={'owner':owner_for_local_room(owner_file)}
                    validate_manifest(manifest)
                # Crash-safe commit, with exclusive cross-process descriptor lock.
                tmp=path.with_name('.manifest-'+secrets.token_hex(8))
                fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                try:
                    with os.fdopen(fd,'w',encoding='utf8') as out:
                        json.dump(manifest,out,ensure_ascii=False,indent=2,allow_nan=False)
                        out.write('\n')
                        out.flush();os.fsync(out.fileno())
                    os.replace(tmp,path)
                finally:
                    tmp.unlink(missing_ok=True)
            return {'path':str(path),'sha256':manifest_digest(manifest),'manifest':manifest}



def start_training(self, payload: dict, *, profile_allocator):
    room=validate_room(payload['room'])
    with self.lock:
        path=self._room_path(room)
        if not path.is_file():
            raise FileNotFoundError('Save room settings before starting training')
        manifest=load_manifest(path)
        from .live_initialization import room_initialization
        initialization=room_initialization(manifest)
        streaming_base=payload.get('streaming_base',False) is True
        if streaming_base:
            if initialization != 'pretrained' or payload.get('base'):
                raise ValueError('ストリーミング学習は事前学習済みモデル・初期状態空欄のみです。新規ルームで始めてください。')
            base=None
            automatic=False
        raw_base = payload.get('base')
        generation_info_file=path.parent/'generation-base.json'
        sharded_base_manifest=None
        sharded_base_store=None
        if generation_info_file.is_file():
            from .generation_rollover import verify_generation
            generation_dir=Path(json.loads(generation_info_file.read_text(encoding='utf-8'))['generation'])
            generation_info=verify_generation(generation_dir)
            if generation_info['new_room_id']!=room or                        manifest_digest(load_manifest(generation_dir/'room-manifest.json'))!=manifest_digest(manifest):
                raise ValueError('pinned generation checkpoint/room configuration changed')
            sharded_base_manifest=(generation_dir/'base.fmanifest').resolve()
            sharded_base_store=(generation_dir/'store').resolve()
            streaming_base=False
            base=None
            automatic=False
        if not sharded_base_manifest and not streaming_base and (raw_base is None or (isinstance(raw_base,str) and not raw_base.strip())):
            # Automatic per-room immutable genesis cache; no path entry needed.
            # Different path from legacy/pretrained bootstrap prevents an
            # old cache silently defeating the requested random initialization.
            genesis_name=('genesis.random.fstate' if initialization=='random' else
                          'genesis.config.fstate' if initialization=='config' else 'genesis.fstate')
            base=(self.workdir/'.fedpara-gui'/'rooms'/room/genesis_name).resolve()
            automatic=True
        elif not sharded_base_manifest and not streaming_base and isinstance(raw_base,str):
            base=Path(raw_base.strip()).expanduser().resolve()
            automatic=False
        elif not sharded_base_manifest and not streaming_base:
            raise ValueError('invalid initial checkpoint path')
        if base is not None and base.is_dir():
            raise IsADirectoryError('初期状態にはフォルダでなく .fstate ファイルが必要です: '+str(base))
        if room not in self.profiles:
            self.profiles[room] = profile_allocator(path.parent)
        profile = self.profiles[room]
        from .restart_policy import read_policy, write_policy
        pinned = read_policy(profile.journal.resolve(), manifest_digest(manifest))
        optimizer = payload.get('optimizer','adamw')
        if pinned is not None:
            if isinstance(raw_base,str) and raw_base.strip():
                requested = Path(raw_base.strip()).expanduser().resolve()
                if not requested.is_file():
                    raise FileNotFoundError('requested .fstate checkpoint not found: ' + str(requested))
                if str(requested) != pinned['base']:
                    raise ValueError('A different genesis cannot replace the saved training journal. Re-select the existing model.')
            # The old launch mode changes the genesis digest. The UI may
            # have reset to default after a restart; restore the identity
            # from this peer's durable launch pin instead of forking it.
            initialization = pinned['initialization']
            streaming_base = pinned['streaming_base']
            automatic = pinned['auto_state']
            base = Path(pinned['base']) if pinned['base'] is not None else None
            optimizer = pinned['optimizer']
            reinit_every = pinned['reinit_every']
            if pinned.get('sharded_base_manifest') != (str(sharded_base_manifest) if sharded_base_manifest else None) or                   pinned.get('sharded_base_store') != (str(sharded_base_store) if sharded_base_store else None):
                raise ValueError('saved generation checkpoint differs from the immutable resume policy')
            if base is not None and not base.is_file() and not automatic:
                raise FileNotFoundError('Previously pinned genesis .fstate is missing: '+str(base))
        else:
            reinit_every=payload.get('reinit_every',20)
            # v38 did not record its launch policy. For pretrained rooms
            # with a journal but NO genesis cache, out-of-core loading was
            # very likely chosen. Never delete the existing journal to
            # recover a mismatched genesis.
            if (profile.journal.exists() and not streaming_base and
                initialization == 'pretrained' and automatic and
                base is not None and not base.is_file()):
                streaming_base=True
                automatic=False
                base=None
        from .mix_gui_v66 import normalize, objective_policy, atomic_json
        mixed_policy_file=path.parent/'mixed-objectives-v66.json'
        existing_policy=None
        if mixed_policy_file.exists():
            try:existing_policy=json.loads(mixed_policy_file.read_text(encoding='utf8'))
            except (OSError,ValueError) as exc:
                raise ValueError('Pinned shared mixed-learning policy is unreadable') from exc
        mixed=normalize(payload.get('mixed_training',{}),
                        task=manifest['model']['architecture'],
                        allow_pt_only=existing_policy is not None)
        if existing_policy is not None and not mixed['enabled']:
            raise ValueError('Mixed training objective policy is immutable for this room. Choose mixed mode or create a NEW room.')
        try:
            mixed_spec=objective_policy(mixed, pinned=mixed_policy_file if existing_policy is not None else None)
        except ValueError as exc:
            raise ValueError('Mixed training objective policy is immutable for this room: '+str(exc)) from exc
        # Local acceleration is NOT a signed objective; validate explicit
        # GPU availability before pinning any new room policy or journal.
        from .mps_launch_v108 import validate_devices
        validate_devices(payload.get('compute_device','auto'), mixed)
        # A known-incompatible HF Critic must fail before the GUI pins a
        # new immutable PPO policy, starts a worker or downloads model shards.
        # This is metadata-only and leaves existing signed state untouched.
        from .mix_gui_v66 import preflight_gui_critic
        preflight_gui_critic(mixed)
        if mixed_spec is not None and existing_policy is None:
            # A room already trained with PT-only has another signed genesis.
            if profile.journal.exists() and profile.journal.stat().st_size:
                raise ValueError('A PT-only journal already exists. Mixed PT/SFT/PPO requires a NEW room.')
            # Peers joining later must receive the policy over an explicit,
            # pinned channel; never synthesize it from different local files.
            atomic_json(mixed_policy_file,mixed_spec)
        critic_dir=path.parent/'critics'/profile.name
        pause_file = path.parent / ('pause-'+profile.name+'.flag')
        journal = profile.journal.resolve()
        # Never overwrite/remove the immutable genesis or the append-only journal.
        # A fresh manifest per run keeps older successful checkpoints available.
        checkpoints = path.parent / 'checkpoints' / profile.name
        model_storage = path.parent / 'models' / profile.name
        checkpoints.mkdir(parents=True,exist_ok=True,mode=0o700)
        model_storage.mkdir(parents=True,exist_ok=True,mode=0o700)
        snapshot_request_file=path.parent / ('eval-snapshot-'+profile.name+'.request')
        checkpoint = checkpoints / ('effective-' + datetime.datetime.now(
            datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-'
            + secrets.token_hex(4) + '.shards.json')
        if type(reinit_every) is not int or not 0 <= reinit_every <= 100000:
            raise ValueError('reinit_every must be 0..100000 local commits')
        # Private run-local file: never mix data between the two GUI windows.
        telemetry_dir=self.workdir/'.fedpara-gui'/'telemetry'
        telemetry_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.chmod(telemetry_dir,0o700)
        # Stable, profile-owned history: do not reset graph to a few
        # hundred steps whenever the learning worker is restarted.
        telemetry_path=telemetry_dir/('history-'+room+'-'+profile.name+'.jsonl')
        cmd=self.train.build_command(room=room,model=manifest['model']['id'],
            revision=manifest['model']['revision'],base=base,manifest=path,
            ws=check_signaling_url(payload['ws']),task=manifest['model']['architecture'],
            kind=manifest.get('adapter',{}).get('kind',payload.get('kind','loha')),
            rank=manifest.get('adapter',{}).get('rank',int(payload.get('rank',16))),journal=journal,
            optimizer=optimizer,reinit_every=reinit_every,
            initialization=initialization,initial_seed=manifest['datasets']['seed'],
            lr=payload.get('lr',1e-4),local_steps=payload.get('local_steps',4),
            max_length=payload.get('max_length',64),
            wide_lora_rank=manifest.get('adapter',{}).get('wide_lora_rank',payload.get('wide_lora_rank',4)),
            peers=None,rounds=0,
            auto_state=automatic,room_public=payload.get('room_public',False) is True,
            telemetry_file=telemetry_path,journal_sync=payload.get('journal_sync','full'),
            push_fanout=payload.get('push_fanout',2),
            shard_state_manifest=checkpoint,shard_store=model_storage,
            max_state_gb=2048,max_journal_gb=payload.get('max_journal_gb',4),
            streaming_base=streaming_base,
            sharded_base_manifest=sharded_base_manifest,
            sharded_base_store=sharded_base_store,
            disk_cache_fraction=payload.get('disk_cache_fraction',.6),pause_file=pause_file,
            snapshot_request_file=snapshot_request_file,
            mixed_options=mixed,mixed_policy=mixed_policy_file if mixed_spec else None,
            critic_state_dir=critic_dir,
            device=payload.get('compute_device','auto'))
        # A bad new setting must never poison a healthy old journal.
        # Record the immutable launch policy before dispatching, so a GUI
        # crash between process creation and first metric cannot forget it.
        write_policy(journal,manifest_digest=manifest_digest(manifest),
                     initialization=initialization,streaming_base=streaming_base,
                     base=base,auto_state=automatic,optimizer=optimizer,
                     reinit_every=reinit_every,
                     sharded_base_manifest=sharded_base_manifest,
                     sharded_base_store=sharded_base_store)
        # Clear stale relay-only sentinel only for an actual new worker.
        # Never touch it if another worker holds this journal.
        if self.train.status():
            raise RuntimeError('This model is already online; resume instead')
        pause_file.unlink(missing_ok=True)
        snapshot_request_file.unlink(missing_ok=True)
        # Never treat an unfinished export from an earlier process as
        # still running: a new worker can service another request.
        snapshot_request_file.with_suffix('.status.json').unlink(missing_ok=True)
        pid=self.train.start(cmd,str(journal)+'.gui.log')
        if mixed['enabled']:
            atomic_json(path.parent/('mixed-local-'+profile.name+'.json'),{'room':room,**mixed})
        self._pause_file=pause_file
        self._snapshot_request_file=snapshot_request_file
        self._current_room=room
        self._telemetry_path=telemetry_path
        self._checkpoint_path=checkpoint
        self._model_storage_path=model_storage
        self._remember_room(room)
        return {'pid':pid,'journal':str(journal),'mixed_enabled':mixed['enabled'],
                'critic_dir':str(critic_dir) if (mixed['ppo_jsonl'] or mixed.get('ppo_hf_repo')) else None,'profile':profile.name,
                'checkpoint_file':str(checkpoint),'model_storage_dir':str(model_storage),
                'auto_state':automatic,'initialization':initialization,
                'base':str(base) if base is not None else '(safetensors streaming genesis)'}
