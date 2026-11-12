"""Business logic behind Fedpara's embedded WebView desktop UI.

No HTTP listener is created. It does not execute custom model code.
Room manifests only contain validated, hash-pinned metadata.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import shutil
import subprocess
import sys
import datetime
from pathlib import Path

from .gui import DesktopController
from .room_manifest import (make_hf_manifest, manifest_digest, validate_manifest,
                            validate_hf_sources, load_manifest)
from .roomid import create_room_id, validate_room, fetch_room_manifest, list_public_rooms, register_room_manifest
from .profiles import allocate_profile
from .webview_validation_v110 import check_signaling_url, _exclusive_manifest_lock


class WebviewController:
    def __init__(self, workdir: Path | None = None):
        if workdir is None:
            from .storage_location import choose_data_root, pin_data_root
            self.workdir = choose_data_root()
            pin_data_root(self.workdir)
        else:
            self.workdir = Path(workdir).resolve()
        self.train = DesktopController()
        self.lock = threading.RLock()
        self.imported: dict[str, dict] = {}
        self.profiles = {}  # one stable, exclusive profile per GUI instance / room
        self._bench_proc = None
        self._bench_log = None
        self._bench_output = None
        self._telemetry_path = None
        self._checkpoint_path = None
        self._model_storage_path = None
        self._pause_file = None
        self._current_room = None
        self._snapshot_request_file = None
        self._eval_proc = None
        self._eval_result = None
        self._eval_log = None


    def _room_path(self, room: str) -> Path:
        validate_room(room)
        return self.workdir / '.fedpara-gui' / 'rooms' / room / 'manifest.json'

    def _last_room_file(self):
        return self.workdir / '.fedpara-gui' / 'last-room.txt'

    def _remember_room(self, room: str):
        validate_room(room)
        location = self._last_room_file()
        location.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        temporary = location.with_name('.last-room-'+secrets.token_hex(8))
        fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL|
                     getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            with os.fdopen(fd,'w',encoding='ascii') as out:
                out.write(room+'\n');out.flush();os.fsync(out.fileno())
            os.replace(temporary,location)
        finally:
            temporary.unlink(missing_ok=True)

    def _last_room(self):
        try:
            value=self._last_room_file().read_text(encoding='ascii').strip()
            if self._room_path(value).is_file():
                return value
        except (OSError,UnicodeError,ValueError):
            pass
        return None

    def settings(self):
        return {'last_room':self._last_room(), 'room':create_room_id(), 'model':'roneneldan/TinyStories-1M',
                'revision':'77f1b168e219585646439073245fe87e56b3023e', 'task':'causal-lm', 'ws':'ws://127.0.0.1:8080/ws',
                'seed':42,'kind':'loha','rank':16,'initialization':'random',
                'reinit_every':20,'lr':0.0001,'local_steps':4,'max_length':64,
                'wide_lora_rank':4,'rounds':0,'push_fanout':2,
                'max_journal_gb':4,'streaming_base':False,'disk_cache_fraction':.6,
                'sources':[{'repo':'roneneldan/TinyStories','revision':'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64','config':None,
                            'split':'train','columns':{'text':'text'},'weight':1.0}]}

    def save(self, payload: dict):
        """Delegated room configuration; see webview_lifecycle_v110."""
        from .webview_lifecycle_v110 import save_room
        return save_room(self, payload)

    def load_room(self, payload: dict):
        room = validate_room(payload['room'])
        ws = check_signaling_url(payload['ws'])
        token = payload.get('token','')
        if not isinstance(token,str) or len(token)>256:
            raise ValueError('room token too large')
        manifest=asyncio.run(fetch_room_manifest(ws,room,token,timeout=8))
        if manifest['version'] != 2:
            raise ValueError('room uses a legacy dataset descriptor (v1)')
        with self.lock:
            self.imported[room]=manifest
        return {'manifest':manifest,'sha256':manifest_digest(manifest), 'warning':
                'Verify the manifest fingerprint with the room creator independently.'}

    def select_model(self, payload):
        """Prefer immutable local descriptors, fetch only on explicit remote selection."""
        room=validate_room(payload.get('room',''))
        path=self._room_path(room)
        if path.is_file():
            m=load_manifest(path)
            # Selecting an idle model restores its historical graph even
            # before launching another worker. Never override an active peer.
            with self.lock:
                if not self.train.status():
                    self._current_room=room
                    self._snapshot_request_file=None
                    self._checkpoint_path=None
                    self._model_storage_path=None
                    logs=self.workdir/'.fedpara-gui'/'telemetry'
                    candidates=list(logs.glob('history-'+room+'-*.jsonl')) if logs.is_dir() else []
                    self._telemetry_path=max(candidates,key=lambda p:p.stat().st_mtime_ns) if candidates else None
            self._remember_room(room)
            return {'manifest':m,'sha256':manifest_digest(m),'local':True}
        result=self.load_room(payload)
        result['local']=False
        return result

    def model_compatibility(self,payload):
        """Conservative training eligibility: no false OOM verdicts from heuristics.

        Try local configuration, then public Hub metadata (small, timeout-bound).
        Never download weights, execute model code, or assume custom kernels work.
        """
        from .room_manifest import HF_REPO, DATA_PART
        name=payload.get('model','');revision=payload.get('revision','main')
        if not isinstance(name,str) or not isinstance(revision,str) or not HF_REPO.fullmatch(name) or not DATA_PART.fullmatch(revision):
            return {'status':'unsupported','reason':'Unsupported HF model identifier','parameters':None}
        cfg=None;parameters=None;required_disk=None
        identity_issue=None
        config_identity_verified=False
        try:
            from .hf_config_v83 import load_hf_config, MissingModelTypeError
            config=load_hf_config(name,revision=revision,
                local_files_only=False,trust_remote_code=False)
            cfg=config.to_dict()
            config_identity_verified=True
        except MissingModelTypeError as exc:
            identity_issue=str(exc)
        except (ImportError,ValueError,OSError,RuntimeError,TypeError,KeyError):
            pass
        if identity_issue:
            return {'status':'unsupported','reason':identity_issue,'parameters':None,
                    'diagnostic_command':f'python -m patchlib.cli hf-diagnose --model {name} --revision {revision}'}
        try:
            from huggingface_hub import HfApi
            info=HfApi().model_info(name,revision=revision,timeout=5,files_metadata=True)
            if not cfg and isinstance(getattr(info,'config',None),dict):
                cfg=info.config
            safe=getattr(info,'safetensors',None)
            if safe:
                parameters=int(safe.get('total') if isinstance(safe,dict) else getattr(safe,'total',0)) or None
            sizes=[int(file.size) for file in (getattr(info,'siblings',None) or [])
                if str(getattr(file,'rfilename','')).endswith('.safetensors') and
                getattr(file,'size',None) is not None]
            if sizes:
                required_disk=sum(sizes) # network download+storage, not VRAM allocation.
        except (ImportError,ValueError,OSError,RuntimeError,TypeError,KeyError,ConnectionError,TimeoutError):
            pass
        hidden=layers=vocab=0
        if cfg:
            try:
                hidden=int(cfg.get('hidden_size') or cfg.get('n_embd') or cfg.get('d_model') or 0)
                layers=int(cfg.get('num_hidden_layers') or cfg.get('n_layer') or cfg.get('num_layers') or 0)
                vocab=int(cfg.get('vocab_size') or 0)
                intermediate=int(cfg.get('intermediate_size') or 4*hidden)
                if not parameters and min(hidden,layers,vocab)>0:
                    parameters=int(layers*(4*hidden*hidden+2*hidden*intermediate)+vocab*hidden)
            except (ValueError,TypeError,OverflowError):pass
        # Model Hub `info.config` can be a truncated summary. It is not a
        # substitute for an auditable config.json; never mark it trainable.
        if not config_identity_verified:
            return {'status':'unknown',
                    'reason':'Model type not verified from the actual config.json. '
                             'Run hf-diagnose on the exact model/revision; '
                             'model task and LoHA adapter kind do not identify the base model architecture.',
                    'parameters':parameters}
        if not parameters and not (cfg and hidden and layers):
            return {'status':'unknown','reason':'Cannot verify size or architecture from cached/public metadata. '
                    'Run hf-diagnose for the exact model ID and revision before starting training.',
                    'parameters':None}
        try:
            from .device_planner import detect_device_budget
            dvc=detect_device_budget(directory=str(self.workdir))
        except (ImportError,OSError,RuntimeError,ValueError):
            return {'status':'unknown','reason':'Cannot read device memory/disk budget',
                    'parameters':parameters}
        # Conservative minimum workset; do not use total params as an OOM proxy
        # since v30 can stream supported frozen tensors.
        least_ram=32*1024**2+max(64*1024**2,hidden*hidden*2) if hidden else None
        if least_ram and dvc.available_ram_bytes and least_ram>dvc.available_ram_bytes:
            return {'status':'unavailable','reason':'Insufficient RAM for the minimum layer workset',
                    'parameters':parameters,'min_ram_bytes':least_ram}
        if required_disk and dvc.free_disk_bytes and required_disk>dvc.free_disk_bytes:
            return {'status':'unavailable','reason':'Insufficient free disk for published safetensors files',
                    'parameters':parameters,'min_disk_bytes':required_disk}
        return {'status':'candidate','reason':('Metadata-based estimate; architecture/remote code and kernels require testing'),
            'parameters':parameters,'min_disk_bytes':required_disk}

    def register_room(self, payload: dict):
        # Explicit network action. Saving a manifest locally remains possible
        # without Mummy online; published metadata is immutable thereafter.
        room=validate_room(payload['room'])
        path=self._room_path(room)
        if not path.is_file():
            raise FileNotFoundError('Save the room settings locally first')
        manifest=load_manifest(path)
        ws=check_signaling_url(payload.get('ws',''))
        asyncio.run(register_room_manifest(ws,room,manifest,
            publish=payload.get('room_public',False) is True,
            token=os.environ.get('APP_TOKEN',''),timeout=8))
        return {'registered':True,'room':room,'public':payload.get('room_public',False) is True}

    def list_rooms(self, payload: dict):
        # Read-only, bound to the user's chosen signaling host. Run on the
        # pywebview bridge worker thread; no HTTP service runs in this app.
        url = check_signaling_url(payload.get('ws',''))
        return {'rooms': list_public_rooms(url, timeout=5)}

    def upload_code(self, payload: dict):
        from .code_registry import register_base64
        return register_base64(self.workdir/'.fedpara-gui'/'codes',payload.get('name'),payload.get('contents'))

    def mixed_settings(self, payload: dict):
        """Return this machine's private datasets/critic selections for a room.

        No local dataset paths are exposed through public room manifests.
        """
        from .mix_gui_v66 import DEFAULT
        room=validate_room(payload['room'])
        folder=self._room_path(room).parent
        if not self._room_path(room).is_file():
            raise FileNotFoundError('Select a locally saved model first')
        with self.lock:
            if room not in self.profiles:
                self.profiles[room]=allocate_profile(folder)
            profile=self.profiles[room]
        config=folder/('mixed-local-'+profile.name+'.json')
        if not config.is_file():
            result=dict(DEFAULT)
        else:
            data=json.loads(config.read_text(encoding='utf8'))
            if not isinstance(data,dict) or data.get('room')!=room or not (set(data)-{'room'}).issubset(DEFAULT):
                raise ValueError('Local mixed training configuration is malformed')
            result={**DEFAULT,**{k:data[k] for k in data if k!='room'}}
        # This contains only room-wide, nonsecret hyperparameters, never local
        # dataset paths or imported critic checkpoint paths.
        pinned=folder/'mixed-objectives-v66.json'
        # A fresh GUI room with no private selections gets a usable default
        # if the user selects an ordinary HF classifier as the value critic.
        # A saved setting or a pinned legacy room is never silently migrated.
        if not config.is_file() and not pinned.exists():
            result['critic_hf_head_init']='new'
        if pinned.exists():
            policy=json.loads(pinned.read_text(encoding='utf8'))
            if not isinstance(policy,dict) or not isinstance(policy.get('objectives'),list):
                raise ValueError('Pinned mixed objective policy is invalid')
            result['pinned_policy']=policy
        return result

    def start(self, payload: dict):
        """Delegated room training launch; see webview_lifecycle_v110."""
        from .webview_lifecycle_v110 import start_training
        return start_training(self, payload, profile_allocator=allocate_profile)

    def status(self):
        with self.lock:
            running=self.train.status()
            proc=self.train.proc
            from .live_telemetry import recent_samples
            # The full chart is fetched independently. Keep the fast status path
            # small enough for subsecond refreshes even for long-running jobs.
            samples=recent_samples(self._telemetry_path,max_samples=32,max_bytes=32*1024)
            return {'running':running,'telemetry':samples,
                    'stopping':bool(running and getattr(self.train,'stop_requested',False)),
                    'relay_only':bool(running and self._pause_file and self._pause_file.exists()),
                    'model_id':self._current_room,
                    'pid':proc.pid if running and proc is not None else None,
                    'exit_code':proc.returncode if proc is not None and not running else None,
                    'checkpoint_file':str(self._checkpoint_path) if self._checkpoint_path else None,
                    'checkpoint_ready':bool(self._checkpoint_path and self._checkpoint_path.is_file()),
                    'model_storage_dir':str(self._model_storage_path) if self._model_storage_path else None,
                    'logs':self.train.tail()}

    def training_history(self, payload: dict):
        """Bounded sampling of this GUI's private on-disk telemetry."""
        if not isinstance(payload, dict):
            raise ValueError('history options required')
        from .live_telemetry import history_samples
        with self.lock:
            path=self._telemetry_path
            room=self._current_room
        samples=history_samples(path,window_seconds=payload.get('window_seconds',1800),
                                max_points=payload.get('max_points',420))
        return {'model_id':room,'telemetry':samples,
                'window_seconds':payload.get('window_seconds',1800)}

    def storage_paths(self, payload: dict):
        """Return stable room-owned save locations, not an arbitrary user path."""
        room=validate_room(payload['room'])
        root=self._room_path(room).parent
        return {'checkpoints':str(root/'checkpoints'),
                'models':str(root/'models'),
                'genesis':str(root),
                'latest_checkpoint':str(self._checkpoint_path) if self._checkpoint_path else None}

    def open_storage(self, payload: dict):
        """Open one of three validated local directories in the native file manager."""
        if not isinstance(payload,dict):
            raise ValueError('invalid storage request')
        kind=payload.get('kind')
        if kind not in ('checkpoints','models','genesis'):
            raise ValueError('invalid storage kind')
        paths=self.storage_paths(payload)
        target=Path(paths[kind]).resolve()
        root=(self.workdir/'.fedpara-gui'/'rooms').resolve()
        if not target.is_relative_to(root):
            raise PermissionError('storage directory escapes room root')
        target.mkdir(parents=True,exist_ok=True,mode=0o700)
        if sys.platform=='win32':
            os.startfile(str(target))
        elif sys.platform=='darwin':
            subprocess.Popen(['open',str(target)],stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        else:
            opener=shutil.which('xdg-open')
            if not opener:
                raise RuntimeError('xdg-open がありません。保存先: '+str(target))
            subprocess.Popen([opener,str(target)],stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return {'path':str(target)}

    def copy_logs(self):
        """Copy the recent diagnostic log using a native clipboard command.

        Avoid navigator.clipboard, which is not consistently available to a
        local pywebview page on macOS. No shell interpolation is performed.
        """
        with self.lock:
            content=self.train.tail(max_bytes=2*1024*1024)
        if not content:
            raise ValueError('コピーできる学習ログがありません。')
        if sys.platform=='darwin':
            command=['pbcopy']
        elif sys.platform=='win32':
            command=['clip']
        else:
            command=next((cmd for cmd in (['wl-copy'],['xclip','-selection','clipboard'],['xsel','--clipboard','--input'])
                          if shutil.which(cmd[0])),None)
            if command is None:
                raise RuntimeError('クリップボードツールが見つかりません。ログをファイルへ保存してください。')
        try:
            subprocess.run(command, input=content, text=True, check=True,timeout=5,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            raise RuntimeError('クリップボードへのコピーに失敗しました。ログをファイルへ保存してください。') from exc
        return {'chars':len(content)}

    def save_logs(self):
        """Save the full local log to a private diagnostic file with a safe name."""
        with self.lock:
            logfile=self.train.logfile
            if logfile is None or not logfile.is_file():
                raise FileNotFoundError('保存する学習ログがまだありません。')
            output=self.workdir/'.fedpara-gui'/'diagnostics'
            output.mkdir(parents=True,exist_ok=True,mode=0o700)
            os.chmod(output,0o700)
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            dest=output/f'kagami-{stamp}-{secrets.token_hex(4)}.log'
            fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            try:
                with os.fdopen(fd,'wb') as out,logfile.open('rb') as source:
                    shutil.copyfileobj(source,out,length=256*1024)
            except BaseException:
                dest.unlink(missing_ok=True)
                raise
            return {'path':str(dest),'bytes':dest.stat().st_size}

    def pause_training(self):
        """Persist relay-only mode without closing authenticated WebRTC edges."""
        with self.lock:
            if not self.train.status() or self._pause_file is None:
                return {'running':False,'relay_only':False}
            self._pause_file.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            # Secure, idempotent sentinel. Worker seals any pending local update,
            # continues signed gossip, catch-up and snapshot service.
            fd=os.open(self._pause_file,os.O_CREAT|os.O_WRONLY|getattr(os,'O_NOFOLLOW',0),0o600)
            os.close(fd)
            # Immediately request a learned-weight snapshot after the worker
            # seals local updates. Relay/Gossip remain available throughout.
            if self._snapshot_request_file and not self._snapshot_request_file.exists():
                self._queue_eval_snapshot()
            return {'running':True,'relay_only':True,'save_requested':True}

    def resume_training(self):
        with self.lock:
            if not self.train.status() or self._pause_file is None:
                return {'running':False,'relay_only':False}
            self._pause_file.unlink(missing_ok=True)
            return {'running':True,'relay_only':False}

    def stop(self):
        """Explicit full disconnect; normal UI stop uses pause_training."""
        with self.lock:
            self.train.stop()
            return {'stopping':True}

    def local_models(self, payload=None):
        """Local manifests, including unpublished models, never remote code."""
        base=self.workdir/'.fedpara-gui'/'rooms'
        items=[]
        if base.exists():
            for candidate in __import__('itertools').islice(base.glob('*/manifest.json'), 200):
                try:
                    obj=load_manifest(candidate)
                    if obj['version'] != 2: continue
                    items.append({'room_id':candidate.parent.name,
                        'model':obj['model']['id'],
                        'revision':obj['model']['revision'],
                        'task':obj['model']['architecture'],
                        'datasets':[x['repo'] for x in obj['datasets']['sources']],
                        'online':0,'capacity':0,'local':True,
                        'is_training':self.train.status() and self._current_room==candidate.parent.name})
                except (OSError,ValueError,KeyError):continue
        return {'models':items}

    def device_capabilities(self, _payload=None):
        from .device_planner import detect_device_budget
        try:
            d=detect_device_budget(directory=str(self.workdir),memory_fraction=.6)
            import torch
            mps_available=bool(torch.backends.mps.is_available())
            accelerator_bytes=d.available_accelerator_bytes
            if mps_available and not accelerator_bytes:
                # The v92 automatic safety policy intentionally reports a CPU
                # planning budget; still display the available unified GPU
                # working-set size (do NOT add it to the RAM budget).
                try:
                    accelerator_bytes=max(0,int(torch.mps.recommended_max_memory())
                                       -int(torch.mps.current_allocated_memory()))
                except (AttributeError,RuntimeError):pass
            return {'device':d.device,'mps_available':mps_available,
                    'cuda_available':bool(torch.cuda.is_available()),
                    'ram_bytes':d.available_ram_bytes,
                    'vram_bytes':accelerator_bytes,
                    'disk_bytes':d.free_disk_bytes,
                    'budget_bytes':d.usable_bytes}
        except (OSError,RuntimeError,ValueError) as e:
            return {'device':'cpu','mps_available':False,'cuda_available':False,
                    'ram_bytes':0,'vram_bytes':0,
                    'disk_bytes':0,'budget_bytes':0,'warning':str(e)}

    def recommend_training(self, payload):
        """Choose local runtime settings without executing downloaded code.

        Never wait on the Hub from the quick-start path. The peer membership
        controller determines cohort size from verified signaling/heartbeats.
        """
        from .auto_policy import estimate_training_policy
        if not isinstance(payload,dict):
            raise ValueError('training plan payload required')
        model=payload.get('model','')
        revision=payload.get('revision','main')
        from .room_manifest import HF_REPO, DATA_PART
        if not isinstance(model,str) or not HF_REPO.fullmatch(model):
            raise ValueError('invalid model repository')
        if not isinstance(revision,str) or not DATA_PART.fullmatch(revision):
            raise ValueError('invalid model revision')
        config=None
        descriptor=None
        room=payload.get('room')
        if isinstance(room,str):
            try:
                path=self._room_path(room)
                descriptor=(load_manifest(path) if path.is_file() else self.imported.get(room))
                if descriptor and descriptor['model']['id']==model and descriptor['model']['revision']==revision:
                    design=descriptor.get('design') or {}
                    if design.get('mode')=='code':config=dict(design['kwargs'])
            except (ValueError,OSError,KeyError):pass
        if not config:
            try:
                from .hf_config_v83 import load_hf_config
                config=load_hf_config(model,revision=revision,
                          local_files_only=True,trust_remote_code=False).to_dict()
            except (ImportError,OSError,KeyError,ValueError,RuntimeError,TypeError):pass
        if not config:
            # Lightweight metadata lookup; this does not import model scripts,
            # execute remote Python or download model weights.
            try:
                from huggingface_hub import HfApi
                info=HfApi().model_info(model,revision=revision,timeout=4)
                if isinstance(getattr(info,'config',None),dict):
                    config=dict(info.config)
            except (ImportError,OSError,KeyError,ValueError,RuntimeError,TypeError,ConnectionError,TimeoutError):pass
        try:
            dev=self.device_capabilities()
        except (OSError,ValueError,RuntimeError):
            dev={}
        from .auto_policy import estimate_adapter_rank
        pinned=(descriptor.get('adapter') if descriptor and
                descriptor['model']['id']==model and
                descriptor['model']['revision']==revision else None)
        kind=pinned['kind'] if pinned else payload.get('kind','loha')
        rank_plan=estimate_adapter_rank(config=config,
                available_ram_bytes=dev.get('ram_bytes') or 0,
                available_vram_bytes=dev.get('vram_bytes') or 0,kind=kind)
        rank=pinned['rank'] if pinned else rank_plan['rank']
        policy=estimate_training_policy(config=config,
                available_ram_bytes=dev.get('ram_bytes') or 0,
                available_vram_bytes=dev.get('vram_bytes') or 0,
                rank=rank,kind=kind)
        capacity=rank_plan['rank_capacity']
        policy.update({'rank':rank,'rank_pinned':bool(pinned),
                       'wide_lora_rank':pinned['wide_lora_rank'] if pinned else 4,
                       'rank_capacity':capacity,
                       'can_train':capacity is None or rank<=capacity,
                       'kind':kind,'rank_estimated_wire_bytes':rank_plan['wire_estimate_bytes']})
        return policy

    def inspect_dataset(self, payload):
        """Bounded read-only Hub inspection; never import remote dataset scripts."""
        from .room_manifest import HF_REPO, DATA_PART
        if not isinstance(payload,dict):raise ValueError('invalid inspection payload')
        repo=payload.get('repo','');revision=payload.get('revision','main')
        if not isinstance(repo,str) or not HF_REPO.fullmatch(repo):
            raise ValueError('invalid dataset namespace/repo')
        if not isinstance(revision,str) or not DATA_PART.fullmatch(revision):
            raise ValueError('invalid dataset revision')
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError('Install huggingface_hub to inspect datasets') from exc
        hub=HfApi()
        # Hub parquet listings are metadata, not dataset downloads or execution.
        entries=hub.list_dataset_parquet_files(repo,revision=revision)
        variants=[]
        seen=set()
        for item in entries:
            config=str(getattr(item,'config','') or '')
            split=str(getattr(item,'split','') or '')
            if not config or not split or len(config)>120 or len(split)>120:continue
            if (config,split) not in seen:
                variants.append({'config':config,'split':split})
                seen.add((config,split))
            if len(variants)>=64:break
        return {'repo':repo,'revision':revision,'variants':variants,'available':bool(variants)}

    def pin_mixed_dataset(self, payload):
        """Resolve a Hub dataset ID to a commit and optional config/split metadata.

        Does not download training rows or run repository code.
        """
        if not isinstance(payload,dict):raise ValueError('Invalid Hugging Face dataset selection')
        from .hf_mixed_datasets_v68 import _REPO, _SHA
        repo=payload.get('repo','')
        if not isinstance(repo,str) or not _REPO.fullmatch(repo):
            raise ValueError('Expected organization/dataset repository ID')
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError('Install huggingface_hub for dataset browsing') from exc
        api=HfApi()
        info=api.dataset_info(repo)
        sha=str(getattr(info,'sha','') or '')
        if not _SHA.fullmatch(sha):
            raise ValueError('Hub did not return a valid immutable dataset SHA')
        variants=[];seen=set()
        try:
            for entry in api.list_dataset_parquet_files(repo,revision=sha):
                config=str(getattr(entry,'config','') or '')
                split=str(getattr(entry,'split','') or '')
                if config and split and (config,split) not in seen:
                    variants.append({'config':config,'split':split})
                    seen.add((config,split))
                if len(variants)>=64:break
        except Exception:
            # Not every Hub dataset exposes Parquet metadata; the pinned SHA
            # remains valid and the user can supply split/config manually.
            pass
        return {'repo':repo,'revision':sha.lower(),'variants':variants,
                'gated':bool(getattr(info,'gated',False))}

    def probe_mixed_dataset(self, payload):
        """Explicit opt-in: validate first streaming row, never download model weights.

        Return field names and selected mapping only, not user data content.
        This is per-peer local configuration; it never changes the room policy.
        """
        from types import SimpleNamespace
        from .hf_mixed_datasets_v68 import source_spec, HFRowCycle
        if not isinstance(payload,dict) or payload.get('role') not in ('sft','ppo'):
            raise ValueError('Expected SFT or PPO dataset role')
        role=payload['role']
        other='response' if role=='sft' else 'reference'
        opts={role+'_hf_'+k:payload.get(k,'') for k in
              ('repo','revision','config','split','prompt_column',other+'_column')}
        opts[role+'_jsonl']=None
        # source_spec handles repo/SHA/field validation; no trust_remote_code.
        spec=source_spec(SimpleNamespace(**opts),role)
        if spec is None:raise ValueError('Choose a pinned Hub dataset first')
        source=HFRowCycle(spec,role)
        try:
            row=source.peek()
            mode,field_names=source._resolved_mapping
            if (role=='ppo' and payload.get('reward','token_match')!='hf_model'
                    and 'reference' not in row):
                raise ValueError('PPO token_match/exact reward requires a reference column')
            return {'role':role,'repo':spec['repo'],'columns':list(source._last_columns),
                    'mode':mode,'prompt_column':field_names[0] if field_names else 'messages',
                    other+'_column':field_names[1] if field_names else 'messages',
                    'has_reference':bool('reference' in row),'ready':True}
        finally:
            source.close()

    def discover_hub_assets(self, payload):
        """Purpose-filtered Hugging Face candidate discovery (metadata only)."""
        from .hub_discovery_v70 import discover_hub_assets, HubDiscoveryCache
        if not hasattr(self, '_hub_discovery_cache'):
            self._hub_discovery_cache=HubDiscoveryCache()
        return discover_hub_assets(payload,cache=self._hub_discovery_cache)

    def search_reward_models(self, payload):
        """Metadata-only Hub discovery; never download or execute model code."""
        if not isinstance(payload,dict):raise ValueError('Expected search object')
        query=payload.get('query','')
        if not isinstance(query,str) or not 1<=len(query.strip())<=100:
            raise ValueError('Search query must contain 1..100 characters')
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError('Install huggingface_hub for Reward Model search') from exc
        hits=HfApi().list_models(search=query,sort='downloads',limit=15)
        from .reward_model_v67 import _HF_REPO, _SHA
        results=[]
        for m in hits:
            name=str(getattr(m,'id',''))
            if _HF_REPO.fullmatch(name):
                commit=str(getattr(m,'sha','') or '')
                results.append({'id':name,'revision':commit if _SHA.fullmatch(commit) else '',
                                'pipeline':str(getattr(m,'pipeline_tag','') or ''),
                                'gated':bool(getattr(m,'gated',False))})
        return {'models':results}

    def inspect_reward_model(self, payload):
        """Resolve a user's selected repo to an immutable pinned HF revision."""
        if not isinstance(payload,dict):raise ValueError('Expected Reward Model object')
        repo=payload.get('repo','')
        from .reward_model_v67 import _HF_REPO, _SHA
        if not isinstance(repo,str) or not _HF_REPO.fullmatch(repo):
            raise ValueError('Expected organization/repo Hugging Face ID')
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError('Install huggingface_hub for Reward Model inspection') from exc
        m=HfApi().model_info(repo,files_metadata=False)
        commit=str(getattr(m,'sha','') or '')
        if not _SHA.fullmatch(commit):
            raise ValueError('Hub did not provide a valid immutable commit SHA')
        return {'id':repo,'revision':commit.lower(),
                'pipeline':str(getattr(m,'pipeline_tag','') or ''),
                'gated':bool(getattr(m,'gated',False))}

    def search_datasets(self, payload):
        """Bounded public HF Hub search, no arbitrary URL or remote Python."""
        if not isinstance(payload,dict):raise ValueError('dataset search object required')
        query=payload.get('query','')
        if not isinstance(query,str) or len(query)>100:raise ValueError('invalid query')
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError('Install huggingface_hub for online dataset search') from exc
        api=HfApi()
        matches=api.list_datasets(search=query or None,sort='downloads',limit=25)
        return {'datasets':[{'id':item.id,
            'downloads':int(item.downloads or 0),
            'revision':getattr(item,'sha',None) or 'main',
            'gated':bool(getattr(item,'gated',False))}
            for item in matches]}

    def available_checkpoints(self,payload):
        """Find local completed learned-weight snapshots; no entire shard scans."""
        from .shards import read_manifest
        room=validate_room(payload.get('room',''))
        root=self._room_path(room).parent.resolve()
        candidates=[]
        collection=root/'checkpoints'
        if collection.is_dir():
            # Bounded result, O(total file count) scan but no memory proportional
            # to the number of checkpoints/profiles. Refuse symlink escapes.
            import heapq
            newest=[]
            for folder in collection.iterdir():
                if not folder.is_dir() or folder.is_symlink():continue
                for file in folder.glob('*.shards.json'):
                    try:
                        if file.is_symlink() or not file.resolve().is_relative_to(root):continue
                        key=(file.stat().st_mtime_ns,str(file))
                        if len(newest)<8:heapq.heappush(newest,(key,file))
                        elif key>newest[0][0]:heapq.heapreplace(newest,(key,file))
                    except OSError:continue
            candidates=[file for key,file in sorted(newest,reverse=True)]
        result=[]
        for path in candidates:
            try:
                info=read_manifest(path,max_state_bytes=2048*1024**3)
                descriptor=load_manifest(self._room_path(room))
                if (info['model_id']!=descriptor['model']['id'] or
                    info['revision']!=descriptor['model']['revision']):continue
                store=root/'models'/path.parent.name
                if not store.is_dir() or store.is_symlink() or not store.resolve().is_relative_to(root):continue
                result.append({'file':str(path),'store':str(store),'seq':info['seq'],
                               'bytes':info['total_tensor_bytes'],
                               'revision':info['revision'], 'tensors':len(info['tensors'])})
            except (OSError,ValueError,KeyError):continue
        return {'checkpoints':result}

    def _queue_eval_snapshot(self):
        if not self._snapshot_request_file:
            raise RuntimeError('Snapshot request path is not configured')
        if self._snapshot_request_file.exists():
            return {'requested':True,'pending':True}
        status=self._snapshot_request_file.with_suffix('.status.json')
        # Existing 'saving' is not an invitation to start two exports.
        try:
            old=json.loads(status.read_text(encoding='utf8'))
            if old.get('state')=='saving' and self.train.status():
                return {'requested':True,'pending':True}
        except (OSError,ValueError,AttributeError):pass
        fd=os.open(self._snapshot_request_file,
                   os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
        with os.fdopen(fd,'wb') as f:
            f.write(b'eval\n')
        status.unlink(missing_ok=True)
        return {'requested':True,'pending':False,
                'path':str(self._checkpoint_path),
                'message':'Saving learned weights; EVAL will be ready after completion.'}

    def request_eval_snapshot(self,payload):
        """Asynchronous on-demand learned-model export without dropping Gossip."""
        room=validate_room(payload.get('room',''))
        with self.lock:
            if not self.train.status() or self._current_room!=room:
                raise RuntimeError('No active trainer. Start or resume this model, then save weights. '
                                   'A stopped worker cannot export its in-memory state.')
            return self._queue_eval_snapshot()

    def eval_snapshot_status(self,payload):
        room=validate_room(payload.get('room',''))
        with self.lock:
            if room!=self._current_room:
                # Old model's completed checkpoints are still discoverable.
                return {'state':'idle','room':room}
            request=self._snapshot_request_file
            if not request:
                return {'state':'idle','room':room}
            status=request.with_suffix('.status.json')
            try:
                info=json.loads(status.read_text(encoding='utf8'))
                if info.get('state') not in ('saving','complete','error'):
                    raise ValueError('invalid save status')
                if info['state']=='complete':
                    if not (self._checkpoint_path and self._checkpoint_path.is_file()):
                        return {'state':'error','error':'Completed checkpoint manifest is missing'}
                    # File existence is not proof of a valid exported model.
                    # Make the status actionable rather than silently retrying
                    # the same snapshot forever when the manifest is corrupt.
                    from .shards import read_manifest
                    try:
                        checkpoint=read_manifest(self._checkpoint_path,max_state_bytes=2048*1024**3)
                        expected=load_manifest(self._room_path(room))['model']
                        if checkpoint['model_id']!=expected['id'] or checkpoint['revision']!=expected['revision']:
                            raise ValueError('checkpoint model/revision mismatch')
                    except (OSError,KeyError,ValueError) as exc:
                        return {'state':'error','error':'Checkpoint manifest invalid: '+str(exc)[:500]}
                if info['state']=='saving' and not self.train.status():
                    return {'state':'error','error':'Worker exited before save completion. Check TRAINING log.'}
                return info
            except (OSError,ValueError,TypeError,KeyError):
                if request.exists():return {'state':'queued','room':room}
                return {'state':'idle','room':room}

    def evaluate(self,payload):
        """Run local HF generation, from the *trained checkpoint* by default."""
        from .room_manifest import HF_REPO, DATA_PART
        if not isinstance(payload,dict):raise ValueError('bad evaluation payload')
        room=validate_room(payload.get('room',''))
        manifest=load_manifest(self._room_path(room))
        name=manifest['model']['id'];revision=manifest['model']['revision']
        if not HF_REPO.fullmatch(name) or not DATA_PART.fullmatch(revision):
            raise ValueError('Only Hub-hosted tokenizer/revision IDs are supported')
        if manifest['model']['architecture'] not in ('causal-lm','seq2seq-lm','masked-lm','sequence-classification'):
            raise ValueError('Unsupported evaluation model task')
        mode=payload.get('precision','auto')
        if mode not in ('auto','fp32','fp16','bf16','int8','fp4','nf4','torchao-int4'):
            raise ValueError('unsupported precision')
        prompt=payload.get('prompt','')
        if not isinstance(prompt,str) or len(prompt)>8000:
            raise ValueError('prompt must be 0..8000 characters')
        tokens=payload.get('max_tokens',128)
        if type(tokens) is not int or not 1<=tokens<=512:
            raise ValueError('max_tokens must be 1..512')
        scope=payload.get('scope','trained')
        if scope not in ('trained','base'):raise ValueError('unsupported inference scope')
        options={'model':name,'revision':revision,'manifest':manifest,
                 'precision':mode,'prompt':prompt,'max_tokens':tokens,'scope':scope}
        if scope=='trained':
            saved=self.available_checkpoints({'room':room})['checkpoints']
            if not saved:
                # One-click trained inference: first snapshot the exact learned
                # weights, then let the UI retry the same evaluation request
                # after the file is atomically published. Never use base silently.
                with self.lock:
                    if self.train.status() and self._current_room==room:
                        pending=self._queue_eval_snapshot()
                        return {'pending_snapshot':True,'message':pending.get('message','Saving learned weights')}
                progress=self.eval_snapshot_status({'room':room})
                if progress.get('state') in ('saving','queued'):
                    return {'pending_snapshot':True,'message':'Checkpoint export is in progress'}
                reason=(' 保存エラー: '+str(progress.get('error')) if progress.get('state')=='error' else '')
                raise FileNotFoundError('学習済みモデルが未保存です。学習を開始または再開し、'
                                        '重み保存を完了してから推論してください。'+reason)
            chosen=saved[0]
            options.update({'checkpoint':chosen['file'],'store':chosen['store'],
                            'checkpoint_revision':chosen['revision']})
        with self.lock:
            if self._eval_proc and self._eval_proc.poll() is None:
                raise RuntimeError('An inference job is already running')
            root=self.workdir/'.fedpara-gui'/'evals'
            root.mkdir(parents=True,exist_ok=True,mode=0o700)
            options['temp_dir']=str(root.resolve())
            token=secrets.token_hex(12)
            inp=root/(token+'.input.json');out=root/(token+'.output.json');log=root/(token+'.log')
            fd=os.open(inp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w',encoding='utf8') as file:
                json.dump(options,file,ensure_ascii=False)
            with log.open('wb') as fh:
                proc=subprocess.Popen([sys.executable,'-m','patchlib.kagami_eval_worker',str(inp),str(out)],
                    cwd=Path(__file__).resolve().parent.parent,stdout=fh,stderr=subprocess.STDOUT,
                    start_new_session=(os.name=='posix'))
            self._eval_proc=proc;self._eval_result=out;self._eval_log=log
            return {'pid':proc.pid,'precision':mode,
                    'scope':('Kagami trained effective weights' if scope=='trained' else 'Explicit HF base'),
                    'checkpoint_seq':chosen['seq'] if scope=='trained' else None}

    def eval_status(self,_payload=None):
        with self.lock:
            proc=self._eval_proc
            if proc is None:return {'running':False,'done':False}
            code=proc.poll()
            result=None
            if code is not None and self._eval_result and self._eval_result.exists():
                try:result=json.loads(self._eval_result.read_text(encoding='utf8'))
                except (OSError,ValueError):result={'error':'inference result unreadable'}
            # An abrupt kill/import failure may produce logs but no result JSON.
            # Never display such a job as a successful generation.
            if code is not None and result is None:
                result={'error':'Inference worker exited without a result file (exit='+str(code)+')',
                        'hint':'Check the EVAL log and install requirements-eval.txt into the interpreter launching Kagami.'}
            logs=''
            if self._eval_log and self._eval_log.exists():
                with self._eval_log.open('rb') as fh:
                    fh.seek(0,2)
                    fh.seek(max(0,fh.tell()-3000))
                    logs=fh.read().decode('utf8','replace')
            return {'running':code is None,'done':code is not None,'exit_code':code,'result':result,'logs':logs}

    def cancel_eval(self,_payload=None):
        with self.lock:
            if self._eval_proc and self._eval_proc.poll() is None:
                self._eval_proc.terminate()
            return {'cancelled':True}

    def start_benchmark(self, payload: dict):
        """One local subprocess per GUI. This does not open signaling sockets."""
        with self.lock:
            if self._bench_proc is not None and self._bench_proc.poll() is None:
                raise RuntimeError('benchmark is already running in this GUI')
            if not isinstance(payload,dict):
                raise ValueError('expected benchmark options object')
            def option(name, default, lo, hi):
                value=payload.get(name,default)
                if type(value) is not int or not lo<=value<=hi:
                    raise ValueError(f'{name} must be an integer in {lo}..{hi}')
                return value
            peers=option('peers',4,2,16)
            rounds=option('rounds',5,1,200)
            steps=option('local_steps',2,1,40)
            reinit_every=option('reinit_every',20,0,200)
            model=payload.get('model','tinystories-1m')
            rank=option('rank',16 if model=='tinystories-1m' else 4,2,16)
            dataset=payload.get('dataset','tinystories' if model=='tinystories-1m' else 'auto')
            sequences=option('sequences',2048 if model=='tinystories-1m' else 64,1,8192)
            validation_sequences=option('validation_sequences',1024 if model=='tinystories-1m' else 64,16,8192)
            if model not in ('toy','tinystories-1m') or dataset not in ('auto','builtin','tinystories'):
                raise ValueError('invalid benchmark model or dataset')
            if model=='toy' and dataset!='auto':
                raise ValueError('toy model only supports automatic synthetic data')
            lr=payload.get('lr',1e-4 if model=='tinystories-1m' else 0.003)
            if type(lr) not in (float,int) or not 1e-6 <= lr <= .1:
                raise ValueError('invalid benchmark learning rate')
            initialization=payload.get('initialization','random')
            if initialization not in ('random','pretrained'):
                raise ValueError('invalid benchmark initialization')
            optimizer=payload.get('optimizer','adamw')
            if optimizer not in ('adamw','lora-muon'):
                raise ValueError('invalid benchmark optimizer')
            kind=payload.get('kind','loha')
            if kind not in ('loha','lora'):
                raise ValueError('invalid adapter kind')
            if optimizer=='lora-muon' and kind!='lora':
                raise ValueError('LoRA-Muon requires kind=lora')
            base=self.workdir/'.fedpara-gui'/'benchmarks'
            base.mkdir(parents=True,exist_ok=True,mode=0o700)
            os.chmod(base,0o700)
            dest=base/('run-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+secrets.token_hex(4))
            dest.mkdir(mode=0o700)
            logfile=dest/'benchmark.log'
            command=[sys.executable,'-m','patchlib.cli','benchmark-effectiveness',
                     '--peers',str(peers),'--rounds',str(rounds),'--local-steps',str(steps),
                     '--kind',kind,'--optimizer',optimizer,'--rank',str(rank),'--lr',str(lr),
                     '--reinit-every',str(reinit_every),
                     '--model',model,'--dataset',dataset,'--initialization',initialization,
                     '--sequences',str(sequences),'--validation-sequences',str(validation_sequences),
                     '--context','64' if model=='tinystories-1m' else '12',
                     '--output-dir',str(dest)]
            env=os.environ.copy()
            env['OMP_NUM_THREADS']='1'
            env['MKL_NUM_THREADS']='1'
            with logfile.open('wb') as out:
                self._bench_proc=subprocess.Popen(command,cwd=str(Path(__file__).resolve().parent.parent),
                     env=env,stdout=out,stderr=subprocess.STDOUT)
            self._bench_log=logfile
            self._bench_output=dest
            return {'pid':self._bench_proc.pid,'output_dir':str(dest)}

    def benchmark_status(self):
        with self.lock:
            proc=self._bench_proc
            if proc is None:
                return {'running':False,'exit_code':None,'logs':'','result':None}
            code=proc.poll()
            logfile=self._bench_log
            logs=''
            if logfile and logfile.is_file():
                with logfile.open('rb') as fh:
                    fh.seek(0,2);fh.seek(max(0,fh.tell()-5000))
                    logs=fh.read().decode('utf8','replace')
            summary=None
            report=self._bench_output/'effectiveness-result.json'
            if code==0 and report.is_file():
                # Report generated by our own fixed command, not a remote payload.
                content=json.loads(report.read_text(encoding='utf8'))
                summary={'loss':content['loss'],'compute':content['compute'],
                         'configuration':content['configuration'],'artifacts':{
                             k:str(self._bench_output/k) for k in
                             ('effectiveness-result.json','effectiveness-trace.csv',
                              'effectiveness-loss.png','effectiveness-time.png') if (self._bench_output/k).exists()},
                         'scope':content['scope']}
            return {'running':code is None,'exit_code':code,'logs':logs,
                    'result':summary,'output_dir':str(self._bench_output)}

    def close(self):
        # Do not release the identity while our training child is still alive.
        with self.lock:
            if self._bench_proc is not None and self._bench_proc.poll() is None:
                self._bench_proc.terminate()
                try:self._bench_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._bench_proc.kill();self._bench_proc.wait(timeout=3)
            try:
                # On a multi-hundred-GB checkpoint the detached worker may
                # outlive the 30s UI cleanup budget; do not SIGKILL it. Its
                # EventJournal still owns the exclusive training lock and it
                # continues saving to the durable model store.
                # Closing Kagami shuts down the peer; relay-only mode persists
                # while the window remains open. Checkpoint export is graceful.
                self.train.stop(wait=True)
            except RuntimeError as exc:
                print('Kagami: model checkpoint still being saved by detached worker; '
                      'do not delete its room directory:',exc,file=sys.stderr)
            finally:
                for profile in self.profiles.values():
                    profile.close()
                self.profiles.clear()

