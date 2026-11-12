"""Model creation, pinned HF loading, adapter placement and disk offload.

Separate from the CLI dispatcher so model and genesis lifecycle can be reviewed
and tested without parsing commands. Existing cli.load_model remains importable.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import torch
from .core import UniversalAdapter
from .state import apply_state, semantic_state_sha256
from .shards import apply_sharded_state, semantic_shard_sha256


AUTO_CLASSES = {
    'base': 'AutoModel',
    'causal-lm': 'AutoModelForCausalLM',
    'masked-lm': 'AutoModelForMaskedLM',
    'seq2seq-lm': 'AutoModelForSeq2SeqLM',
    'sequence-classification': 'AutoModelForSequenceClassification',
    'token-classification': 'AutoModelForTokenClassification',
    'question-answering': 'AutoModelForQuestionAnswering',
    'image-classification': 'AutoModelForImageClassification',
    'audio-classification': 'AutoModelForAudioClassification',
    'speech-seq2seq': 'AutoModelForSpeechSeq2Seq',
    'object-detection': 'AutoModelForObjectDetection',
    'semantic-segmentation': 'AutoModelForSemanticSegmentation',
}


def _untrusted_guard(args):
    """A policy check, NOT a security boundary: sandbox isolation is external."""
    custom = bool(getattr(args,'trust_remote_code',False) or getattr(args,'factory_file',None))
    if not custom:
        return
    if os.environ.get('FEDPARA_ISOLATED_RUNTIME') != '1':
        raise PermissionError('Custom Python code requires the offline Docker sandbox; see sandbox/run_untrusted.sh')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise PermissionError('Custom Python code can only run in explicitly offline runtime')
    if getattr(args,'allow_legacy_bin',False):
        raise PermissionError('Pickle/legacy checkpoint loading forbidden for untrusted code')
    if getattr(args,'trust_remote_code',False):
        if not re.fullmatch(r'[a-fA-F0-9]{40}', args.revision):
            raise PermissionError('Custom Hugging Face remote code requires a pinned 40-hex commit')
        if not Path(args.model).is_dir():
            raise PermissionError('Custom HF code requires a predownloaded local model directory')
    if getattr(args,'cmd','') in ('train-p2p', 'train-live', 'train-elastic'):
        raise PermissionError('Cannot run untrusted model code inside the networked P2P process. Train offline and exchange files outside sandbox.')


def _create_factory(args):
    factory = Path(args.factory_file)
    if not factory.is_file() or factory.suffix != '.py':
        raise ValueError('factory-file must be a local .py file')
    spec = importlib.util.spec_from_file_location('_fedpara_user_factory', factory)
    if spec is None or spec.loader is None:
        raise ValueError('could not load factory')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # ONLY inside isolated offline container
    builder = getattr(module,'build_model',None)
    if not callable(builder):
        raise ValueError('factory-file must define build_model() -> torch.nn.Module')
    model = builder()
    if not isinstance(model, torch.nn.Module):
        raise TypeError('factory must return torch.nn.Module')
    return model


def resolve_device(device):
    """Fail closed on Apple's native MPS aborts; GPU use requires explicit opt-in."""
    from .mps_policy_v92 import check_device, safe_auto_device, notify_safe_cpu
    if str(device) == 'auto':
        cuda = torch.cuda.is_available()
        mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
        selected = safe_auto_device(cuda=cuda, mps=mps)
        if selected == 'cpu' and mps:
            notify_safe_cpu()
        return torch.device(selected)
    check_device(device)
    result = torch.device(device)
    if result.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS selected, but PyTorch MPS is unavailable')
    if result.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected, but no available CUDA device was found')
    return result


def input_device(model, fallback):
    """Use the compute device, not a frozen CPU-mapped disk weight."""
    for mod in model.modules():
        if getattr(mod,'_fedpara_disk_path',None):
            overlay=getattr(mod,'_fedpara_disk_overlay',None)
            if overlay is not None:
                return overlay.local.parts[0].device
            return resolve_device(fallback)
    # An arbitrary PyTorch LM may expose the HF method without implementing it.
    try:
        emb = model.get_input_embeddings() if hasattr(model,'get_input_embeddings') else None
    except (AttributeError, NotImplementedError):
        emb = None
    if emb is not None:
        for param in emb.parameters():
            if param.device.type != 'meta':
                return param.device
    for param in model.parameters():
        return param.device
    return resolve_device(fallback)


def check_adapter_placement(adapter):
    """Detect mixed-device factors *before* the first forward/backward."""
    for group in adapter.groups:
        modpath, _, name = group.aliases[0].rpartition('.')
        original = adapter.model.get_submodule(modpath).parametrizations[name].original
        module=adapter.model.get_submodule(modpath)
        expected=(resolve_device(getattr(adapter.model,'_fedpara_compute_device','cpu'))
                  if hasattr(module,'_fedpara_disk_source') else original.device)
        for factor in group.overlay.local.parts:
            if factor.device != expected:
                raise RuntimeError(f'LoRA/LoHA factor for {group.canonical} on {factor.device}; expected {expected}')


def load_model(args, *, use_adapter: bool):
    _untrusted_guard(args)
    initialization = getattr(args, 'initialization', 'pretrained')
    if initialization not in ('pretrained', 'random', 'config'):
        raise ValueError('initialization must be pretrained/random/config')
    streaming=bool(getattr(args,'streaming_base',False))
    lazy_write=bool(getattr(args,'streaming_lazy_write',False))
    page_cow=bool(getattr(args,'streaming_page_cow',False))
    factor_ledger=bool(getattr(args,'streaming_factor_ledger',False))
    if factor_ledger and (not lazy_write or page_cow):
        raise ValueError('--streaming-factor-ledger requires --streaming-lazy-write and excludes --streaming-page-cow')
    if page_cow and not lazy_write:
        raise ValueError('--streaming-page-cow requires --streaming-lazy-write')
    if page_cow and getattr(args,'generation_v2_trainer',False):
        raise ValueError('paged COW exporter is not yet certified for autonomous v2 generations')
    if lazy_write and (not streaming or getattr(args,'cmd',None)!='train-elastic'
                       or getattr(args,'mode','elastic')!='elastic'):
        raise ValueError('--streaming-lazy-write requires --streaming-base and elastic train-elastic mode')
    if lazy_write and getattr(args,'disk_linear_dir',None):
        raise ValueError('--streaming-lazy-write cannot share --disk-linear-dir scratch')
    if streaming and initialization != 'pretrained':
        raise ValueError('--streaming-base requires pretrained safetensors, not random initialization')
    if streaming and getattr(args,'factory_file',None):
        raise ValueError('--streaming-base requires a Hugging Face meta-compatible architecture')
    if streaming and (getattr(args,'state_base',None) or getattr(args,'sharded_base_manifest',None)):
        raise ValueError('--streaming-base does not yet accept a separate .fstate/sharded initial override')
    if initialization == 'config':
        from .model_design import build_random_model
        from .room_manifest import load_manifest
        manifest_path=getattr(args,'room_manifest',None)
        if not manifest_path:
            raise ValueError('config creation needs an immutable room manifest')
        manifest=load_manifest(manifest_path)
        if manifest['model']['id']!=args.model or manifest['model']['revision']!=args.revision:
            raise ValueError('config manifest/worker identity mismatch')
        mdl, revision=build_random_model(manifest,args.hf_task)
    elif initialization == 'random':
        # Create the SAME GPT-Neo architecture, but do not fetch pretrained
        # weights. This seed only governs model genesis, not data sampling.
        from .live_initialization import create_random_tinystories_model
        if getattr(args, 'factory_file', None):
            raise ValueError('random TinyStories genesis cannot use custom model factory')
        mdl, revision = create_random_tinystories_model(
            args.model, args.revision, getattr(args, 'initial_seed', 42))
    elif getattr(args,'factory_file',None):
        mdl = _create_factory(args)
        factory_sha = hashlib.sha256(Path(args.factory_file).read_bytes()).hexdigest()
        revision = 'factory-sha256:'+factory_sha
    else:
        import transformers
        clsname = args.auto_class or AUTO_CLASSES[args.hf_task]
        if not (clsname == 'AutoModel' or (clsname.startswith('AutoModelFor') and clsname[12:].isalnum())):
            raise ValueError('unknown Transformers AutoModel class')
        cls = getattr(transformers,clsname)
        custom = bool(getattr(args,'trust_remote_code',False))
        if streaming:
            from .hf_config_v83 import load_hf_config
            from .streaming_base import materialize_meta_from_safetensors
            # The HF model factory must support meta construction without
            # eagerly materializing its own base weights.
            cfg=load_hf_config(args.model,revision=args.revision,
                       trust_remote_code=custom,local_files_only=custom)
            from .hf_compat_v63 import validate_training_config
            validate_training_config(cfg, args.model, streaming=True)
            with torch.device('meta'):
                mdl=cls.from_config(cfg,trust_remote_code=custom)
            if Path(args.model).is_dir():
                snapshot_dir=args.model
            else:
                from huggingface_hub import snapshot_download
                snapshot_dir=snapshot_download(args.model,revision=args.revision,
                    allow_patterns=['*.safetensors','*.json'],local_files_only=custom)
            writable=(getattr(args,'cmd','') in ('train-live','train-elastic','train-p2p') and not lazy_write)
            materialize_meta_from_safetensors(mdl,snapshot_dir,writable=writable,
                preserve_checkpoint_dtype=True,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                chunk_mib=getattr(args,'stream_chunk_mib',8),strict=True,
                hash_genesis=getattr(args,'cmd','') in ('train-live','train-elastic'))
            print('STREAMING BASE:',json.dumps(mdl._fedpara_streaming_base_info),flush=True)
        else:
            from .hf_compat_v63 import validate_training_config
            # Give actionable errors for unsupported DeepSeek FP8/MoE families.
            auto_config=getattr(transformers,'AutoConfig',None)
            config = None
            if auto_config is not None:
                from .hf_config_v83 import load_hf_config
                config = load_hf_config(args.model,revision=args.revision,
                    trust_remote_code=custom,local_files_only=custom)
                validate_training_config(config, args.model, streaming=False)
            # v4 calls this torch_dtype; v5 calls it dtype. Avoid loading a
            # BF16 Llama checkpoint as FP32 (often twice the necessary RAM).
            version=str(getattr(transformers,'__version__',''))
            dtype_arg=({'dtype':'auto'} if version.startswith('5.') else
                       {'torch_dtype':'auto'} if version.startswith('4.') else {})
            mdl = cls.from_pretrained(args.model, revision=args.revision,
                        trust_remote_code=custom,
                        local_files_only=custom,
                        use_safetensors=not args.allow_legacy_bin,
                        **({'config':config} if config is not None else {}), **dtype_arg)
        revision = getattr(mdl.config,'_commit_hash',None) or args.revision
    device = resolve_device(args.device)
    if not streaming:
        mdl.to(device)
    if getattr(args, 'auto_rank', False) or getattr(args, 'resource_report', None):
        from .device_planner import suggest_rank_and_experts
        from .moe_planner import inspect_expert_shapes
        groups = inspect_expert_shapes(mdl)
        costs = [n for group in groups for n in group.bytes_per_expert]
        recommendation = suggest_rank_and_experts(mdl, device=str(device),
            disk_dir=getattr(args,'disk_linear_dir',None) or '.',kind=args.kind,
            expert_bytes=costs if costs else None)
        from dataclasses import asdict
        from .device_planner import automatic_expert_counts
        recommendation['expert_shapes'] = [asdict(g) for g in groups]
        if groups:
            try:
                recommendation['expert_defaults_by_group'] = automatic_expert_counts(
                    groups, max(0,recommendation['budget']['usable_bytes']//2))
            except MemoryError:
                recommendation['expert_defaults_by_group'] = None
                recommendation['expert_default_warning'] = 'Cannot fit one expert per group in budget'
        if getattr(args,'resource_report',None):
            Path(args.resource_report).write_text(json.dumps(recommendation,indent=2),encoding='utf-8')
        if getattr(args,'auto_rank',False):
            if getattr(args,'cmd','') != 'train':
                raise ValueError('--auto-rank is local-only: P2P peers must agree on the exact signed rank/layout')
            if recommendation['rank_default'] is None:
                raise MemoryError('no safe automatic rank; inspect the resource report')
            args.rank = recommendation['rank_default']
        print('RESOURCE PLAN',json.dumps({'rank_default':recommendation['rank_default'],
              'expert_count_default':recommendation['expert_count_default'],
              'source':recommendation['budget']['source']}),flush=True)
    if hasattr(mdl,'config') and hasattr(mdl.config,'use_cache'):
        mdl.config.use_cache = False
    if getattr(args,'gradient_checkpointing',False):
        from .training_compat_v84 import enable_adapter_safe_checkpointing
        enable_adapter_safe_checkpointing(mdl)
    model_id = args.model_id or args.model
    if getattr(args,'state_base',None) and getattr(args,'sharded_base_manifest',None):
        raise ValueError('--state-base and --sharded-base-manifest are mutually exclusive')
    if bool(getattr(args,'sharded_base_manifest',None)) != bool(getattr(args,'sharded_base_store',None)):
        raise ValueError('--sharded-base-manifest and --sharded-base-store must both be supplied')
    if getattr(args,'sharded_base_manifest',None):
        # A generation rollover manifest can be pinned to the original room
        # revision string, while Transformers resolves a branch name to a
        # concrete commit. The model data is independently SHA256 verified.
        from .shards import read_manifest
        base_info=read_manifest(Path(args.sharded_base_manifest),
                                max_state_bytes=args.max_state_gb*1024**3)
        if base_info['revision'] != revision:
            from .room_manifest import load_manifest
            pinned=load_manifest(Path(args.room_manifest)) if getattr(args,'room_manifest',None) else None
            if (not pinned or base_info['revision']!=args.revision or
                pinned['model']['revision']!=args.revision or
                pinned['model']['id']!=model_id):
                raise ValueError('sharded base revision differs from pinned model metadata')
            revision=base_info['revision']
        snapshot=apply_sharded_state(mdl,Path(args.sharded_base_manifest),
            Path(args.sharded_base_store),model_id=model_id,revision=revision,
            max_state_bytes=args.max_state_gb*1024**3)
        revision = 'fstate-sha256:'+semantic_shard_sha256(snapshot)
    if getattr(args,'state_base',None):
        state = Path(args.state_base)
        apply_state(mdl, state, model_id=model_id,revision=revision,
                    max_bytes=args.max_state_gb*1024**3)
        # The archive ZIP header contains timestamps and peer_id can vary.
        # Bind to tensor CONTENT, not incidental archive bytes.
        revision = 'fstate-sha256:'+semantic_state_sha256(state)
    if getattr(args,'moe_plan',None):
        from .moe_light import slice_moe_modulelist
        plan=json.loads(Path(args.moe_plan).read_text(encoding='utf-8'))
        if not isinstance(plan,list) or not 1 <= len(plan) <= 1024:
            raise ValueError('MoE plan must be a nonempty list of layer plans')
        for entry in plan:
            if not isinstance(entry,dict) or set(entry)!={'experts','gate','count','seed'}:
                raise ValueError('MoE entries require experts, gate, count, seed')
            sl=slice_moe_modulelist(mdl,entry['experts'],entry['gate'],
                                    count=entry['count'],seed=entry['seed'])
            print('MoE retained expert indices:',sl.path,sl.keep,flush=True)
    if not use_adapter:
        return mdl,revision
    resolved_scope=getattr(args,'target_scope','all')
    # Automatic target detection is architecture-neutral, not GPT-Neo only.
    # The adapter records each selected layer's kind/rank in the layout digest
    # so incompatible P2P updates are rejected before application.
    args.target_scope=resolved_scope
    adapter = UniversalAdapter(mdl, model_id, revision,
                               kind=args.kind,rank=args.rank,alpha=args.alpha,
                               max_peers=args.max_peers,target_scope=resolved_scope,
                               init_style=getattr(args,'init_style','balanced'),
                               wide_lora_rank=getattr(args,'wide_lora_rank',4))
    print(f'adapter policy scope={resolved_scope} init={getattr(args,"init_style","balanced")}',flush=True)
    check_adapter_placement(adapter)
    if getattr(args,'disk_linear_dir',None) or streaming:
        from .layer_offload import install_disk_linear_offload
        dest=(getattr(args,'disk_linear_dir',None) or
              (Path(getattr(args,'streaming_scratch_dir',None) or '.') / '.fedpara-linear-pages'))
        info=install_disk_linear_offload(mdl,adapter,dest,
            min_weight_mib=(getattr(args,'disk_linear_min_mib',0.) or (1. if streaming else 0.)),
            cache_fraction=getattr(args,'disk_cache_fraction',.6),
            cache_mib=getattr(args,'disk_cache_mib',0),
            tile_mib=getattr(args,'disk_tile_mib',16),include_embeddings=streaming,
            max_dir_bytes=(getattr(args,'live_disk_linear_max_gb',32)*1024**3
                           if getattr(args,'cmd','') in ('train-live','train-elastic')
                           and getattr(args,'disk_linear_dir',None) else None))
        print('DISK LINEAR OFFLOAD:',json.dumps(info),flush=True)
        if streaming and device.type!='cpu':
            from .layer_offload import place_streaming_compute_device
            offload_cache=next((getattr(module,'_fedpara_disk_cache')
                          for module in mdl.modules() if hasattr(module,'_fedpara_disk_cache')),None)
            placement=place_streaming_compute_device(mdl,adapter,device,cache=offload_cache)
            object.__setattr__(mdl,'_fedpara_compute_device',str(device))
            print('STREAMING ACCELERATOR PLACEMENT:',json.dumps(placement),flush=True)
            # Recheck factor placements after custom CPU/GPU dispatch.
            check_adapter_placement(adapter)
    if lazy_write:
        if factor_ledger:
            from .factor_ledger_v62 import attach_factor_ledger
            manager=attach_factor_ledger(mdl,adapter,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                max_entries=getattr(args,'stream_factor_max_events',32),
                limit_gib=getattr(args,'stream_lazy_limit_gb',0))
        elif page_cow:
            from .paged_mutable_v61 import attach_paged_mutable
            manager=attach_paged_mutable(mdl,adapter,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                page_kib=getattr(args,'stream_page_kib',64),
                limit_gib=getattr(args,'stream_lazy_limit_gb',0))
        else:
            from .lazy_mutable_weights_v60 import attach_lazy_mutable
            manager=attach_lazy_mutable(mdl,adapter,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                copy_mib=getattr(args,'stream_chunk_mib',8),
                limit_gib=getattr(args,'stream_lazy_limit_gb',0))
        print('LAZY STREAM MUTABLE:',json.dumps(manager.stats()),flush=True)
    expert_path=getattr(args,'streaming_expert_path',None)
    if expert_path:
        if not streaming:
            raise ValueError('--streaming-expert-path requires --streaming-base')
        from .expert_disk import install_streamed_experts
        count=install_streamed_experts(mdl,expert_path)
        print('STREAMED FROZEN EXPERTS:',count,expert_path,flush=True)
    print('adapter:',json.dumps({**adapter.counts(),'layout':adapter.layout,
                                   'resolved_revision':revision},indent=2))
    return mdl,adapter

