"""Conservative, deterministic training defaults. No untrusted code execution.

These are memory/network *estimates*, not a claim of OOM-freedom. Network
membership remains an observed live quantity, never a user-set peer target.
"""
from __future__ import annotations

import math

_POSITION_KEYS = ('max_position_embeddings', 'n_positions', 'n_ctx',
                  'max_seq_len', 'max_sequence_length', 'seq_length',
                  'model_max_length')


def estimate_training_policy(*, config: dict | None, available_ram_bytes: int = 0,
                             available_vram_bytes: int = 0, rank: int = 16,
                             kind: str = 'loha') -> dict:
    if type(rank) is not int or not 1 <= rank <= 256 or kind not in ('loha', 'lora'):
        raise ValueError('invalid adapter configuration')
    cfg = config if isinstance(config, dict) else {}
    bounds = [int(cfg[key]) for key in _POSITION_KEYS
              if type(cfg.get(key)) is int and 16 <= cfg[key] <= 2**20]
    hard_max = min(bounds) if bounds else 1024  # unknown => application safety ceiling
    h = int(cfg.get('hidden_size') or cfg.get('n_embd') or cfg.get('d_model') or 768)
    depth = int(cfg.get('num_hidden_layers') or cfg.get('n_layer') or
                cfg.get('num_layers') or cfg.get('num_hidden_layers') or 12)
    h = max(32, min(131072, h))
    depth = max(1, min(512, depth))
    vram = max(0, int(available_vram_bytes or 0))
    ram = max(0, int(available_ram_bytes or 0))
    # User asks to use available memory, not arbitrarily cap to a tiny context.
    # Reserve base model/optimizer/allocator headroom; paging is not free.
    usable = int((vram * .24) if vram >= 768*1024**2 else (ram * .10))
    if usable <= 0:
        usable = 384*1024**2
    # Approximate activation checkpointing budget for batch size 1-2.
    bytes_per_token = max(16384, h * depth * 44)
    fitting = min(hard_max, max(32, usable // bytes_per_token))
    # Avoid odd sizes; no exception for non-power-of-two architecture limits.
    length = max(32, 1 << (int(fitting).bit_length()-1))
    length = min(length, hard_max, 8192)
    # LoRA/LoHa delta proxy: attention + feedforward projections and adapter factors.
    # Per-event wire is usually larger than a single rank factor; plan cautiously.
    estimate_bytes = h * depth * rank * 7 * 2 * 2 * (2 if kind == 'loha' else 1)
    mib = estimate_bytes / (1024*1024)
    if mib < 8: baseline = 4
    elif mib < 16: baseline = 8
    elif mib < 32: baseline = 16
    elif mib < 64: baseline = 32
    else: baseline = 64
    return {'max_length':int(length), 'local_steps':int(baseline),
            'model_context_limit':int(hard_max),
            'estimated_adapter_wire_mib':round(mib,2),
            'model_metadata_available':bool(cfg),
            'online_peers':None,
            'reason':('Context is a device-memory/architecture estimate; '
                      'publish interval also adapts to observed peer count. '
                      'Live topology is measured after connecting.')}


def estimate_wan_publish_cadence(*, config: dict | None, adapter_wire_mib: float = 0.,
                                 default_uplink_mib_s: float = 4.) -> dict:
    """Local advisory publication interval, monotone in model and adapter size.

    This is a WAN *starting point*, not a bandwidth measurement or a claim
    of optimizer convergence. The signed room's 20-event ReLoRA schedule stays
    untouched. Peer-specific hardware/bandwidth never enters the manifest.
    """
    cfg = config if isinstance(config,dict) else {}
    def positive_int(*keys, fallback=0):
        for key in keys:
            value=cfg.get(key)
            if type(value) is int and 0 < value <= 10**9:
                return value
        return fallback
    # Actual numbers (when config declares them) beat rough shape estimates.
    params=positive_int('num_parameters','n_parameters','parameter_count')
    if not params:
        h=positive_int('hidden_size','n_embd','d_model','dim')
        layers=positive_int('num_hidden_layers','n_layer','num_layers','n_layers')
        if h and layers:
            intermediate=positive_int('intermediate_size','n_inner','ffn_dim',fallback=4*h)
            vocab=positive_int('vocab_size',fallback=0)
            # Transformer attention QKV+output, SwiGLU MLP, embeddings;
            # rough overestimate for some architectures and an underestimate
            # for MoE models (which should provide num_parameters).
            params=min(10**13, layers*(4*h*h+3*h*intermediate)+vocab*h)
    tiers=((500_000_000,32),(1_500_000_000,64),(6_000_000_000,128),
           (20_000_000_000,256),(60_000_000_000,512))
    steps=32
    if params:
        steps=next((rate for limit,rate in tiers if params<=limit),1024)
    try:
        wire=float(adapter_wire_mib)
        uplink=float(default_uplink_mib_s)
    except (TypeError,ValueError):
        wire=0.; uplink=4.
    if not (math.isfinite(wire) and wire>=0):wire=0.
    if not (math.isfinite(uplink) and 0.125<=uplink<=1024):uplink=4.
    # Large signed deltas must not be produced at tiny time intervals.
    # The proxy does not know real compression, uplink or instantaneous RTT.
    transfer_seconds=wire/uplink
    if transfer_seconds>240:steps=max(steps,512)
    elif transfer_seconds>120:steps=max(steps,256)
    elif transfer_seconds>60:steps=max(steps,128)
    elif transfer_seconds>30:steps=max(steps,64)
    return {'local_steps':steps,'estimated_parameters':params or None,
            'estimated_transfer_seconds':round(transfer_seconds,1),
            'assumed_uplink_mib_s':uplink,
            'reason':('WAN publication interval increases with the model parameter '
                      'count and estimated signed update transfer cost. '
                      'This does not alter the room-wide ReLoRA refresh.')}


# The canonical target layout and MAXIMUM rank are shared and immutable.
# Each upgraded peer independently chooses its physical LoHA rank below the cap.
RANK_CHOICES = (4, 8, 16, 32, 64)


def estimate_adapter_rank(*, config: dict | None, available_ram_bytes: int = 0,
                          available_vram_bytes: int = 0, kind: str = 'loha') -> dict:
    """Advisory capacity + network-conscious genesis rank, using metadata only.

    Deliberately uses the same worst-case proxy as context/publication planning.
    Never treats disk space as fast working memory. Unknown architecture receives
    the legacy conservative rank rather than an unjustified high-rank estimate.
    """
    if kind not in ('loha', 'lora'):
        raise ValueError('unknown adapter kind')
    cfg = config if isinstance(config, dict) else {}
    h = int(cfg.get('hidden_size') or cfg.get('n_embd') or cfg.get('d_model') or 768)
    depth = int(cfg.get('num_hidden_layers') or cfg.get('n_layer') or cfg.get('num_layers') or 12)
    h = max(32, min(131072, h))
    depth = max(1, min(512, depth))
    ram = max(0, int(available_ram_bytes or 0))
    vram = max(0, int(available_vram_bytes or 0))
    # Use accelerator capacity when one is present. CPU/MPS uses system RAM;
    # this is an upper bound, not proof that an OOM will not happen.
    working = vram if vram else ram
    def estimate(rank):
        # 7 projections, 2 matrices each, LoHa has two factor pairs.
        factors = h * depth * rank * 7 * 2 * 4 * (2 if kind == 'loha' else 1)
        return factors * 4  # factors + grads + Adam moments
    mem_limit = int(working * .15) if working else 0
    feasible = [rank for rank in RANK_CHOICES if not working or estimate(rank) <= mem_limit]
    max_safe = (max(feasible) if feasible else 0) if working else None
    if not cfg:
        chosen = 16 if 16 in feasible else (max(feasible) if feasible else 4)
    else:
        # Max ~32 MiB/commit where reasonable; the continual-gossip wire
        # traffic grows with rank even if a large GPU could fit rank=64.
        wire_ceiling = 32 * 1024**2
        recommended = [rank for rank in feasible if estimate(rank) // 4 <= wire_ceiling]
        chosen = max(recommended) if recommended else (min(feasible) if feasible else 4)
    return {'rank':chosen,'rank_capacity':max_safe,
            'memory_estimate_bytes':estimate(chosen),
            'wire_estimate_bytes':estimate(chosen) // 4,
            'metadata_available':bool(cfg),
            'warning':'Metadata-only estimation. Real peaks and bandwidth may differ; room rank is immutable.'}
