"""Physical per-peer LoHA factor rank selection (NO rank masking or zero padding).

The room-wide rank remains a canonical upper bound and layout identity.  The
local journal's optimizer snapshot MUST use a stable physical shape.  Pin
chosen rank before allocating the adapter, and preserve full rank for legacy
journals which have never recorded a heterogeneous rank.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from .device_planner import detect_device_budget, estimate_adapter_memory


def choose_physical_rank(model, args, *, device='cpu', budget=None):
    maximum = args.rank
    request = str(getattr(args,'local_loha_rank','auto'))
    if not isinstance(maximum,int) or maximum < 1 or maximum > 256:
        raise ValueError('room rank out of range')
    if request == 'full':
        rank = maximum
        reason = 'explicit full physical rank'
    elif request != 'auto':
        rank = int(request)
        if not 1 <= rank <= maximum:
            raise ValueError('physical rank must be within signed room maximum')
        reason = 'explicit per-peer physical rank'
    else:
        if budget is None:
            budget = detect_device_budget(device=device,memory_fraction=.5)
        usable = budget.usable_bytes
        # Keep at least 7/8 of currently usable memory for model activations,
        # inference buffers, Python, CUDA workspace, snapshot storage, etc.
        # A conservative estimate, NOT a proof OOM cannot occur.
        budget_bytes = usable // 8
        options = sorted(set([1,2,4,8,16,32,64,128,256,maximum]))
        options = [r for r in options if r<=maximum]
        estimate = {r: estimate_adapter_memory(model,kind='loha',rank=r)['factor_grad_optimizer_bytes']
                    for r in options}
        # CPU adapts more conservatively than GPU even on RAM-rich hosts:
        # an ample DRAM budget is not evidence of CUDA-class throughput.
        accelerator_cap = (8 if str(device).startswith('cpu') else
                           16 if str(device).startswith('mps') else maximum)
        viable = [r for r in options if r <= accelerator_cap and estimate[r] <= budget_bytes]
        rank = max(viable) if viable else 1
        reason = (f'automatic device {device} cap {accelerator_cap}; '
                  f'free {usable / (1024**3):.2f} GiB; '
                  f'allocated factor+grad+AdamW estimate {estimate[rank] / (1024**2):.2f} MiB')
    journal = getattr(args,'journal',None)
    pinned_path = Path(str(journal)+'.physical-rank.json') if journal else None
    contract = {'version':1,'kind':args.kind,'max_rank':maximum,
                'model':getattr(args,'model_id',None) or args.model,
                'revision':args.revision,
                'scope':getattr(args,'target_scope','auto'),
                'wide_lora_rank':getattr(args,'wide_lora_rank',4)}
    if pinned_path is not None:
        if pinned_path.is_file():
            record = json.loads(pinned_path.read_text(encoding='utf8'))
            if not isinstance(record,dict) or any(record.get(k)!=v for k,v in contract.items()):
                raise ValueError('physical rank pin does not match this room/model configuration')
            pinned=record.get('physical_rank')
            if type(pinned) is not int or not 1<=pinned<=maximum:
                raise ValueError('invalid local physical rank pin')
            if request not in ('auto','full') and int(request)!=pinned:
                raise ValueError('cannot resize live LoHA factors: use a new local peer/journal after sealing current updates')
            if request=='full' and pinned!=maximum:
                raise ValueError('cannot resize pinned physical LoHA rank: start a new peer')
            rank=pinned
            reason='restored pinned rank for signed journal and optimizer'
        else:
            # Old journals predate this rank protocol. They may contain an
            # arbitrary full-rank unsealed factor/optimizer checkpoint. Never
            # silently truncate or overwrite that working state.
            old_journal=Path(journal)
            old_working=Path(str(journal)+'.working.safetensors')
            if old_journal.is_file() and old_journal.stat().st_size>0 or old_working.is_file():
                if request not in ('auto','full',str(maximum)):
                    raise ValueError('legacy journal is full rank: seal/retire it before selecting a smaller rank')
                rank=maximum
                reason='legacy journal retained at full physical rank'
            # Persist the retained legacy rank, too. If the journal is
            # archived later the peer must not silently change factor shapes.
            pinned_path.parent.mkdir(parents=True,exist_ok=True)
            # Avoid overwriting an existing pin when two processes start
            # concurrently. The signed event journal also enforces its
            # exclusive train.lock.
            record={**contract,'physical_rank':rank}
            try:
                fd=os.open(str(pinned_path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            except FileExistsError:
                return choose_physical_rank(model,args,device=device,budget=budget)
            with os.fdopen(fd,'w',encoding='utf8') as f:
                json.dump(record,f,sort_keys=True)
                f.flush();os.fsync(f.fileno())
    return {'physical_rank':rank,'room_max_rank':maximum,'reason':reason,
            'wire_rank':rank,'optimizer_rank':rank,'heterogeneous_protocol':True}
