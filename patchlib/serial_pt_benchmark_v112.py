"""Single-resident-model PT peer-count comparison for memory-constrained hosts.

This is an *offline*, serial, synchronous FedSum experiment using Kagami's
UniversalAdapter, continual factor anchors and actual LM gradients. It does
NOT run WebRTC, signed-event transport or independent concurrent processes.

Each peer has the same number of local PT steps across arms; total compute
and total tokens grow linearly with the number of peers. Peer 0's exact
minibatches match across arms. Virtual network time includes model broadcast,
upload, bandwidth queues, jitter, loss/retry and synchronous barrier waiting.
Only one model/optimizer is resident. Per-peer factor and optimizer state lives
on disk and is loaded into that same object before its scheduled local steps.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .continual import enable_anchors
from .core import UniversalAdapter
from .effectiveness_benchmark import TinyCausalLM, _batch, _data, _local_text_data, _logits
from .serial_pt_network_v1122 import NetworkConfig, StarNetworkSimulator


@dataclass(frozen=True)
class SerialPTConfig:
    peers: tuple[int, ...] = (1, 2, 4)
    rounds: int = 8
    steps_per_round: int = 8
    batch_size: int = 4
    context: int = 16
    width: int = 32
    rank: int = 4
    lr: float = 0.003
    train_sequences: int = 512
    valid_sequences: int = 96
    seed: int = 20261107
    threads: int = 1
    device: str = 'cpu'
    kind: str = 'loha'
    text_file: str | None = None
    scratch_dir: str | None = None
    eval_batch_size: int = 16
    rtt_ms: float = 50.
    jitter_ms: float = 5.
    bandwidth_mbps: float = 50.
    server_bandwidth_mbps: float = 100.
    payload_bytes: int | None = None
    message_overhead_bytes: int = 1024
    loss_rate: float = 0.
    retry_rto_ms: float = 250.
    max_retries: int = 3
    compute_ms_per_step: float | None = None
    aggregation_ms: float | None = None

    def network_config(self):
        return NetworkConfig(rtt_ms=self.rtt_ms, jitter_ms=self.jitter_ms,
                             bandwidth_mbps=self.bandwidth_mbps,
                             server_bandwidth_mbps=self.server_bandwidth_mbps,
                             payload_bytes=self.payload_bytes,
                             message_overhead_bytes=self.message_overhead_bytes,
                             loss_rate=self.loss_rate, retry_rto_ms=self.retry_rto_ms,
                             max_retries=self.max_retries, seed=self.seed)

    def validate(self):
        from .mps_policy_v92 import check_device
        check_device(self.device)
        if (not self.peers or self.peers[0] != 1 or
                len(set(self.peers)) != len(self.peers) or
                any(type(p) is not int or p < 1 or p > 32 for p in self.peers)):
            raise ValueError('peer counts must be unique positive integers (<=32), beginning with 1')
        if not (1 <= self.rounds <= 10000 and 1 <= self.steps_per_round <= 1024 and
                1 <= self.batch_size <= 256 and 2 <= self.context <= 256 and
                8 <= self.width <= 256 and 1 <= self.rank <= min(self.width, 32) and
                0 < self.lr <= .1 and 8 <= self.train_sequences <= 1_000_000 and
                8 <= self.valid_sequences <= 100_000 and 1 <= self.eval_batch_size <= 256 and
                1 <= self.threads <= 16):
            raise ValueError('invalid model/data/training settings for bounded offline benchmark')
        self.network_config().validate()
        if self.compute_ms_per_step is not None and (not 0 <= self.compute_ms_per_step <= 1e7):
            raise ValueError('compute-ms-per-step must be in [0, 1e7]')
        if self.aggregation_ms is not None and (not 0 <= self.aggregation_ms <= 1e7):
            raise ValueError('aggregation-ms must be in [0, 1e7]')
        if self.kind not in ('loha', 'lora'):
            raise ValueError('kind must be loha or lora')
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('device must be cpu/mps/cuda')
        if self.device == 'mps' and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            raise RuntimeError('MPS is not available; run --device cpu instead')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        if self.text_file is not None and not Path(self.text_file).expanduser().is_file():
            raise FileNotFoundError(self.text_file)
        if self.scratch_dir is not None and not Path(self.scratch_dir).expanduser().is_dir():
            raise NotADirectoryError('scratch directory must already exist')
        corpus_bytes = (self.train_sequences + self.valid_sequences) * (self.context + 1) * 8
        if corpus_bytes > 128 * 1024 * 1024:
            raise ValueError('data allocation would exceed 128MiB; lower sequences or context')


def _original(model, group):
    modname, _, pname = group.aliases[0].rpartition('.')
    return model.get_submodule(modname).parametrizations[pname].original


def _synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def _fingerprint_batch_schedule(count, cfg: SerialPTConfig):
    """Fingerprint peer-0 schedule; invariant across arms."""
    h = hashlib.sha256()
    for round_no in range(cfg.rounds):
        for slot in range(cfg.steps_per_round):
            offset = (round_no * cfg.steps_per_round + slot) * cfg.batch_size
            # Validation uses the exact per-peer batch bytes for a stronger check.
            h.update(f'{offset}:{count}:peer-0\n'.encode('ascii'))
    return h.hexdigest()


def _evaluation(model, heldout, batch_size, device):
    """Weighted heldout CE/accuracy, not training-batch accuracy."""
    model.eval()
    total_loss, hits, tokens = 0.0, 0, 0
    try:
        with torch.inference_mode():
            for chunk in heldout.split(batch_size):
                x = chunk[:, :-1].to(device)
                y = chunk[:, 1:].to(device)
                logits = _logits(model(x))
                total_loss += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                              y.reshape(-1), reduction='sum').item()
                hits += (logits.argmax(dim=-1) == y).sum().item()
                tokens += y.numel()
        _synchronize(device)
    finally:
        model.train()
    return {'valid_ce': total_loss / tokens, 'valid_top1': hits / tokens}


def _train_one(model, optimizer, sample, device):
    x, y = sample
    optimizer.zero_grad(set_to_none=True)
    logits = _logits(model(x.to(device)))
    target = y.to(device)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    if not torch.isfinite(loss).item():
        raise FloatingPointError('nonfinite PT training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0,
                                   error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def _cpu_copy(item):
    if isinstance(item, torch.Tensor):
        return item.detach().to('cpu').clone()
    if isinstance(item, dict):
        return {key: _cpu_copy(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_cpu_copy(value) for value in item]
    if isinstance(item, tuple):
        return tuple(_cpu_copy(value) for value in item)
    return item


def _write_peer(path, adapter, optimizer, step_count):
    """Atomic disk write; no other peer optimizer state stays alive in RAM."""
    factors = []
    for group in adapter.groups:
        for i, p in enumerate(group.overlay.local.parts):
            if not torch.equal(p.detach(), getattr(group.overlay.anchor, f'part{i}')):
                raise RuntimeError('attempted to save an uncommitted factor state')
            factors.append(_cpu_copy(p))
    state = {'layout': adapter.layout, 'factors': factors,
             'optimizer': _cpu_copy(optimizer.state_dict()), 'steps': int(step_count)}
    temp_path = path.with_suffix('.new')
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    del state


def _load_peer(path, adapter, optimizer):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['layout'] != adapter.layout:
        raise RuntimeError('peer factor schema mismatch')
    expected = sum(len(g.overlay.local.parts) for g in adapter.groups)
    if len(payload['factors']) != expected:
        raise RuntimeError('peer factor count mismatch')
    with torch.no_grad():
        idx = 0
        for group in adapter.groups:
            for j, param in enumerate(group.overlay.local.parts):
                factor = payload['factors'][idx]
                if factor.shape != param.shape:
                    raise RuntimeError('peer factor shape mismatch')
                param.copy_(factor.to(device=param.device, dtype=param.dtype))
                getattr(group.overlay.anchor, f'part{j}').copy_(param.detach())
                param.grad = None
                idx += 1
    optimizer.load_state_dict(payload['optimizer'])
    # torch Optimizer.load_state_dict casts states onto the live parameter device.
    return int(payload['steps'])


class _FedSumRound:
    """Exact once-per-peer *effective* delta SUM (no averaging of factors)."""
    def __init__(self, model, adapter):
        self.totals = {g.canonical: torch.zeros_like(_original(model, g), device='cpu', dtype=torch.float32)
                       for g in adapter.groups}
        self.seen: set[int] = set()

    def add(self, peer_id, changes):
        if peer_id in self.seen:
            return False
        if changes.keys() != self.totals.keys():
            raise ValueError('FedSum inventory mismatch')
        for key, delta in changes.items():
            target = self.totals[key]
            if target.shape != delta.shape or not torch.isfinite(delta).all():
                raise ValueError(f'invalid FedSum effective update for {key}')
        for key, delta in changes.items():
            self.totals[key].add_(delta)
        self.seen.add(peer_id)
        return True

    def apply(self, model, adapter, expected_peers):
        if len(self.seen) != expected_peers:
            raise RuntimeError('FedSum barrier missing peer updates')
        with torch.no_grad():
            for group in adapter.groups:
                original = _original(model, group)
                value = original.float() + self.totals[group.canonical].to(original.device)
                if not torch.isfinite(value).all():
                    raise FloatingPointError('nonfinite committed base weights')
                original.copy_(value.to(original.dtype))


def _seal_effective_delta(adapter):
    """Subtract previous anchor; no double counting across rounds."""
    changes = {}
    with adapter.lock, torch.no_grad():
        for group in adapter.groups:
            current = group.overlay.local
            anchor = group.overlay.anchor
            change = (current.delta().float() - anchor.delta().float()).detach().to('cpu')
            if not torch.isfinite(change).all():
                raise FloatingPointError('nonfinite LoHA effective delta')
            changes[group.canonical] = change
            for i, param in enumerate(current.parts):
                getattr(anchor, f'part{i}').copy_(param.detach())
    return changes


def _get_data(cfg):
    if cfg.text_file:
        train, valid, info = _local_text_data(Path(cfg.text_file), 1, cfg.train_sequences,
                                             cfg.context, cfg.seed, cfg.valid_sequences)
        return train[0], valid, 256, info
    train, valid = _data(1, cfg.train_sequences, cfg.context, 32, cfg.seed, cfg.valid_sequences)
    return train[0], valid, 32, {'type': 'deterministic-synthetic-markov-next-token',
                                 'heldout_disjoint': True, 'note': 'Smoke test, not natural-language benchmark'}


def _save_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(outdir, traces):
    """Quality vs simulated parallel wall time; show compute-only and token axes separately."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    metrics = [('valid_ce', 'Heldout cross-entropy (nats)', 'loss'),
               ('valid_top1', 'Heldout next-token Top-1 accuracy', 'accuracy')]
    axes = [
        ('virtual_time_s', 'Simulated parallel time (seconds; network + barrier)', ''),
        ('critical_compute_s', 'Parallel critical-path compute only (seconds)', '-ideal-compute'),
        ('steps_per_peer', 'Local PT optimizer steps per peer', '-local-steps'),
        ('tokens_seen', 'Aggregate training tokens (increases with peers)', '-total-tokens'),
    ]
    for key, ylabel, short in metrics:
        for x_key, xlabel, suffix in axes:
            fig, ax = plt.subplots(figsize=(9, 5))
            for peers in sorted({row['peers'] for row in traces}):
                subset = [r for r in traces if r['peers'] == peers]
                ax.plot([r[x_key] for r in subset], [r[key] for r in subset],
                        label=f'{peers} peer' + ('s' if peers > 1 else ''),
                        marker='.', markersize=3)
            ax.set(xlabel=xlabel, ylabel=ylabel,
                   title='Serial PT: synchronous FedSum / virtual star network (NOT live P2P)')
            ax.grid(alpha=.25)
            ax.legend()
            if x_key == 'virtual_time_s':
                fig.text(.5, .005, 'Virtual time using measured per-peer compute; not real concurrent wall clock',
                         ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .035, 1, 1))
            fig.savefig(outdir / f'pt-heldout-{short}{suffix}.png', dpi=140)
            plt.close(fig)
    return True


def _resolve_scratch_dir(cfg: SerialPTConfig):
    return str(Path(cfg.scratch_dir).expanduser()) if cfg.scratch_dir is not None else None


def run_serial_pt_benchmark(cfg: SerialPTConfig, output_dir: str | Path):
    """Execute each full arm and each virtual peer sequentially, never concurrently."""
    cfg.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(cfg.threads)
    try:
        with tempfile.TemporaryDirectory(prefix='kagami-v112-pt-',
                                             dir=_resolve_scratch_dir(cfg)) as temp_root:
            state_dir = Path(temp_root)
            train, validation, vocab, data_info = _get_data(cfg)
            # Same RNG seed and genesis across arms; each peer has a stable local PT sample stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed)
                model = TinyCausalLM(vocab=vocab, width=cfg.width, context=cfg.context)
                adapter = UniversalAdapter(model, 'serial-pt-v112', 'frozen-genesis',
                                           kind=cfg.kind, rank=cfg.rank, target_scope='linear')
                enable_anchors(adapter)
            trainable = {id(p) for p in adapter.trainable_parameters()}
            for p in model.parameters():
                if id(p) not in trainable:
                    p.requires_grad_(False)
            device = torch.device(cfg.device)
            model.to(device)
            optimizer = torch.optim.AdamW(list(adapter.trainable_parameters()), lr=cfg.lr)

            initial_base = state_dir / 'initial-bases.pt'
            torch.save([_cpu_copy(_original(model, group)) for group in adapter.groups], initial_base)
            initial_peer = state_dir / 'initial-peer.pt'
            _write_peer(initial_peer, adapter, optimizer, 0)
            base_digest = hashlib.sha256(initial_base.read_bytes()).hexdigest()
            data_digest = hashlib.sha256(train.numpy().tobytes() + validation.numpy().tobytes()).hexdigest()
            schedule_digest = _fingerprint_batch_schedule(len(train), cfg)
            rows, summary = [], []
            tokens_per_peer_expected = cfg.rounds * cfg.steps_per_round * cfg.batch_size * cfg.context
            payload_bytes = sum(t.numel() * t.element_size() for t in _FedSumRound(model, adapter).totals.values())
            resident_models = 1

            def restore_base():
                originals = torch.load(initial_base, map_location='cpu', weights_only=True)
                with torch.no_grad():
                    for group, weight in zip(adapter.groups, originals, strict=True):
                        dest = _original(model, group)
                        dest.copy_(weight.to(device=dest.device, dtype=dest.dtype))
                del originals

            for count in cfg.peers:
                restore_base()
                peers_dir = state_dir / f'peers-{count}'
                peers_dir.mkdir()
                for i in range(count):
                    (peers_dir / f'{i}.pt').write_bytes(initial_peer.read_bytes())
                arm_start = time.perf_counter()
                network = StarNetworkSimulator(cfg.network_config())
                optim_s, delta_s, aggregate_s, IO_s, eval_s = 0., 0., 0., 0., 0.
                processed = 0
                sample_hash = hashlib.sha256()
                peer0_hash = hashlib.sha256()
                start_e = time.perf_counter()
                baseline = _evaluation(model, validation, cfg.eval_batch_size, device)
                eval_s += time.perf_counter() - start_e
                rows.append({'peers': count, 'round': 0, 'steps': 0, 'steps_per_peer': 0,
                             'tokens_seen': 0, 'tokens_per_peer': 0,
                             'train_ce': float('nan'), **baseline,
                             'virtual_time_s': 0., 'network_sync_s': 0.,
                             'critical_compute_s': 0., 'barrier_wait_s': 0.,
                             'simulated_messages': 0, 'simulated_retries': 0,
                             'simulated_drops': 0, 'simulated_wire_bytes': 0,
                             'serial_wall_s': 0., 'optimizer_compute_s': 0., 'delta_compute_s': 0.,
                             'aggregation_compute_s': 0.,
                             'state_io_s': 0., 'evaluation_s': 0.})
                print(f'[PT {count}p] 0/{cfg.rounds}: valid_ce={baseline["valid_ce"]:.4f} '
                      f'valid_acc={baseline["valid_top1"]:.4f}', flush=True)
                for round_no in range(cfg.rounds):
                    accumulator = _FedSumRound(model, adapter)
                    loss_sum = 0.
                    per_peer_steps = cfg.steps_per_round
                    peer_compute_seconds = []
                    for i in range(count):
                        state_path = peers_dir / f'{i}.pt'
                        start_io = time.perf_counter()
                        before_steps = _load_peer(state_path, adapter, optimizer)
                        IO_s += time.perf_counter() - start_io
                        # Each peer keeps its own stable sampling stream. Peer 0 sees
                        # exactly the same sequence in the 1p, 2p, ... arms.
                        local_compute_seconds = 0.
                        for local_slot in range(per_peer_steps):
                            batch = _batch(train, round_no, local_slot, cfg.batch_size,
                                           cfg.seed, i, cfg.steps_per_round)
                            sample_hash.update(batch[0].numpy().tobytes())
                            sample_hash.update(batch[1].numpy().tobytes())
                            if i == 0:
                                peer0_hash.update(batch[0].numpy().tobytes())
                                peer0_hash.update(batch[1].numpy().tobytes())
                            _synchronize(device)
                            train_begin = time.perf_counter()
                            loss = _train_one(model, optimizer, batch, device)
                            _synchronize(device)
                            step_compute = time.perf_counter() - train_begin
                            optim_s += step_compute
                            local_compute_seconds += (cfg.compute_ms_per_step / 1000. if
                                                      cfg.compute_ms_per_step is not None else step_compute)
                            loss_sum += loss
                            processed += cfg.batch_size * cfg.context
                        delta_begin = time.perf_counter()
                        changes = _seal_effective_delta(adapter)
                        assert accumulator.add(i, changes)
                        del changes
                        _synchronize(device)
                        delta_elapsed = time.perf_counter() - delta_begin
                        delta_s += delta_elapsed
                        if cfg.compute_ms_per_step is None:
                            local_compute_seconds += delta_elapsed
                        peer_compute_seconds.append(local_compute_seconds)
                        start_io = time.perf_counter()
                        _write_peer(state_path, adapter, optimizer, before_steps + per_peer_steps)
                        IO_s += time.perf_counter() - start_io
                    aggregate_started = time.perf_counter()
                    accumulator.apply(model, adapter, count)
                    _synchronize(device)
                    aggregation_seconds = time.perf_counter() - aggregate_started
                    aggregate_s += aggregation_seconds
                    # An explicitly fixed compute speed should yield repeatable
                    # virtual timestamps, independent of host scheduling noise.
                    virtual_aggregation = (cfg.aggregation_ms / 1000. if cfg.aggregation_ms is not None
                        else 0. if cfg.compute_ms_per_step is not None else aggregation_seconds)
                    network.round(peer_compute_seconds, payload_bytes,
                                  aggregation_seconds=virtual_aggregation)
                    if len(accumulator.seen) != count:
                        raise AssertionError('not all peer updates applied')
                    del accumulator
                    start_e = time.perf_counter()
                    metrics = _evaluation(model, validation, cfg.eval_batch_size, device)
                    eval_s += time.perf_counter() - start_e
                    row = {'peers': count, 'round': round_no + 1,
                           'steps': (round_no + 1) * cfg.steps_per_round * count,
                           'steps_per_peer': (round_no + 1) * per_peer_steps,
                           'tokens_seen': processed, 'tokens_per_peer': processed // count,
                           'train_ce': loss_sum / (cfg.steps_per_round * count), **metrics,
                           'virtual_time_s': network.now_s,
                           'network_sync_s': network.total_network_sync_s,
                           'critical_compute_s': network.total_compute_critical_s,
                           'barrier_wait_s': network.total_wait_s,
                           'simulated_messages': network.total_messages,
                           'simulated_retries': network.total_retries,
                           'simulated_drops': network.total_drops,
                           'simulated_wire_bytes': network.total_wire_bytes,
                           'serial_wall_s': time.perf_counter() - arm_start,
                           'optimizer_compute_s': optim_s, 'delta_compute_s': delta_s,
                           'aggregation_compute_s': aggregate_s,
                           'state_io_s': IO_s,
                           'evaluation_s': eval_s}
                    rows.append(row)
                    print(f'[PT {count}p] {round_no+1}/{cfg.rounds}: '
                          f'tokens={processed} valid_ce={metrics["valid_ce"]:.4f} '
                          f'valid_acc={metrics["valid_top1"]:.4f}', flush=True)
                if processed != tokens_per_peer_expected * count:
                    raise AssertionError('training tokens do not scale by peer count')
                summary.append({'peers': count, 'valid_ce': rows[-1]['valid_ce'],
                                'valid_top1': rows[-1]['valid_top1'],
                                'total_tokens': processed, 'total_steps': cfg.rounds * cfg.steps_per_round * count,
                                'tokens_per_peer': processed // count,
                                'steps_per_peer': cfg.rounds * cfg.steps_per_round,
                                'virtual_time_s': network.now_s,
                                'simulated_tokens_per_s': processed / network.now_s if network.now_s else 0.,
                                'network_sync_s': network.total_network_sync_s,
                                'critical_compute_s': network.total_compute_critical_s,
                                'barrier_wait_s': network.total_wait_s,
                                'simulated_messages': network.total_messages,
                                'simulated_retries': network.total_retries,
                                'simulated_drops': network.total_drops,
                                'simulated_wire_bytes': network.total_wire_bytes,
                                'serial_wall_s': time.perf_counter() - arm_start,
                                'optimizer_compute_s': optim_s, 'delta_compute_s': delta_s,
                                'aggregation_compute_s': aggregate_s,
                                'state_io_s': IO_s,
                                'evaluation_s': eval_s,
                                'sample_schedule_sha256': sample_hash.hexdigest(),
                                'peer0_samples_sha256': peer0_hash.hexdigest(),
                                'initial_base_sha256': base_digest})
                # Drop old peer files immediately to minimize disk pressure.
                for peer_file in peers_dir.iterdir():
                    peer_file.unlink()
                peers_dir.rmdir()
                gc.collect()
            assert len({r['peer0_samples_sha256'] for r in summary}) == 1, 'peer 0 training stream diverged'
            assert len({r['initial_base_sha256'] for r in summary}) == 1, 'initial base diverged'

        _save_rows(output_dir / 'trace.csv', rows)
        _save_rows(output_dir / 'summary.csv', summary)
        plots_available = _plot(output_dir, rows)
        report = {
            'kind': 'OFFLINE_SERIAL_SYNCHRONOUS_FEDSUM_PT_WITH_VIRTUAL_NETWORK',
            'configuration': asdict(cfg),
            'data': data_info, 'data_sha256': data_digest,
            'schedule_sha256': schedule_digest,
            'model': {'name': 'TinyCausalLM', 'vocab': vocab, 'target_scope': 'linear',
                      'adapter_groups': len(adapter.groups), 'resident_models': resident_models,
                      'resident_optimizers': 1},
            'network': {**asdict(cfg.network_config()), 'measured_effective_delta_bytes': payload_bytes,
                        'topology': 'shared-server star, separate ingress/egress queues; barrier per round',
                        'virtual_clock': True,
                        'aggregation_source': ('fixed --aggregation-ms' if cfg.aggregation_ms is not None
                            else 'zero for fixed-compute comparisons' if cfg.compute_ms_per_step is not None
                            else 'measured FedSum apply time'),
                        'compute_source': ('fixed --compute-ms-per-step (optimizer + delta)' if
                            cfg.compute_ms_per_step is not None else 'measured per-peer optimizer + delta time')},
            'fairness': {'identical_genesis': True,
                         'identical_peer0_minibatches': True,
                         'equal_local_steps_per_peer': True,
                         'total_optimizer_steps_proportional_to_peers': True,
                         'total_training_tokens_proportional_to_peers': True,
                         'optimizer': 'AdamW', 'effective_delta_aggregation': 'FedSum without division'},
            'limitations': [
                'ONE resident model and optimizer; virtual peers run serially on this host.',
                'Virtual parallel time from recorded per-peer training compute plus simulated network, not real concurrent wall time.',
                'Synchronous coordinator-star FedSum; NOT live asynchronous peer-to-peer Gossip.',
                'Network uses RTT/2 propagation, Gaussian jitter, shared server ingress/egress bandwidth, whole-message drops/retry.',
                'No real sockets, TCP packetization/congestion control, NAT, cryptographic signing, journals or churn.',
                'A base-update download and effective-delta upload are modelled once per peer per round; payload byte count is estimated.',
                'All arms use the same starting weights, dataset, local steps and per-peer sampling streams; total work grows with peer count.',
                'Serial wall-clock includes disk swaps and validations; virtual parallel time deliberately excludes those serial artifacts.',
                'Evaluation runs after each round only to record quality; its duration is excluded from projected training time.',
                'Synthetic default corpus is not a natural-language benchmark; actual hardware speed is not validated.',
            ],
            'summary': summary,
            'outputs': {'plots': plots_available, 'trace': 'trace.csv', 'summary': 'summary.csv',
                        'primary_plot_x': 'virtual_time_s',
                        'secondary_plot_x': 'steps_per_peer',
                        'third_plot_x': 'tokens_seen'},
        }
        (output_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
    finally:
        torch.set_num_threads(old_threads)
