"""Deterministic discrete-event *virtual* star-network for the serial PT benchmark.

No sleep/network sockets/parallel peer objects. Each synchronous round has:
  coordinator -> all peers (base delta download), per-peer compute,
  peers -> coordinator (effective delta upload), then an all-peer barrier.

Modelled: separate shared server uplink/downlink queues, peer link caps,
one-way propagation with seeded jitter, whole-message drops and timeout/retry.
Unmodelled: KCIA P2P gossip, real packetization/TCP congestion, security,
NAT, byte-level protocol encoding, and actual network performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random


@dataclass(frozen=True)
class NetworkConfig:
    rtt_ms: float = 50.0
    jitter_ms: float = 5.0
    bandwidth_mbps: float = 50.0
    server_bandwidth_mbps: float = 100.0
    payload_bytes: int | None = None
    message_overhead_bytes: int = 1024
    loss_rate: float = 0.0
    retry_rto_ms: float = 250.0
    max_retries: int = 3
    seed: int = 20261107

    def validate(self):
        values = (self.rtt_ms, self.jitter_ms, self.bandwidth_mbps,
                  self.server_bandwidth_mbps, self.loss_rate, self.retry_rto_ms)
        if any(not math.isfinite(v) for v in values):
            raise ValueError('network timing/rates must be finite')
        if (self.rtt_ms < 0 or self.jitter_ms < 0 or self.bandwidth_mbps <= 0 or
                self.server_bandwidth_mbps <= 0 or not 0 <= self.loss_rate < 1 or
                self.retry_rto_ms <= 0 or not 0 <= self.max_retries <= 100 or
                self.message_overhead_bytes < 0 or
                (self.payload_bytes is not None and self.payload_bytes < 0)):
            raise ValueError('invalid simulated network parameters')


class SimulatedNetworkError(RuntimeError):
    pass


class StarNetworkSimulator:
    """A single seeded simulator per arm; timestamps are virtual seconds.

    Shared server ingress/egress have separate queues (full duplex).
    Per-peer links are independent and capped at ``bandwidth_mbps``.
    A rejected message consumes wire capacity and retries after RTO.
    """
    def __init__(self, config: NetworkConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        self.now_s = 0.0
        self.total_wire_bytes = 0
        self.total_messages = 0
        self.total_retries = 0
        self.total_drops = 0
        self.total_network_sync_s = 0.0
        self.total_compute_critical_s = 0.0
        self.total_wait_s = 0.0

    def round(self, compute_seconds, payload_bytes: int, *, aggregation_seconds: float = 0.0):
        """Advance one barrier round given individually *measured* peer compute durations."""
        if not compute_seconds or any(not math.isfinite(t) or t < 0 for t in compute_seconds):
            raise ValueError('per-peer compute time must be finite and non-negative')
        if payload_bytes < 0 or aggregation_seconds < 0:
            raise ValueError('invalid payload or aggregation time')
        n = len(compute_seconds)
        start_s = self.now_s
        # One peer needs no artificial network. The local optimizer still does real work.
        if n == 1:
            self.now_s += compute_seconds[0] + aggregation_seconds
            self.total_compute_critical_s += compute_seconds[0]
            return {'round_virtual_s': self.now_s - start_s, 'round_network_sync_s': 0.,
                    'round_critical_compute_s': compute_seconds[0], 'round_wait_s': 0.,
                    'messages': 0, 'retries': 0, 'drops': 0, 'wire_bytes': 0}

        cfg = self.config
        bytes_each = (cfg.payload_bytes if cfg.payload_bytes is not None else payload_bytes) + cfg.message_overhead_bytes
        # Timestamp, stable insertion index, phase, peer, previous retry count.
        events = []
        serial = 0
        def schedule(ready, phase, peer, retry):
            nonlocal serial
            heapq.heappush(events, (ready, serial, phase, peer, retry))
            serial += 1

        for peer in range(n):
            schedule(start_s, 'download', peer, 0)
        server_free = {'download': start_s, 'upload': start_s}
        peer_free = {'download': [start_s] * n, 'upload': [start_s] * n}
        uploaded = {}
        count_messages = count_retries = count_drops = count_bytes = 0
        while events:
            ready, _, phase, peer, retry = heapq.heappop(events)
            # Server transport channel is shared; per-peer channels are independent.
            begins = max(ready, server_free[phase], peer_free[phase][peer])
            server_tx = 8 * bytes_each / (cfg.server_bandwidth_mbps * 1e6)
            peer_tx = 8 * bytes_each / (cfg.bandwidth_mbps * 1e6)
            server_free[phase] = begins + server_tx
            peer_free[phase][peer] = begins + peer_tx
            jitter = self.rng.gauss(0.0, cfg.jitter_ms) if cfg.jitter_ms else 0.0
            propagation_s = max(0.0, (cfg.rtt_ms / 2 + jitter) / 1000.)
            arrived = begins + max(server_tx, peer_tx) + propagation_s
            count_messages += 1
            count_bytes += bytes_each
            if self.rng.random() < cfg.loss_rate:
                count_drops += 1
                if retry >= cfg.max_retries:
                    raise SimulatedNetworkError(
                        f'simulated {phase} to/from peer={peer} lost after {retry + 1} attempts; '
                        'lower --loss-rate or increase --max-retries')
                count_retries += 1
                schedule(arrived + cfg.retry_rto_ms / 1000., phase, peer, retry + 1)
            elif phase == 'download':
                schedule(arrived + compute_seconds[peer], 'upload', peer, 0)
            else:
                if peer in uploaded:
                    raise RuntimeError('duplicate virtual upload for a peer')
                uploaded[peer] = arrived
        if len(uploaded) != n:
            raise RuntimeError('virtual barrier missing updates')
        barrier_s = max(uploaded.values())
        wait_s = sum(barrier_s - time for time in uploaded.values())
        self.now_s = barrier_s + aggregation_seconds
        compute_critical = max(compute_seconds)
        network_sync = max(0., barrier_s - start_s - compute_critical)
        self.total_wire_bytes += count_bytes
        self.total_messages += count_messages
        self.total_retries += count_retries
        self.total_drops += count_drops
        self.total_network_sync_s += network_sync
        self.total_compute_critical_s += compute_critical
        self.total_wait_s += wait_s
        return {'round_virtual_s': self.now_s - start_s,
                'round_network_sync_s': network_sync,
                'round_critical_compute_s': compute_critical,
                'round_wait_s': wait_s, 'messages': count_messages,
                'retries': count_retries, 'drops': count_drops, 'wire_bytes': count_bytes}
