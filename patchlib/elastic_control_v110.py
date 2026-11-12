"""Bounded signed anti-entropy control plane and snapshot-source policy.

Deliberately excludes network IO, model ownership and generation rollover.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from .core import PEER_ID, PatchError, _canonical_json


CTRL_VERSION = 3
CTRL_MAX = 16*1024
INVENTORY_PAGE = 64


class HandshakeError(PatchError):
    """Unrecoverable peer identity or model-generation mismatch."""
    pass
REQUEST_PAGE = 32
REQUEST_BACKLOG_LIMIT = 4096  # identities, not patch bodies; bounded at each neighbor
FRONTIER_WINDOW = 256
# Signed control messages are capped at 16 KiB. Large rooms MUST page the
# origin frontier; never try to transmit thousands of origins in one frame.
FRONTIER_ORIGIN_LIMIT = 96  # accept v17-sized signed pages for compatibility
FRONTIER_PAGE_SIZE = 48  # v18 sends small rotating pages
FRONTIER_KNOWN_ORIGINS_LIMIT = 16384
FRONTIER_PENDING_LIMIT = 4096
PUSH_FANOUT_DEFAULT = 2
NEED_IN_FLIGHT_RETRY_SECONDS = 6.0  # avoid repeatedly requesting large events still in flight


def frontier_advertisement_delay(room_members: int, *, catchup: bool = False) -> float:
    """Pace redundant frontier pages without delaying initial hole discovery.

    A peer with missing events receives the fast control plane until its
    inventory/frontier repair has caught up. Large stable rooms amortize the
    per-neighbor vector clock cycle; full-event NEED stays demand-driven.
    """
    if catchup or room_members <= 64:
        return .3
    if room_members <= 512:
        return .75
    return 1.5


def inventory_cycle_delay(configured: float, room_members: int, *,
                          peer_synced: bool, has_frontier: bool) -> float:
    """A safety audit of event identities remains, but not every tick.

    The authenticated frontier/NEED path remains active between audits.
    Without v1 signed frontiers, keep the original interval unchanged.
    """
    if not peer_synced or not has_frontier:
        return configured
    return max(configured, 120.0 if room_members > 512 else 30.0)


def frontier_advertisement_page(heads, cursor: int, *, limit=FRONTIER_PAGE_SIZE):
    """A bounded, rotating page of the per-origin vector clock.

    The cursor advances only after a successful signed send. A new origin
    arriving or disappearing from the ordered list cannot cause starvation:
    every finite, stable list is fully advertised within ceil(N/limit) sends.
    """
    if not 1 <= limit <= FRONTIER_PAGE_SIZE:
        raise ValueError('invalid frontier page size')
    if not heads:
        return [], 0
    start = cursor % len(heads)
    # For a huge list use a fixed slice and wrap on the next send. Avoid
    # an oversized control by never carrying more than `limit` heads.
    page = heads[start:start + limit]
    return page, 0 if start + len(page) >= len(heads) else start + len(page)


def selective_push_targets(local_peer: str, candidate_peers, *,
                           exclude_peer=None, origin_peer=None,
                           event_seed: bytes = b'', fanout=PUSH_FANOUT_DEFAULT):
    """Bound unsolicited large-event pushes per hop, not per room size.

    Prefer the next clockwise connected ring neighbor (a deterministic
    connected backbone in bounded-v2). Choose any remaining shortcuts using
    a stable event-keyed rendezvous score; NEED/frontier repair remains the
    durable fallback under churn, network partitions and queue overflow.
    """
    if not 1 <= fanout <= 8:
        raise ValueError('invalid gossip push fanout')
    candidates = sorted({p for p in candidate_peers
                         if p not in (local_peer, exclude_peer, origin_peer)})
    if len(candidates) <= fanout:
        return candidates
    clockwise = next((p for p in candidates if p > local_peer), candidates[0])
    remaining = [p for p in candidates if p != clockwise]
    scored = sorted(remaining, key=lambda p: hashlib.blake2b(
        event_seed + p.encode('ascii'), digest_size=8).digest(), reverse=True)
    return [clockwise, *scored[:fanout-1]]



def validated_need(entries):
    """Validate an entire signed NEED before servicing any of its IDs."""
    if not isinstance(entries, list) or not 1 <= len(entries) <= REQUEST_PAGE:
        raise PatchError('bad need request')
    result=[]
    for ident in entries:
        if not (isinstance(ident,list) and len(ident)==2 and
                isinstance(ident[0],str) and PEER_ID.fullmatch(ident[0]) and
                type(ident[1]) is int and 1<=ident[1]<2**63):
            raise PatchError('bad requested event ID')
        result.append((ident[0],ident[1]))
    return result


def validated_frontiers(entries):
    """Only authenticated peers may advertise bounded per-origin maxima."""
    if not isinstance(entries,list) or len(entries)>FRONTIER_ORIGIN_LIMIT:
        raise PatchError('bad frontier advertisement')
    maxima={}
    for entry in entries:
        if not (isinstance(entry,list) and len(entry)==2 and
                isinstance(entry[0],str) and PEER_ID.fullmatch(entry[0]) and
                type(entry[1]) is int and 1<=entry[1]<2**63):
            raise PatchError('bad frontier entry')
        if entry[0] in maxima:
            raise PatchError('duplicate frontier origin')
        maxima[entry[0]]=entry[1]
    return maxima



def inventory_missing_bounded(node, journal, entries):
    """Index untrusted-but-signed inventory without unbounded pending growth.

    The sender will advertise the same identities again in periodic inventory
    and frontier messages, so it is safe to stop admitting IDs at the bound.
    Never mark an incomplete peer 'ready' while the page overflowed.
    """
    if not isinstance(entries, list) or len(entries) > INVENTORY_PAGE:
        raise PatchError('malformed inventory')
    missing=[]
    for item in entries:
        if not (isinstance(item,list) and len(item)==3 and
                isinstance(item[0],str) and PEER_ID.fullmatch(item[0]) and
                type(item[1]) is int and 1 <= item[1] < 2**63 and
                isinstance(item[2],str) and len(item[2])==64):
            raise PatchError('malformed event identity in inventory')
        key=(item[0],item[1])
        original=journal.has(*key)
        if original and original[0] != item[2]:
            raise PatchError('conflicting event in inventory')
        if original is None and key not in node.pending:
            if len(node.pending) >= FRONTIER_PENDING_LIMIT:
                node.ready=False
                node.control_overflow_drops+=1
                continue
            missing.append([*key])
            node.pending.add(key)
    if missing:
        node.ready=False
    return missing



def eligible_snapshot_sources(slots, local_peer: str):
    """Prefer verified, available remote sources; never fan out full states.

    A per-source retry counter provides automatic failover to less-failed
    sources. A stable hash spreads first-time requests across room holders.
    """
    candidates=[slot for slot in slots.values()
                if not slot.disabled and slot.node is not None and
                slot.node.connected.is_set() and slot.node.verified_at > 0 and
                slot.node.remote_can_serve_snapshot]
    return sorted(candidates, key=lambda s:(s.retries,hashlib.sha256(
        (local_peer+'/'+s.peer).encode('ascii')).digest()))


@dataclass
class SnapshotProviderPool:
    """Leaderless full-state bootstrap: prefer the active verified provider.

    No peer owns an administrator role. Once any peer has installed a verified
    snapshot and its durable journal, it can provide a snapshot to a newcomer.
    The receiver switches providers if the chosen peer leaves or stalls, while
    never requesting multi-GB exports from multiple providers at once.
    """
    current: str | None = None
    cooldown_until: dict[str, float] = field(default_factory=dict)

    def select(self, slots, local_peer: str, now: float):
        candidates = eligible_snapshot_sources(slots, local_peer)
        # Keep memory bounded as peers churn in long-lived rooms.
        present = {s.peer for s in candidates}
        self.cooldown_until = {k: v for k, v in self.cooldown_until.items()
                               if k in present and v > now}
        available = [s for s in candidates if s.peer not in self.cooldown_until]
        # A cooldown must not force an unsafe unverified peer into service.
        if not available:
            self.current = None
            return None
        for slot in available:
            if slot.peer == self.current:
                return slot
        self.current = available[0].peer
        return available[0]

    def failed(self, peer: str, now: float, delay: float = 30.0):
        self.cooldown_until[peer] = now + delay
        if self.current == peer:
            self.current = None


def snapshot_history_still_unchanged(last_mismatch, *, manifest_sha: str,
                                    inventory_sha: str, event_count: int,
                                    journal_count: int) -> bool:
    """Skip repeated O(history) inventory scans until durable history changes.

    The signed boundary and the local append-only journal count together
    identify a previously failed catch-up attempt. Missing events can only
    become available when a new durable event is appended.
    """
    return last_mismatch == (manifest_sha, inventory_sha, event_count, journal_count)


def can_supply_snapshot(*, room_only: bool, bootstrap) -> bool:
    """Verified state, never creator identity, confers snapshot-serving rights."""
    return not room_only or (bootstrap.snapshot_installed and not bootstrap.fresh_join)


def encode_control(msg: dict, signer=None) -> str:
    body = {'v':CTRL_VERSION, **msg}
    raw = _canonical_json(body)
    digest = hashlib.sha256(raw).hexdigest()
    header = {'size':len(raw),'sha256':digest,
              'sig':signer(len(raw),digest) if signer else None}
    wire = json.dumps({**body, '_auth':header},separators=(',',':'))
    if len(wire) > CTRL_MAX:
        raise PatchError('oversized elastic control')
    return wire


def decode_control(wire: str, verifier=None, unsafe=False) -> dict:
    if not isinstance(wire,str) or len(wire)>CTRL_MAX:
        raise PatchError('invalid elastic control size')
    try: msg=json.loads(wire)
    except ValueError as e: raise PatchError('bad elastic control JSON') from e
    if not isinstance(msg,dict) or msg.get('v')!=CTRL_VERSION or '_auth' not in msg:
        raise PatchError('not an elastic control frame')
    auth=msg.pop('_auth')
    if not isinstance(auth,dict) or set(auth)!={'size','sha256','sig'}:
        raise PatchError('invalid control auth')
    payload = _canonical_json(msg)
    if type(auth['size']) is not int or auth['size']!=len(payload) or hashlib.sha256(payload).hexdigest()!=auth['sha256']:
        raise PatchError('corrupt elastic control frame')
    if not unsafe:
        if verifier is None: raise PatchError('peer public key missing')
        verifier(auth)
    return msg


async def pump_heartbeats(slots, stop: asyncio.Event, *, interval: float, peer_timeout: float):
    """Transport keepalive must progress during async disk/GPU checkpoint work.

    This coroutine owns no model weights, and never awaits remote peers.
    """
    period=max(0.2, min(float(interval), float(peer_timeout)/3))
    while not stop.is_set():
        for slot in tuple(slots.values()):
            node=slot.node
            if node and not slot.disabled and node.connected.is_set():
                if hasattr(node, 'can_serve_snapshot'):
                    node.control('heartbeat',serves_snapshot=node.can_serve_snapshot)
                else:
                    node.control('heartbeat')
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except asyncio.TimeoutError:
            pass

