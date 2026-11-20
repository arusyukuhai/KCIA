"""Bounded, local-only communication-cost contextual bandit for KCIA WAN training.

A quadratic ridge model predicts *observed training-loss trend* per minibatch
versus log2(publish interval). A conservative neighbor-only UCB search trades that
weak proxy against expected signed wire MiB and staleness. This is NOT a causal
estimate of generalization or global convergence. No unsupervised network peer
can set the reward, policy, or signed room state.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import weakref
from collections import deque
from pathlib import Path

_MIB = 1024 * 1024


def _finite(value):
    try:
        x = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def eligible_live_push_slots(slots, *, now: float, peer_timeout: float) -> dict:
    """Exactly the authenticated, synchronized connections eligible for a body push.

    Shared between the gossip send path and the WAN cost estimator so an
    ICE-only or snapshot-joining link is never counted as a live recipient.
    """
    if (_finite(now) is None or _finite(peer_timeout) is None or
            peer_timeout <= 0):
        raise ValueError('invalid publication liveness window')
    eligible = {}
    for peer, slot in slots.items():
        node = getattr(slot, 'node', None)
        if (getattr(slot, 'disabled', False) or node is None or
                getattr(node, 'bridge_only', False) or
                not getattr(node, 'ready', False) or
                not getattr(node, 'connected', None) or
                not node.connected.is_set()):
            continue
        verified = _finite(getattr(node, 'verified_at', None))
        if verified is None or not 0 < verified <= now or now - verified >= peer_timeout:
            continue
        eligible[getattr(slot, 'peer', peer)] = slot
    return eligible


def live_push_recipients(slots, *, fanout: int, now: float,
                         peer_timeout: float) -> int:
    """Upper bound on eligible immediate recipients for one local publication."""
    if type(fanout) is not int or not 1 <= fanout <= 8:
        raise ValueError('invalid peer publication fanout')
    return min(fanout, len(eligible_live_push_slots(
        slots, now=now, peer_timeout=peer_timeout)))


class WireEgressTracker:
    """Count each RTC generation's actual push/repair bytes once.

    The previous sum-of-all-nodes counter resets when ICE reconnects or a
    peer leaves. A single global subtraction then loses traffic for many
    events until newly created channels exceed the old departed counters.
    Track node objects weakly so churn cannot grow this diagnostic cache.
    """

    def __init__(self):
        self._last = weakref.WeakKeyDictionary()

    def poll(self, slots) -> int:
        increment = 0
        seen = set()
        for slot in slots.values():
            node = getattr(slot, 'node', None)
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            counts = (getattr(node, 'push_sent_wire_bytes', 0),
                      getattr(node, 'repair_sent_wire_bytes', 0))
            if any(type(v) is not int or v < 0 for v in counts):
                raise ValueError('negative or invalid RTC wire counters')
            previous = self._last.get(node, (0, 0))
            # An in-place counter reset is also possible when transports are
            # reinitialized; count post-reset bytes instead of losing them.
            increment += sum((now-old if now >= old else now)
                             for now, old in zip(counts, previous))
            self._last[node] = counts
        return increment


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Small, regularized 3x3 system (no numpy / model device dependency)."""
    a = [row[:] + [float(value)] for row, value in zip(matrix, vector)]
    n = len(a)
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(a[i][k]))
        if abs(a[p][k]) < 1e-10:
            raise ValueError('singular online regression')
        a[k], a[p] = a[p], a[k]
        scale = a[k][k]
        for j in range(k, n + 1):
            a[k][j] /= scale
        for i in range(n):
            if i == k:
                continue
            mul = a[i][k]
            for j in range(k, n + 1):
                a[i][j] -= mul * a[k][j]
    return [row[n] for row in a]


class WanCostBandit:
    """Policy updates only after completed, locally signed publication blocks.

    Quality is normalized minibatch training-loss slope, not validation loss.
    Every observed event is marked invalid if received peer updates intervened,
    or if its local batch was incomplete, so no false cause is ascribed to WAN.
    """

    SCHEMA = 1
    HISTORY = 64

    def __init__(self, *, baseline: int, ceiling: int = 1024,
                 wire_cost: float = 0.08, exploration: float = 0.04,
                 path: Path | None = None):
        if (type(baseline) is not int or type(ceiling) is not int or
                baseline < 4 or ceiling < baseline or ceiling > 10000):
            raise ValueError('invalid cadence bounds')
        if (_finite(wire_cost) is None or not 0 <= wire_cost <= 10 or
                _finite(exploration) is None or not 0 <= exploration <= 1):
            raise ValueError('invalid bandit utility weights')
        self.baseline = baseline
        self.ceiling = ceiling
        self.wire_cost = float(wire_cost)
        self.exploration = float(exploration)
        self.actions = [baseline]
        while self.actions[-1] < ceiling:
            next_value = min(ceiling, self.actions[-1] * 2)
            if next_value == self.actions[-1]:
                break
            self.actions.append(next_value)
        self.selected = baseline
        self.history: list[dict] = []
        self.wire_estimate_mib = 0.0
        self.last_utility: float | None = None
        self.last_prediction: float | None = None
        self.last_improvement: float | None = None
        self.last_wire_per_step: float | None = None
        self.events = 0
        self.path = Path(path) if path is not None else None
        self._head: list[float] = []
        self._tail: deque[float] = deque(maxlen=8)
        self._count = 0
        self._inbound_start: int | None = None
        self._dirty = False
        self.load()

    @property
    def samples(self) -> int:
        return len(self.history)

    def step(self, loss: float, *, inbound_applied: int):
        """Collect a few loss values, never tensors or private training data."""
        if self._inbound_start is None:
            self._inbound_start = inbound_applied
        if inbound_applied != self._inbound_start:
            self._dirty = True
        self._count += 1
        v = _finite(loss)
        if v is None:
            self._dirty = True
            return
        if len(self._head) < 8:
            self._head.append(v)
        self._tail.append(v)

    def _reset_block(self):
        self._head.clear()
        self._tail.clear()
        self._count = 0
        self._inbound_start = None
        self._dirty = False

    def observe(self, *, steps: int, signed_bytes: int, inbound_applied: int,
                recipients: int = 1, forced: bool = False,
                measured_egress_bytes: int = 0) -> bool:
        """Finish a block, price one signed patch per expected remote recipient.

        Signed bytes, not bytes already sent, are available synchronously when
        sealing: network send queues drain asynchronously. Runtime telemetry
        tracks actual push/repair bytes separately for later calibration.
        """
        if (type(signed_bytes) is not int or signed_bytes < 0 or
                type(steps) is not int or steps < 0 or
                type(recipients) is not int or not 0 <= recipients <= 8 or
                type(measured_egress_bytes) is not int or measured_egress_bytes < 0):
            raise ValueError('invalid signed publication measurement')
        if recipients == 0:
            # A locally sealed update while offline has zero prospective WAN
            # recipients. There is no observed communication trade-off, even
            # if counters include late traffic from previously sent events.
            # Do not teach the online policy that one whole huge LoHA patch
            # was transmitted or turn an unconnected session into an arm trial.
            self._reset_block()
            return False
        self.events += 1
        # Actual push/repair counters may lag a published event because
        # the patch DataChannel drains asynchronously. Never mistake a zero
        # counter for free communication; signed bytes remain a lower bound.
        mib = max(signed_bytes * recipients, measured_egress_bytes) / _MIB
        if mib > 0:
            self.wire_estimate_mib = (mib if self.wire_estimate_mib <= 0
                                     else 0.75 * self.wire_estimate_mib + 0.25 * mib)
        valid = (not forced and not self._dirty and steps == self._count and
                 steps >= 8 and steps in self.actions and
                abs(steps-self.selected) <= max(2,self.selected // 16) and
                 self._inbound_start == inbound_applied and
                 len(self._head) >= 4 and len(self._tail) >= 4)
        if valid:
            # Use disjoint windows for short blocks. With 8 minibatches,
            # comparing the first 8 to the last 8 silently scores ZERO gain.
            window = min(8, max(4, steps // 4))
            start = statistics.median(self._head[:window])
            end = statistics.median(list(self._tail)[-window:])
            if abs(start) <= 1e-8:
                valid = False
            else:
                # Dimensionless relative-loss improvement / 1000 minibatches.
                # Clip outliers: data may be non-IID, particularly mixed SFT/PPO.
                rate = max(-2., min(2., 1000. * (start - end) /
                                       max(abs(start), 1e-4) / steps))
                self.last_improvement = rate
                self.last_wire_per_step = mib / steps
                self.history.append({'arm': self.selected, 'gain': rate})
                self.history = self.history[-self.HISTORY:]
        self._reset_block()
        # Prevent frequent flips: every third event at most, and >=3 usable
        # local observations; local regime changes gradually (<=2x per flip).
        if self.events % 3 == 0 and len(self.history) >= 3:
            self.choose()
        self.persist()
        return valid

    def _model(self):
        # Exponential forgetting lets local model/data phase change over time.
        # Ridge shrinkage prevents explosive parabola extrapolation from just
        # one chosen arm; confidence bonus encourages cautious exploration.
        a = [[3.0,0.,0.],[0.,4.0,0.],[0.,0.,8.0]]
        b = [0.,0.,0.]
        for age, item in enumerate(reversed(self.history)):
            w = 0.97 ** age
            x = math.log2(item['arm'] / self.baseline)
            features = [1., x, x*x]
            for i in range(3):
                b[i] += w * features[i] * item['gain']
                for j in range(3):
                    a[i][j] += w * features[i] * features[j]
        return _solve(a,b)

    def utility(self, steps: int) -> float:
        if not self.history:
            return 0.
        coefficients = self._model()
        x = math.log2(steps / self.baseline)
        gain = max(-2., min(2., coefficients[0] + coefficients[1]*x + coefficients[2]*x*x))
        # Opportunity cost per minibatch; explicit staleness penalty guards
        # against infinitely long unpublished local updates.
        cost = self.wire_cost * self.wire_estimate_mib / steps
        staleness = 0.012 * x
        n_arm = sum(1 for event in self.history if event['arm'] == steps)
        bonus = self.exploration / math.sqrt(n_arm + 1)
        return gain - cost - staleness + bonus

    def choose(self) -> int:
        if self.samples < 3:
            return self.selected
        index = self.actions.index(self.selected)
        lo, hi = max(0,index-1), min(len(self.actions)-1,index+1)
        options = self.actions[lo:hi+1]
        values = [(self.utility(k), k) for k in options]
        best = max(values, key=lambda pair:(pair[0], -abs(pair[1] - self.selected)))
        current_value = self.utility(self.selected)
        self.last_prediction = best[0]
        # Hysteresis prevents oscillations on small noisy reward differences.
        if best[1] != self.selected and best[0] > current_value + 0.015:
            self.selected = best[1]
        self.last_utility = self.utility(self.selected)
        return self.selected

    def persist(self):
        """Local-only, atomic advisory cache; never modifies signed history."""
        if self.path is None or self.path.is_symlink():
            return
        item = {'version':self.SCHEMA, 'baseline':self.baseline, 'ceiling':self.ceiling,
                'selected':self.selected, 'events':self.events,
                'wire_cost':self.wire_cost,'exploration':self.exploration,
                'wire_estimate_mib':self.wire_estimate_mib,'history':self.history}
        part = self.path.with_name(self.path.name+'.tmp-'+os.urandom(6).hex())
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            fd = os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                json.dump(item,f,allow_nan=False,separators=(',',':'))
                f.flush()
                os.fsync(f.fileno())
            os.replace(part,self.path)
        except OSError:
            pass  # policy cache failure must not terminate training
        finally:
            # An advisory cache must not crash training even if directory
            # permissions change or the filesystem fails during cleanup.
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass

    def load(self):
        if self.path is None or self.path.is_symlink() or not self.path.is_file():
            return
        try:
            if self.path.stat().st_size > 50_000:
                return
            obj = json.loads(self.path.read_text(encoding='utf-8'))
            # A JSON cache is disposable and may be truncated, edited or
            # replaced by a previous version. Non-object roots used to crash
            # model startup with an uncaught AttributeError.
            if type(obj) is not dict:
                return
            if (type(obj.get('version')) is not int or
                    obj.get('version') != self.SCHEMA or obj.get('baseline') != self.baseline
                    or obj.get('ceiling') != self.ceiling or
                    obj.get('wire_cost') != self.wire_cost or
                    obj.get('exploration') != self.exploration):
                # Do not reuse a previous decision after the user changes
                # the cost/exploration preference in the GUI.
                return
            selected = obj.get('selected')
            events = obj.get('events')
            wire = _finite(obj.get('wire_estimate_mib'))
            rows = obj.get('history')
            if type(selected) is not int or selected not in self.actions or type(events) is not int or not 0 <= events <= 10**12:
                return
            if wire is None or wire < 0 or wire > 10**9 or not isinstance(rows,list) or len(rows)>self.HISTORY:
                return
            clean=[]
            for row in rows:
                if type(row) is not dict:
                    return
                arm=row['arm']; gain=_finite(row['gain'])
                if type(arm) is not int or arm not in self.actions or gain is None or not -2 <= gain <= 2:
                    return
                clean.append({'arm':arm,'gain':gain})
            self.selected=selected
            self.events=events
            self.wire_estimate_mib=wire
            self.history=clean
        except (OSError,ValueError,TypeError,KeyError,UnicodeError):
            return
