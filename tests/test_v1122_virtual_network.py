"""Deterministic event-time network modeling: RTT, shared links, jitter, loss/retry."""
import csv
import pytest

from patchlib.serial_pt_network_v1122 import NetworkConfig, SimulatedNetworkError, StarNetworkSimulator
from patchlib.serial_pt_benchmark_v112 import SerialPTConfig, run_serial_pt_benchmark


def cfg(**kw):
    args = dict(rtt_ms=100., jitter_ms=0., bandwidth_mbps=1e8,
                server_bandwidth_mbps=1e8, payload_bytes=0,
                message_overhead_bytes=0, loss_rate=0., seed=9)
    args.update(kw)
    return NetworkConfig(**args)


def test_one_peer_no_artificial_network_cost():
    net = StarNetworkSimulator(cfg(rtt_ms=900., loss_rate=.5))
    result = net.round([.3], payload_bytes=8192)
    assert result['round_virtual_s'] == pytest.approx(.3)
    assert result['messages'] == 0
    assert net.now_s == pytest.approx(.3)


def test_two_peers_rtt_both_legs_and_barrier():
    net = StarNetworkSimulator(cfg())
    result = net.round([.3, .6], payload_bytes=0)
    # Each peer receives an update after RTT/2; then computes; uploads after RTT/2.
    assert result['round_virtual_s'] == pytest.approx(.7)
    assert result['round_critical_compute_s'] == pytest.approx(.6)
    assert result['round_network_sync_s'] == pytest.approx(.1)
    assert result['round_wait_s'] == pytest.approx(.3)
    assert result['messages'] == 4
    net.round([.3, .6], payload_bytes=0)
    assert net.now_s == pytest.approx(1.4)


def test_message_bandwidth_and_shared_server_serialization():
    fast = StarNetworkSimulator(cfg(bandwidth_mbps=100, server_bandwidth_mbps=100, payload_bytes=None))
    slow = StarNetworkSimulator(cfg(bandwidth_mbps=100, server_bandwidth_mbps=1, payload_bytes=None))
    fast.round([.1] * 4, payload_bytes=1_000_000)
    slower = slow.round([.1] * 4, payload_bytes=1_000_000)
    assert slower['round_virtual_s'] > fast.now_s + 25
    assert slower['wire_bytes'] == 8_000_000  # 2 directions * 4 peers


def test_jitter_is_seeded_and_mean_propagation_nonnegative():
    a = StarNetworkSimulator(cfg(jitter_ms=20))
    b = StarNetworkSimulator(cfg(jitter_ms=20))
    c = StarNetworkSimulator(cfg(jitter_ms=20, seed=11))
    assert a.round([.1] * 4, payload_bytes=100) == b.round([.1] * 4, payload_bytes=100)
    assert a.now_s != c.round([.1] * 4, payload_bytes=100)['round_virtual_s']


def test_loss_retries_and_failure():
    with pytest.raises(SimulatedNetworkError, match='lost after 1 attempts'):
        StarNetworkSimulator(cfg(loss_rate=.999999, max_retries=0)).round([.1] * 2, 1000)
    # Real message attempts are counted even when a virtual attempt is lost.
    observed = []
    for seed in range(12):
        try:
            n = StarNetworkSimulator(cfg(loss_rate=.4, max_retries=30, seed=seed))
            r = n.round([.1] * 4, 1000)
        except SimulatedNetworkError:
            continue
        observed.append(r)
    assert any(r['retries'] > 0 and r['drops'] == r['retries'] for r in observed)
    assert any(r['messages'] > 8 for r in observed)


def test_invalid_network_parameters_rejected():
    for invalid in (dict(rtt_ms=-1), dict(jitter_ms=-1), dict(loss_rate=1),
                    dict(bandwidth_mbps=0), dict(max_retries=-1),
                    dict(message_overhead_bytes=-1), dict(retry_rto_ms=0)):
        with pytest.raises(ValueError):
            cfg(**invalid).validate()


def test_equal_local_steps_scaled_budget_with_deterministic_virtual_time(tmp_path):
    options = dict(peers=(1, 2, 4), rounds=2, steps_per_round=2,
                   batch_size=2, context=8, width=16, rank=2,
                   train_sequences=32, valid_sequences=12, seed=18,
                   compute_ms_per_step=100., jitter_ms=0., rtt_ms=100.,
                   bandwidth_mbps=100., server_bandwidth_mbps=100.)
    result = run_serial_pt_benchmark(SerialPTConfig(**options), tmp_path)
    summary = result['summary']
    assert [r['total_steps'] for r in summary] == [4, 8, 16]
    assert [r['total_tokens'] for r in summary] == [64, 128, 256]
    assert {r['steps_per_peer'] for r in summary} == {4}
    assert {r['tokens_per_peer'] for r in summary} == {64}
    assert len({r['peer0_samples_sha256'] for r in summary}) == 1
    assert summary[1]['network_sync_s'] > 0
    assert summary[2]['network_sync_s'] > summary[0]['network_sync_s']
    assert summary[1]['virtual_time_s'] > summary[0]['virtual_time_s']
    assert summary[0]['critical_compute_s'] == pytest.approx(.4)
    assert result['fairness']['total_optimizer_steps_proportional_to_peers']
    assert result['network']['virtual_clock']
    rows = list(csv.DictReader((tmp_path / 'trace.csv').open()))
    assert len(rows) == 9
    assert int(rows[-1]['simulated_messages']) == 16


def test_cpu_sequence_determinism_and_rtt_affects_only_time(tmp_path):
    args = dict(peers=(1, 2), rounds=1, steps_per_round=2, batch_size=2,
                context=8, width=16, rank=2, train_sequences=32,
                valid_sequences=12, compute_ms_per_step=100., jitter_ms=0., seed=18)
    short = run_serial_pt_benchmark(SerialPTConfig(**args, rtt_ms=0), tmp_path / 'low')
    long = run_serial_pt_benchmark(SerialPTConfig(**args, rtt_ms=300), tmp_path / 'high')
    for a, b in zip(short['summary'], long['summary']):
        assert a['valid_ce'] == b['valid_ce']
        assert a['valid_top1'] == b['valid_top1']
        assert a['peer0_samples_sha256'] == b['peer0_samples_sha256']
    assert long['summary'][1]['virtual_time_s'] > short['summary'][1]['virtual_time_s'] + .29


def test_fixed_compute_and_aggregation_reproduce_exact_virtual_time(tmp_path):
    args = dict(peers=(1, 2, 4), rounds=2, steps_per_round=2,
                batch_size=2, context=8, width=16, rank=2,
                train_sequences=32, valid_sequences=12, seed=41,
                rtt_ms=70., jitter_ms=11., compute_ms_per_step=40.,
                aggregation_ms=3.)
    first = run_serial_pt_benchmark(SerialPTConfig(**args), tmp_path / 'a')
    second = run_serial_pt_benchmark(SerialPTConfig(**args), tmp_path / 'b')
    assert [r['virtual_time_s'] for r in first['summary']] == [
        r['virtual_time_s'] for r in second['summary']]
    assert [r['valid_top1'] for r in first['summary']] == [
        r['valid_top1'] for r in second['summary']]
    assert first['network']['aggregation_source'] == 'fixed --aggregation-ms'
