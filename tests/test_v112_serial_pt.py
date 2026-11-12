"""V112.2 CPU regression: equal per-peer work, scaled total, serial disk peers, FedSum."""
import csv
import json
from pathlib import Path

import pytest
import torch

from benchmarks.bench_serial_pt_v112 import main
from patchlib.continual import enable_anchors
from patchlib.core import UniversalAdapter
from patchlib.effectiveness_benchmark import TinyCausalLM
from patchlib.serial_pt_benchmark_v112 import (
    SerialPTConfig, _FedSumRound, _load_peer, _original, _seal_effective_delta,
    _write_peer, run_serial_pt_benchmark,
)


def config(**extra):
    values = dict(peers=(1, 2, 4), rounds=2, steps_per_round=4,
                  batch_size=2, context=8, width=16, rank=2,
                  train_sequences=32, valid_sequences=12, seed=18)
    values.update(extra)
    return SerialPTConfig(**values)


def test_serial_peer_comparison_equal_tokens_and_samples(tmp_path):
    out = tmp_path / 'results'
    result = run_serial_pt_benchmark(config(), out)
    assert result['kind'] == 'OFFLINE_SERIAL_SYNCHRONOUS_FEDSUM_PT_WITH_VIRTUAL_NETWORK'
    assert result['model']['resident_models'] == 1
    assert result['model']['resident_optimizers'] == 1
    summaries = result['summary']
    assert [r['peers'] for r in summaries] == [1, 2, 4]
    assert [r['total_tokens'] for r in summaries] == [128, 256, 512]
    assert [r['total_steps'] for r in summaries] == [8, 16, 32]
    assert len({r['tokens_per_peer'] for r in summaries}) == 1
    assert len({r['steps_per_peer'] for r in summaries}) == 1
    assert len({r['peer0_samples_sha256'] for r in summaries}) == 1
    assert len({r['sample_schedule_sha256'] for r in summaries}) == 3
    assert len({r['initial_base_sha256'] for r in summaries}) == 1
    assert all(0 <= r['valid_top1'] <= 1 and r['valid_ce'] > 0 for r in summaries)
    assert all(r['serial_wall_s'] > 0 for r in summaries)
    for r in summaries:
        assert r['tokens_per_peer'] == r['total_tokens'] // r['peers']
        assert r['steps_per_peer'] == r['total_steps'] // r['peers']
    rows = list(csv.DictReader((out / 'trace.csv').open()))
    assert len(rows) == (2 + 1) * 3
    initial = [float(row['valid_ce']) for row in rows if int(row['round']) == 0]
    for row in rows:
        assert int(row['tokens_per_peer']) * int(row['peers']) == int(row['tokens_seen'])
        assert int(row['steps_per_peer']) * int(row['peers']) == int(row['steps'])
    assert max(initial) - min(initial) < 1e-8
    assert (out / 'summary.csv').is_file()
    assert (out / 'result.json').is_file()
    assert result['outputs']['primary_plot_x'] == 'virtual_time_s'
    assert result['fairness']['equal_local_steps_per_peer']
    assert all(r['simulated_messages'] == (0 if r['peers'] == 1 else 2 * 2 * r['peers']) for r in summaries)
    assert all(r['virtual_time_s'] > 0 for r in summaries)
    assert summaries[0]['network_sync_s'] == 0
    assert summaries[1]['network_sync_s'] > 0
    if result['outputs']['plots']:
        for metric in ('loss', 'accuracy'):
            assert (out / f'pt-heldout-{metric}.png').is_file()
            assert (out / f'pt-heldout-{metric}-total-tokens.png').is_file()
            assert (out / f'pt-heldout-{metric}-local-steps.png').is_file()
            assert (out / f'pt-heldout-{metric}-ideal-compute.png').is_file()
    assert set(result['limitations'])


def test_runs_reproducibly(tmp_path):
    a = run_serial_pt_benchmark(config(), tmp_path / 'one')
    b = run_serial_pt_benchmark(config(), tmp_path / 'two')
    for x, y in zip(a['summary'], b['summary']):
        assert x['valid_ce'] == pytest.approx(y['valid_ce'], abs=1e-7)
        assert x['valid_top1'] == y['valid_top1']


def test_factor_anchor_and_moments_disk_roundtrip(tmp_path):
    torch.manual_seed(19)
    model = TinyCausalLM(vocab=32, width=16, context=8)
    adapter = UniversalAdapter(model, 'test', 'rev', kind='loha', rank=2, target_scope='linear')
    enable_anchors(adapter)
    for p in model.parameters():
        if not any(p is q for q in adapter.trainable_parameters()):
            p.requires_grad_(False)
    optim = torch.optim.AdamW(list(adapter.trainable_parameters()), lr=.003)
    original = [_original(model, g).detach().clone() for g in adapter.groups]
    before = _seal_effective_delta(adapter)
    assert all(torch.equal(t, torch.zeros_like(t)) for t in before.values())
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), y.reshape(-1))
    loss.backward()
    optim.step()
    changed = _seal_effective_delta(adapter)
    assert any(torch.count_nonzero(v).item() > 0 for v in changed.values())
    saved_factor = [p.detach().clone() for p in adapter.trainable_parameters()]
    state_file = tmp_path / 'peer.pt'
    _write_peer(state_file, adapter, optim, 5)
    for param in adapter.trainable_parameters():
        param.data.zero_()
    optim.state.clear()
    assert _load_peer(state_file, adapter, optim) == 5
    for x, y in zip(saved_factor, adapter.trainable_parameters()):
        torch.testing.assert_close(x, y)
    assert optim.state
    repeat = _seal_effective_delta(adapter)
    assert all(torch.equal(t, torch.zeros_like(t)) for t in repeat.values())
    for p, baseline in zip([_original(model, g) for g in adapter.groups], original):
        torch.testing.assert_close(p, baseline)


def test_fedsum_adds_once_and_is_not_average():
    model = TinyCausalLM(vocab=32, width=16, context=8)
    adapter = UniversalAdapter(model, 'test', 'rev', kind='loha', rank=2, target_scope='linear')
    enable_anchors(adapter)
    originals = {g.canonical: _original(model, g).clone().detach() for g in adapter.groups}
    a = {key: torch.full_like(value, .1) for key, value in originals.items()}
    b = {key: torch.full_like(value, .3) for key, value in originals.items()}
    accumulator = _FedSumRound(model, adapter)
    assert accumulator.add(0, a)
    assert not accumulator.add(0, a)
    assert accumulator.add(1, b)
    accumulator.apply(model, adapter, 2)
    for group in adapter.groups:
        torch.testing.assert_close(_original(model, group), originals[group.canonical] + .4)
    with pytest.raises(RuntimeError, match='missing peer'):
        _FedSumRound(model, adapter).apply(model, adapter, 1)


def test_peer_counts_do_not_need_divisibility():
    config(peers=(1, 3), steps_per_round=2).validate()
    with pytest.raises(ValueError, match='beginning with 1'):
        config(peers=(2, 4)).validate()


def test_local_corpus_and_cli_entrypoint(tmp_path):
    text = tmp_path / 'stories.txt'
    text.write_text('Once upon a time the small robot learned by reading books.\n' * 60)
    report = main(['--peers', '1,2', '--rounds', '1', '--steps-per-round', '2',
                   '--context', '8', '--width', '16', '--rank', '2',
                   '--train-sequences', '16', '--valid-sequences', '8',
                   '--batch-size', '2', '--text-file', str(text),
                   '--output-dir', str(tmp_path / 'out')])
    assert report['data']['type'] == 'local-utf8-bytes'
    assert len(report['summary']) == 2
    assert (tmp_path / 'out' / 'pt-heldout-loss.png').is_file() or not report['outputs']['plots']


def test_custom_scratch_is_cleaned(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    cfg = config(peers=(1, 2), rounds=1, steps_per_round=2,
                 scratch_dir=str(scratch))
    run_serial_pt_benchmark(cfg, tmp_path / 'output')
    assert list(scratch.iterdir()) == []


def test_large_input_budget_is_refused():
    with pytest.raises(ValueError, match='128MiB'):
        config(peers=(1,), train_sequences=1_000_000, context=256,
               valid_sequences=100_000).validate()
