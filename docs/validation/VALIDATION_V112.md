# KCIA v112 — Serial PT Comparison Validation

## Scope

This release **only adds** a standalone offline serial causal-LM PT benchmark and documentation. It does NOT edit the live CLI, GUI, network/cryptographic protocol, or runtime model training code. The v111 release inventory test was relaxed from an exact 147 Python module count to >=147 to allow new modules.

## Algorithms and fidelity

- One resident TinyCausalLM + UniversalAdapter + AdamW; distinct virtual peers use disk-backed LoHA/AdamW states, no parallel peers.
- Actual Torch `forward()`, causal next-token cross-entropy, `backward()`, clipping, AdamW update.
- Uses `continual.enable_anchors` and effective factor delta `local.delta()-anchor.delta()`; sums those *effective* deltas at the end of each round without averaging (synchronous FedSum).
- All conditions share the exact same initial frozen base and initial factors; same deterministic minibatches and total gradient steps/tokens.
- Checkpoint swapping preserves separate peer optimizer states. State files use `weights_only=True` when loaded and are automatically removed when the benchmark ends.
- The simulation does **not** reproduce signed wire messages, asynchronous Gossip, real network conditions, or failover. This is an offline synchronization baseline, **not** a live-network certification.

## Tests

`PYTHONPATH=.:tests python -m pytest -q tests/test_v111_release_cleanup.py tests/test_v112_serial_pt.py tests/test_v110_modular_refactor.py tests/test_effectiveness_benchmark.py tests/test_continual_elastic.py tests/test_continual_fedsum_v3.py tests/test_p2p_training_peer_sweep_v23.py tests/test_p2p_training_sim_v21.py tests/test_p2p_training_sim_v22_timing.py`

**38 passed** in the test environment. Covers same minibatch hashes, exact total training tokens, initial loss, serial 1/2/4 peers, reproducibility, disk optimizer state, FedSum sum/dedup, local UTF-8 text, temporary scratch cleanup, invalid settings, former v111 source layout and existing effectiveness/P2P simulation tests. This environment cannot run Apple MPS hardware tests.

## Demonstration (CPU synthetic token task, not natural text)

```
python -m benchmarks.bench_serial_pt_v112 \
  --peers 1,2,4,8 --rounds 6 --steps-per-round 8 \
  --context 12 --width 24 --batch-size 4 \
  --train-sequences 256 --valid-sequences 64 \
  --output-dir <example>
```

All arms trained on **2,304 identical token positions, 48 optimizer steps**, and started at validation CE approximately **3.6342**.

| Peers | Final heldout CE | Final Top-1 | Serial wall time (s, local environment only) |
|------:|-----------------:|------------:|--------------------------------------------:|
| 1 | 2.9251 | 0.2044 | 0.234 |
| 2 | 3.0903 | 0.1497 | 0.293 |
| 4 | 3.1614 | 0.1354 | 0.382 |
| 8 | 3.1929 | 0.1198 | 0.621 |

The higher peer counts do **not** outperform 1 peer on this small, synchronous, non-natural-language test; do not generalize these results to actual asynchronous/distributed throughput or other datasets.

## Compatibility

- Existing `kcia.py`, `patchlib/cli.py`, `patchlib/elasticnet.py`, `web/` remain byte-for-byte identical to v111.
- No room reset, data migration, genesis change or GUI wiring.
- No performance/accuracy guarantees for Mac MPS or for real large language models.
