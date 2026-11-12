# Kagami v112.2: PT scaled-work and discrete-event latency validation

Scope: offline serial benchmark only (`benchmarks/bench_serial_pt_v112.py`,
`patchlib/serial_pt_benchmark_v112.py`, new `patchlib/serial_pt_network_v1122.py`).
The GUI, signed event protocol, P2P runtime and persistent room stores were not changed.

## Regression testing

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/test_v112_serial_pt.py tests/test_v1122_virtual_network.py \
  tests/test_effectiveness_benchmark.py tests/test_mps_policy_v92.py \
  tests/test_swarm_gossip_fairness.py tests/test_v108_mps_acceleration.py
```

Tests validate:
- every peer has the same steps/tokens in all arms; aggregate increases in proportion to peers;
- identical initial base and peer-0 sample-stream hashes;
- successful PT optimizer updates with one shared resident model/optimizer and per-peer disk state;
- correct FedSum (not FedAvg) and once-only effective delta application;
- virtual RTT/2 per direction, seeded jitter, separate shared uplink/downlink contention;
- message-level loss, timeout/retry and explicit exhaustion failure;
- one-peer local baseline has zero virtual network traffic;
- changing RTT affects virtual time, but cannot silently alter synchronous PT weights or accuracy;
- fixed synthetic compute/aggregation durations reproduce identical virtual timestamps;
- CLI, CSV/JSON and all four axes for CE/Accuracy charts.

## Limitations

This does not test actual simultaneous execution or asynchronous signed Gossip. The virtual
star-network is a simplified synchronously aggregated system. Host compute microbenchmarks
are measured by physical serial execution, and virtual wall clock is an estimate, not a
macOS network speed claim. Real MPS execution was not tested here.
