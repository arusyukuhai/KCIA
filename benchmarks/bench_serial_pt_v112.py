#!/usr/bin/env python3
"""One physical model, serial virtual peers: PT-only 1 vs multi-peer benchmark.

From repository root:
    python -m benchmarks.bench_serial_pt_v112 --peers 1,2,4 --rounds 8
"""
from __future__ import annotations

import argparse
import json
from patchlib.serial_pt_benchmark_v112 import SerialPTConfig, run_serial_pt_benchmark


def _peers(value):
    try:
        values = tuple(int(s.strip()) for s in value.split(',') if s.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError('peers must be comma-separated integers') from exc
    if not values:
        raise argparse.ArgumentTypeError('at least one peer count is required')
    return values


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--peers', type=_peers, default=(1, 2, 4))
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--steps-per-round', type=int, default=8,
                   help='LOCAL PT updates per peer per round; total updates = this * peers')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--context', type=int, default=16)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--kind', choices=['loha', 'lora'], default='loha')
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--train-sequences', type=int, default=512)
    p.add_argument('--valid-sequences', type=int, default=96)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    p.add_argument('--rtt-ms', type=float, default=50., help='Round-trip propagation estimate (ms); each message has RTT/2')
    p.add_argument('--jitter-ms', type=float, default=5., help='Stddev of one-way Gaussian propagation jitter (ms)')
    p.add_argument('--bandwidth-mbps', type=float, default=50., help='Independent peer link bandwidth (Mbit/s)')
    p.add_argument('--server-bandwidth-mbps', type=float, default=100., help='Shared coordinator bandwidth per direction (Mbit/s)')
    p.add_argument('--payload-bytes', type=int, help='Override effective delta message payload size for network what-if experiments')
    p.add_argument('--message-overhead-bytes', type=int, default=1024, help='Emulated framing/metadata overhead per message')
    p.add_argument('--loss-rate', type=float, default=0., help='Whole-message drop probability in virtual network [0,1)')
    p.add_argument('--retry-rto-ms', type=float, default=250., help='Delay before retry of a dropped message (ms)')
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--compute-ms-per-step', type=float, help='Use fixed local compute cost instead of measured optimizer time (for reproducible network analysis)')
    p.add_argument('--aggregation-ms', type=float, help='Override measured FedSum aggregation time; defaults to zero when fixed local compute is selected')
    p.add_argument('--text-file', help='Use local <=8MiB UTF-8 corpus, modeling UTF-8 bytes instead of synthetic tokens')
    p.add_argument('--scratch-dir', help='Directory for temporary per-peer factor/optimizer checkpoints; cleaned after run')
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--output-dir', default='benchmark-results/serial-pt-v112')
    a = vars(p.parse_args(argv))
    output_dir = a.pop('output_dir')
    cfg = SerialPTConfig(**a)
    report = run_serial_pt_benchmark(cfg, output_dir)
    print(json.dumps({'output_dir': output_dir, 'network': report['network'], 'summary': report['summary']}, indent=2))
    return report


if __name__ == '__main__':
    main()
