#!/usr/bin/env python3
"""KCIA WAN pilot preflight. Never reads/signals local private keys or secrets."""
from __future__ import annotations
import argparse
import asyncio
import importlib.util
import ipaddress
import json
from pathlib import Path
import platform
import socket
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def signal_host(value: str) -> str:
    p = urlsplit(value)
    if p.scheme != 'wss' or not p.hostname or p.path != '/ws' or p.username or p.password or p.query or p.fragment:
        raise ValueError('Public pilot signaling must be wss://HOST/ws (no credentials/query)')
    if p.hostname in ('localhost', '127.0.0.1'):
        raise ValueError('Use the Cloudflare published WSS hostname for WAN tests')
    return p.hostname


def check_dns_only_turn(host: str) -> dict:
    # This is advisory, not proof of Cloudflare Proxy status; the operator must
    # check the orange/grey cloud in Cloudflare DNS dashboard.
    info = socket.getaddrinfo(host, 3478, type=socket.SOCK_DGRAM)
    addresses = sorted({item[4][0] for item in info})
    return {'host': host, 'addresses': addresses,
            'note': 'Verify Cloudflare DNS record is DNS only (grey cloud); DNS lookup alone cannot prove this.'}


def local_report(role: str) -> dict:
    dependencies = ['cryptography', 'websockets']
    if role in ('mac','colab'):
        dependencies += ['aiortc', 'torch', 'safetensors', 'transformers', 'huggingface_hub', 'datasets']
    installed = {name: importlib.util.find_spec(name) is not None for name in dependencies}
    report = {'role':role,'python':platform.python_version(), 'platform':platform.platform(),
              'dependencies':installed, 'missing': [name for name, ok in installed.items() if not ok],
              'project_files': {'cli':(ROOT/'patchlib/cli.py').exists(),
                                'signaling_probe':(ROOT/'scripts/check_signaling_vps.py').exists(),
                                'wan_handshake':(ROOT/'scripts/check_wan_p2p.py').exists(),
                                'model_text':(ROOT/'examples/wan_sample.txt').exists()}}
    if installed.get('torch'):
        # Avoid importing heavyweight torch unless the user asks for local diagnostics.
        try:
            import torch
            report['torch'] = torch.__version__
            report['cuda_available'] = bool(torch.cuda.is_available())
            report['mps_available'] = bool(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())
        except Exception as exc:
            report['torch_error'] = str(exc)[:200]
    if role == 'colab' and not report.get('cuda_available', False):
        report['gpu_note'] = 'CUDA unavailable: enable a GPU runtime in Colab or use --device cpu.'
    if role == 'mac' and not report.get('mps_available', False):
        report['gpu_note'] = 'MPS unavailable: CPU smoke test remains possible.'
    if not ((3,11) <= sys.version_info[:2] <= (3,13)):
        report['python_note'] = 'Pilot documentation expects Python 3.11-3.13.'
    return report


async def run(args: argparse.Namespace) -> int:
    report=local_report(args.role)
    if args.ws:
        report['signal_host']=signal_host(args.ws)
    if args.turn_host:
        if args.ws and args.turn_host.lower()==report['signal_host'].lower():
            raise ValueError('TURN host must differ from signaling hostname')
        if args.network:
            try: report['turn_dns']=check_dns_only_turn(args.turn_host)
            except OSError as exc: report['turn_dns_error']=str(exc)
    if args.network:
        if not args.ws: raise ValueError('--network requires --ws')
        if not importlib.util.find_spec('websockets'):
            report['network_error']='Install websockets first'
        else:
            from scripts.check_signaling_vps import probe
            try:
                await probe(args.ws, timeout=args.timeout)
                report['signaling_pass']=True
            except Exception as exc:
                report['signaling_pass']=False
                report['network_error']=f'{type(exc).__name__}: {exc}'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing=bool(report['missing']) if args.require_deps else False
    failed=(args.network and report.get('signaling_pass') is not True)
    bad_dns=args.network and bool(report.get('turn_dns_error'))
    return 1 if (missing or failed or bad_dns or not all(report['project_files'].values())) else 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--role', choices=['mac','colab','vps'], required=True)
    ap.add_argument('--ws', help='Cloudflare WSS endpoint, e.g. wss://signal.example.com/ws')
    ap.add_argument('--turn-host', help='Unproxied TURN hostname (DNS only)')
    ap.add_argument('--network', action='store_true', help='Run real WSS/health/SDP signaling probe')
    ap.add_argument('--require-deps', action='store_true', help='Nonzero exit when requirements are absent')
    ap.add_argument('--timeout', type=float, default=15.0)
    a=ap.parse_args()
    try: raise SystemExit(asyncio.run(run(a)))
    except ValueError as exc: ap.error(str(exc))

if __name__=='__main__':main()
