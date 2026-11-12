#!/usr/bin/env python3
"""Generate VPS secrets once. Cloudflare Tunnel token must be added privately."""
import argparse
import ipaddress
import os
from pathlib import Path
import re
import secrets

ROOT = Path(__file__).resolve().parent
DOMAIN_RE = re.compile(r'(?=.{4,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[A-Za-z]{2,63}\Z')
PLACEHOLDER = 'PASTE_CLOUDFLARE_TUNNEL_TOKEN_HERE'


def configuration(signal_host: str, turn_host: str, public_ip: str) -> str:
    for name, host in [('signal-host',signal_host),('turn-host',turn_host)]:
        if not DOMAIN_RE.fullmatch(host):
            raise ValueError(f'{name} must be a public DNS hostname, e.g. {name.split("-")[0]}.example.com')
    if signal_host.lower() == turn_host.lower():
        raise ValueError('signal-host and turn-host must differ: TURN uses a DNS-only record')
    address = ipaddress.ip_address(public_ip)
    if address.version != 4 or not address.is_global:
        raise ValueError('PUBLIC_IP must be a globally routable IPv4 address')
    entries = {
        'SIGNAL_HOST': signal_host.lower(),
        'TURN_HOST': turn_host.lower(),
        'TURN_REALM': turn_host.lower(),
        'PUBLIC_IP': public_ip,
        'CF_TUNNEL_TOKEN': PLACEHOLDER,
        'TURN_SECRET': secrets.token_hex(32),
    }
    return ''.join(f'{key}={value}\n' for key, value in entries.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signal-host', required=True)
    parser.add_argument('--turn-host', required=True)
    parser.add_argument('--public-ip', required=True)
    args = parser.parse_args()
    path = ROOT / '.env'
    data = configuration(args.signal_host, args.turn_host, args.public_ip)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print(f'Created {path} with 0600 permissions; NEVER commit/share it.')
    print(f'In Cloudflare create a remotely-managed Tunnel, public hostname {args.signal_host}: http://signal:8080')
    print('Paste its TUNNEL_TOKEN into CF_TUNNEL_TOKEN in .env (do not share the token).')
    print(f'Create a separate DNS-only A record {args.turn_host} -> {args.public_ip}')
    print(f'Signaling URL: wss://{args.signal_host}/ws')


if __name__ == '__main__':
    main()
