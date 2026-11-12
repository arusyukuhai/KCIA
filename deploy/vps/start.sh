#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONF="$ROOT/deploy/vps"
if [[ ! -f "$CONF/.env" ]]; then
    echo 'Missing .env: run deploy/vps/setup_env.py with --signal-host, --turn-host, --public-ip.' >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo 'Docker Compose v2 required.' >&2
    exit 1
fi
python3 - "$CONF/.env" <<'PY'
from pathlib import Path
import os, stat, sys
p=Path(sys.argv[1]); mode=p.stat().st_mode
if mode & 0o077:
    raise SystemExit('.env must have permissions 0600 (chmod 600 deploy/vps/.env)')
env=dict(line.split('=',1) for line in p.read_text().splitlines() if line and not line.startswith('#') and '=' in line)
for key in ('CF_TUNNEL_TOKEN','APP_TOKEN','TURN_SECRET','ROOM_PUBLISH_TOKEN'):
    value=env.get(key,'')
    if not value or any(word in value.upper() for word in ('PASTE_', 'REPLACE_', 'YOUR_')):
        raise SystemExit(f'Set real {key} in deploy/vps/.env before starting')
if env.get('SIGNAL_HOST')==env.get('TURN_HOST'):
    raise SystemExit('SIGNAL_HOST and TURN_HOST must be distinct')
PY
cd "$ROOT"
docker compose --env-file "$CONF/.env" -f "$CONF/compose.yml" config --quiet
docker compose --env-file "$CONF/.env" -f "$CONF/compose.yml" up --detach --build
docker compose --env-file "$CONF/.env" -f "$CONF/compose.yml" ps
printf '\nSignaling (Cloudflare Tunnel): wss://%s/ws\n' "$(sed -n 's/^SIGNAL_HOST=//p' "$CONF/.env")"
echo 'STUN/TURN connects directly to TURN_HOST (Cloudflare DNS only, NOT proxied).'
echo 'Next: python scripts/check_signaling_vps.py --ws wss://SIGNAL_HOST/ws --token YOUR_APP_TOKEN'
