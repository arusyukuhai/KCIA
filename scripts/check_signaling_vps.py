#!/usr/bin/env python3
"""WSS signaling + TURN credentials preflight. No model or keys needed."""
from __future__ import annotations
import argparse
import asyncio
import json
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def _health_url(ws: str) -> str:
    parsed = urlsplit(ws)
    if parsed.scheme not in {'ws','wss'} or parsed.path != '/ws' or not parsed.hostname:
        raise ValueError('--ws must be ws(s)://HOST/ws')
    return urlunsplit(('https' if parsed.scheme=='wss' else 'http',parsed.netloc,'/healthz','',''))


async def probe(ws: str, *, timeout: float=12.0, expect_turn: bool=True) -> None:
    import secrets
    import websockets
    room = 'kcia-probe-' + secrets.token_hex(14)
    health = _health_url(ws)
    # urllib is synchronous; execute off-loop so a local test server can respond.
    def request_health():
        req = Request(health, headers={'User-Agent': 'KCIA-WAN-Pilot/1.0'})
        with urlopen(req,timeout=timeout) as response:
            if response.status != 200 or json.load(response).get('ok') is not True:
                raise RuntimeError('unhealthy signaling server')
    await asyncio.to_thread(request_health)
    opts={'open_timeout':timeout,'max_size':200_000,'close_timeout':3,
          'user_agent_header':'KCIA-WAN-Pilot/1.0'}
    async with websockets.connect(ws, **opts) as a:
        await a.send(json.dumps({'type':'join','mode':'p2p','room':room}))
        first=json.loads(await asyncio.wait_for(a.recv(),timeout))
        if first.get('type')!='joined' or first.get('role')!='initiator':
            raise RuntimeError(f'first peer rejected: {first.get("type")} / {first.get("message")}')
        ices=first.get('iceServers',[])
        if not isinstance(ices,list): raise RuntimeError('ICE server list missing')
        if expect_turn and not any('turn:' in str(item.get('urls','')) for item in ices):
            raise RuntimeError('TURN credentials not returned: check TURN_SECRET and TURN_HOST')
        async with websockets.connect(ws, **opts) as b:
            await b.send(json.dumps({'type':'join','mode':'p2p','room':room}))
            second=json.loads(await asyncio.wait_for(b.recv(),timeout))
            joined=json.loads(await asyncio.wait_for(a.recv(),timeout))
            if second.get('role')!='answerer' or joined.get('type')!='peer-joined':
                raise RuntimeError('pair signaling failed')
            # Verify that SDP messages are forwarded; no secret SDP needed.
            demo='v=0\\r\\ns=-\\r\\n'
            await a.send(json.dumps({'type':'offer','sdp':demo}))
            forwarded=json.loads(await asyncio.wait_for(b.recv(),timeout))
            if forwarded.get('type')!='offer' or forwarded.get('sdp')!=demo:
                raise RuntimeError('SDP forwarding failed')
    print('PASS: HTTPS health, public WSS, two peer admission, SDP forwarding')
    print('PASS: TURN credentials advertised' if expect_turn else 'TURN advertisement not required')
    print('NOTE: this does NOT prove ICE relay, data synchronization or model training')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ws',required=True)
    ap.add_argument('--timeout',type=float,default=12)
    ap.add_argument('--no-require-turn',action='store_true')
    a=ap.parse_args()
    asyncio.run(probe(a.ws,timeout=a.timeout,expect_turn=not a.no_require_turn))

if __name__ == '__main__':main()
