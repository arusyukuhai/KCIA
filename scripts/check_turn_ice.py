#!/usr/bin/env python3
"""KCIA WAN diagnostic: discover signaling-advertised ICE servers, gather local ICE candidates.
Run separately on Mac and Colab. Does not load a model or establish a peer link.
Does not print TURN credentials, signing keys or candidate private addresses.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import secrets
import sys


def summarize_sdp(sdp: str) -> Counter:
    result: Counter = Counter()
    for line in sdp.splitlines():
        if not line.startswith("a=candidate:"):
            continue
        parts = line.split()
        if "typ" not in parts:
            continue
        k = parts.index("typ")
        if k + 1 < len(parts):
            result[parts[k + 1]] += 1
    return result


def servers_from_join(payload: dict, transport: str):
    from aiortc import RTCIceServer

    items = payload.get("iceServers")
    if not isinstance(items, list):
        raise RuntimeError("KCIA join response has no iceServers array")
    servers = []
    urls_summary = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_urls = item.get("urls", [])
        raw_urls = [raw_urls] if isinstance(raw_urls, str) else raw_urls
        if not isinstance(raw_urls, list):
            continue
        urls = []
        for url in raw_urls:
            if not isinstance(url, str):
                continue
            if url.startswith("stun:"):
                continue  # Force TURN-gathering visibility, no fallback STUN.
            if not url.startswith("turn:"):
                continue
            is_tcp = "transport=tcp" in url
            is_udp = "transport=udp" in url or "?" not in url
            if (transport == "tcp" and is_tcp) or (transport == "udp" and is_udp):
                urls.append(url)
        if urls:
            servers.append(RTCIceServer(urls=urls, username=item.get("username"),
                                        credential=item.get("credential")))
            urls_summary.extend(["TURN/TCP" if "transport=tcp" in u else "TURN/UDP" for u in urls])
    return servers, urls_summary


async def run(ws_url: str, timeout: float, transport: str):
    import websockets
    from aiortc import RTCPeerConnection, RTCConfiguration

    room = "kcia-turn-check-" + secrets.token_hex(16)
    print(f"CONNECT {ws_url}", flush=True)
    async with websockets.connect(ws_url, user_agent_header="KCIA-WAN-Client/1.0",
                                  open_timeout=15, close_timeout=5,
                                  ping_interval=30, ping_timeout=150,
                                  max_size=200_000) as ws:
        await ws.send(json.dumps({"type": "join", "mode": "p2p", "room": room}))
        joined = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        if joined.get("type") != "joined":
            raise RuntimeError("Signaling join was not accepted: " + str(joined.get("type")))
        servers, kinds = servers_from_join(joined, transport)
        print("TURN advertised:", ", ".join(kinds) if kinds else "NONE", flush=True)
        if not servers:
            raise RuntimeError("TURN URLs missing; check signaling TURN_HOST / TURN_SECRET")

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
        try:
            pc.createDataChannel("kcia-ice-preflight")
            print(f"GATHER {transport.upper()} TURN candidates (limit {timeout:g}s)", flush=True)
            await asyncio.wait_for(pc.setLocalDescription(await pc.createOffer()), timeout=timeout)
            counts = summarize_sdp(pc.localDescription.sdp if pc.localDescription else "")
            print("ICE candidate types:", dict(counts), flush=True)
            if counts["relay"]:
                print("PASS: TURN relay candidate allocated on this machine.", flush=True)
                print("NOTE: This does not yet prove relay port reachability to the other peer.")
            else:
                print("FAIL: no relay candidate; check coturn auth, reachability and logs.", flush=True)
                raise SystemExit(2)
        finally:
            await pc.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ws", default="wss://signal.kcia.tech/ws")
    ap.add_argument("--transport", choices=["udp", "tcp"], default="udp",
                    help="TURN client-to-server transport to test; relay candidate itself uses UDP")
    ap.add_argument("--timeout", type=float, default=45)
    a = ap.parse_args()
    try:
        asyncio.run(run(a.ws, a.timeout, a.transport))
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
