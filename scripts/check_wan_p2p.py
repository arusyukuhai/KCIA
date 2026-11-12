#!/usr/bin/env python3
"""Two-MACHINE signed WebRTC handshake test, Mac <-> Colab, without loading a model.

Identity setup on each host:
  python scripts/check_wan_p2p.py identity --key ./private/mac.pem
  python scripts/check_wan_p2p.py identity --key ./private/colab.pem
Exchange only the printed public key through a trusted channel. Share a freshly
randomized --room-id, and run connect at BOTH ends (either can start first).
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patchlib.roomid import auto_identity, id_from_raw_key, pair_room, validate_room


def public_key_info(key: Path):
    peer,signer,pub,verify=auto_identity(key)
    return peer,signer,pub,verify


async def connect(args):
    from patchlib.elasticnet import ElasticNode
    local,signer,public,_=public_key_info(args.key)
    try:
        remote_bytes=base64.b64decode(args.remote_public_key,validate=True)
        remote=id_from_raw_key(remote_bytes)
    except (ValueError,TypeError) as exc:
        raise ValueError('remote public key must be exactly 32 Ed25519 bytes in base64') from exc
    if local==remote: raise ValueError('remote public key is the local one')
    room=pair_room(validate_room(args.room_id),local,remote)
    node=ElasticNode(args.ws,room,args.token,origin=local,remote=remote,
                     genesis='0'*64,signer=signer,public_key=public,
                     turn_only=args.turn_only)
    task=asyncio.create_task(node.run())
    print('Connecting to the other machine (signed hello required).')
    try:
        loop=asyncio.get_running_loop()
        deadline=loop.time()+args.timeout
        while loop.time()<deadline:
            if node.connected.is_set() and node.verified_at:
                print('PASS: cross-host ICE/DTLS/SCTP and signed Ed25519 peer handshake')
                print('Local peer:',local,'Remote peer:',remote)
                print('ICE:',node.ice_state,'RTC:',node.rtc_state,
                      'TURN relay mode:',args.turn_only)
                await asyncio.sleep(min(2, max(0,deadline-loop.time())))
                if node.protocol_failure.is_set() or not node.connected.is_set():
                    raise RuntimeError('handshake succeeded but link immediately failed')
                return
            if task.done():
                exc=task.exception()
                raise RuntimeError(f'WebRTC exited: {type(exc).__name__}: {exc}' if exc else 'WebRTC exited')
            await asyncio.sleep(.2)
        raise TimeoutError(f'peer handshake timeout; phase={node.phase} ICE={node.ice_state} RTC={node.rtc_state} error={node.last_error!r}')
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest='mode',required=True)
    identity=subs.add_parser('identity',help='Create/show local persistent peer identity')
    identity.add_argument('--key',type=Path,required=True)
    conn=subs.add_parser('connect',help='Check signed peer handshake to a remote machine')
    conn.add_argument('--key',type=Path,required=True)
    conn.add_argument('--ws',required=True)
    conn.add_argument('--token',default='')
    conn.add_argument('--room-id',required=True)
    conn.add_argument('--remote-public-key',required=True)
    conn.add_argument('--turn-only',action='store_true',help='Only advertise TURN relay candidates on this endpoint')
    conn.add_argument('--timeout',type=float,default=120)
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(name)s %(message)s')
    if args.mode=='identity':
        peer,_,pub,_=public_key_info(args.key)
        print('Public peer ID:',peer)
        print('Public key (share this):',pub)
        print('Private key path (never share):',args.key)
    else:
        asyncio.run(connect(args))

if __name__=='__main__':main()
