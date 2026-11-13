# KCIA v131: TURN/TCP selection for Colab

This is a focused diagnostic / client-only change from KCIA v130. The relay service remains authenticated internally; no server secrets are printed.

A signaling preflight can pass while TURN allocation still fails. The KCIA server advertises both UDP and TCP TURN URLs, but aiortc's ICE gathering uses only its first selected TURN server. When UDP egress is restricted, force TCP for the client-to-TURN leg. Note the TURN-allocated relay transport is UDP from coturn to the remote peer.

1. On Colab run: `python scripts/check_turn_ice.py --transport tcp` (if you already downloaded the diagnostic). A result containing `relay` proves TURN allocation, not end-to-end P2P.
2. On Colab for P2P: `python scripts/check_wan_p2p.py connect --turn-only --turn-transport tcp --key private/colab.pem --ws wss://signal.kcia.tech/ws --room-id YOUR_ROOM_ID --remote-public-key MAC_PUBLIC_KEY --timeout 180`.
3. On Mac use the same command with its own key, the other's public key, and `--turn-transport udp` (or TCP if Mac UDP is restricted).
4. For Colab training after signed P2P passes: set `KCIA_TURN_TRANSPORT=tcp` in the Python process environment before launching `python scripts/start_wan_pt.py run --role colab ...` (e.g. `os.environ['KCIA_TURN_TRANSPORT']='tcp'` in Colab, or `KCIA_TURN_TRANSPORT=tcp python ...` in shell). For Mac leave default `auto` unless UDP is blocked.

The VPS must allow TCP/UDP 3478 and relay UDP 49160-49200, both at OS and provider firewall. The TURN DNS entry must be DNS-only (not Cloudflare-proxied). Inspect `docker compose --env-file deploy/vps/.env -f deploy/vps/compose.yml logs --tail=100 coturn` while probing. Never disclose TURN_SECRET or TURN passwords. If both UDP and TCP fail to obtain a relay candidate, test DNS/port 3478, coturn state, secret matching and server quota before training.
