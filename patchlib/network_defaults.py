"""KCIA network destinations. WAN is production; loopback is explicit debug only.

The verified live service is signal.kcia.tech on the kcia.tech domain.
Do not substitute wss://kcia.tech/ws until that apex endpoint is deployed.
"""
WAN_SIGNALING_URL = 'wss://signal.kcia.tech/ws'
LOCAL_DEBUG_SIGNALING_URL = 'ws://127.0.0.1:8080/ws'
