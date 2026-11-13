"""TURN TCP selection regression tests (no external network)."""
import os
import unittest
from unittest.mock import patch

from patchlib.p2p import PatchNode, _select_turn_transport


SAMPLE = [
    {"urls": "stun:turn.kcia.tech:3478"},
    {"urls": ["turn:turn.kcia.tech:3478?transport=udp",
              "turn:turn.kcia.tech:3478?transport=tcp"],
     "username": "example", "credential": "example-secret"},
]


class TestTurnSelection(unittest.TestCase):
    def test_tcp_selection_drops_udp(self):
        out = _select_turn_transport(SAMPLE, 'tcp')
        self.assertEqual(out[1]['urls'], ["turn:turn.kcia.tech:3478?transport=tcp"])
        self.assertEqual(out[0]['urls'], ["stun:turn.kcia.tech:3478"])
        self.assertEqual(len(SAMPLE[1]['urls']), 2)
        self.assertEqual(out[1]['credential'], "example-secret")

    def test_udp_selection_drops_tcp(self):
        out = _select_turn_transport(SAMPLE, 'udp')
        self.assertEqual(out[1]['urls'], ["turn:turn.kcia.tech:3478?transport=udp"])

    def test_auto_preserves_advertisement(self):
        self.assertIs(_select_turn_transport(SAMPLE, 'auto'), SAMPLE)

    def test_missing_url_fails_explicitly(self):
        with self.assertRaisesRegex(ConnectionError, "TURN/tcp"):
            _select_turn_transport(SAMPLE[:1], 'tcp')

    def test_invalid_transport_fails_closed(self):
        with self.assertRaises(ValueError):
            _select_turn_transport(SAMPLE, 'sctp')

    def test_env_selects_transport_for_training_nodes(self):
        with patch.dict(os.environ, {"KCIA_TURN_TRANSPORT": "tcp"}):
            n = PatchNode("wss://signal.kcia.tech/ws", "A"*32)
            self.assertEqual(n.turn_transport, "tcp")
        n = PatchNode("wss://signal.kcia.tech/ws", "A"*32, turn_transport="udp")
        self.assertEqual(n.turn_transport, "udp")


if __name__ == '__main__':
    unittest.main()
