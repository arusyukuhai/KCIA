"""Owner-signed, immutable, authenticated bootstrap for 1..3 peers.

One peer certifies with its own durable signature (1/1); two need both (2/2),
three need a majority (2/3). Bootstrap must never be recomputed from transient
online presence after it has been signed. Existing 3-peer rosters stay pinned.
An eventual fourth peer is an ordinary participant, not a new voter.

This is crash-fault trust with pinned identities, NOT Byzantine consensus.
The owner's solo authority is explicit and remains solo until a separately
certified membership-reconfiguration protocol exists.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (Encoding, PublicFormat,
    load_pem_private_key, load_pem_public_key)

from .core import PatchError, _canonical_json
from .roomid import id_from_raw_key

PROTOCOL_V88 = 'kagami-automatic-committee-v88'
PROTOCOL = 'kagami-automatic-committee-v90'


def bootstrap_ready(*, online_count: int, elapsed: float, stable: float) -> bool:
    """Bootstrap only after authenticated discovery settles.

    Offline presence does not change an already signed roster; only the
    initial, owner-authorized bootstrap may use the observed peer count.
    """
    if type(online_count) is not int or not 1 <= online_count <= 31:
        return False
    return online_count >= 3 or (elapsed >= 20.0 and stable >= 12.0)



def _owner_key(owner_b64: str):
    try:
        raw = base64.b64decode(owner_b64, validate=True)
        if len(raw) != 32 or base64.b64encode(raw).decode('ascii') != owner_b64:
            raise ValueError('noncanonical public key')
        return Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, TypeError) as exc:
        raise PatchError('invalid immutable room generation owner') from exc


def validate_owner(manifest):
    config = manifest.get('generation')
    if not isinstance(config, dict) or set(config) != {'owner'}:
        raise PatchError('generation owner must be immutable in room manifest')
    if not isinstance(config['owner'], str):
        raise PatchError('invalid room generation owner')
    pub = _owner_key(config['owner'])
    return pub


def owner_for_local_room(path: Path):
    """Provision a key *only for a new, owner-created GUI room*.

    A re-imported room never receives a locally invented owner.  The key is not
    transmitted with the manifest, and must remain mode 0600.
    """
    from cryptography.hazmat.primitives.serialization import PrivateFormat, NoEncryption
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise PatchError('refusing to replace an existing room owner private key')
    path.parent.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    raw = priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'wb') as fp:
            fp.write(raw); fp.flush(); os.fsync(fp.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return base64.b64encode(priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode('ascii')


def _owner_private(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
        return None
    key = load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise PatchError('owner identity is not Ed25519')
    if key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw) != expected.public_bytes(Encoding.Raw, PublicFormat.Raw):
        raise PatchError('owner private key disagrees with immutable room manifest')
    return key


def _validate_wire(wire, *, room, root_genesis, manifest_sha, owner):
    if not isinstance(wire, dict) or set(wire) != {'payload', 'signature'}:
        raise PatchError('invalid automatic committee envelope')
    payload = wire['payload']
    if not isinstance(payload, dict) or set(payload) != {'protocol', 'room', 'genesis', 'manifest', 'keys'}:
        raise PatchError('invalid automatic committee payload')
    if (payload['protocol'] not in (PROTOCOL_V88, PROTOCOL) or payload['room'] != room or
        payload['genesis'] != root_genesis or payload['manifest'] != manifest_sha):
        raise PatchError('automatic committee is bound to a different room/genesis/manifest')
    keys = payload['keys']
    if (not isinstance(keys, dict) or not 1 <= len(keys) <= 3 or
            (payload['protocol'] == PROTOCOL_V88 and len(keys) != 3)):
        raise PatchError('automatic committee requires 1..3 distinct authenticated identities')
    from .generation_committee_v43 import read_generation_committee
    # Validate in memory; forbid arbitrary file paths and PEM substitutions.
    normalized = {}
    for peer, pem in keys.items():
        if not isinstance(peer, str) or not isinstance(pem, str) or len(pem) > 4096:
            raise PatchError('malformed committee entry')
        try:
            pub = load_pem_public_key(pem.encode('ascii'))
            if not isinstance(pub, Ed25519PublicKey):
                raise PatchError('non-Ed25519 committee key')
            if id_from_raw_key(pub.public_bytes(Encoding.Raw, PublicFormat.Raw)) != peer:
                raise PatchError('key-derived peer ID mismatch')
            normalized[peer] = pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode('ascii')
        except (UnicodeError, ValueError, TypeError) as exc:
            raise PatchError('malformed authenticated committee public key') from exc
    if normalized != keys:
        raise PatchError('committee must use canonical Ed25519 PEM keys')
    if not isinstance(wire['signature'], str) or len(wire['signature']) > 256:
        raise PatchError('invalid owner signature encoding')
    try:
        from cryptography.exceptions import InvalidSignature
        owner.verify(base64.b64decode(wire['signature'], validate=True), _canonical_json(payload))
    except (ValueError, InvalidSignature) as exc:
        raise PatchError('automatic committee is not signed by the pinned room owner') from exc
    if len(_canonical_json(wire)) > 4096:
        raise PatchError('committee bootstrap frame exceeds 4096 bytes')
    return keys


def _atomic_write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PatchError('unsafe symlink in automatic committee state')
    tmp = path.with_name('.' + path.name + '.' + secrets.token_hex(8))
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'wb') as fp:
            fp.write(raw); fp.flush(); os.fsync(fp.fileno())
        os.replace(tmp, path)
        if hasattr(os, 'O_DIRECTORY'):
            dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
    finally:
        tmp.unlink(missing_ok=True)


class AutoCommittee:
    def __init__(self, *, room, genesis, manifest_sha, manifest, journal, manifest_path):
        self.room, self.genesis, self.manifest_sha = room, genesis, manifest_sha
        self.owner = validate_owner(manifest)
        self.private = _owner_private(Path(manifest_path).parent / 'generation-owner.pem', self.owner)
        self.path = Path(str(journal) + '.generation-auto-v88.json')
        self.committee_path = Path(str(journal) + '.generation-autopinned-v88.json')
        self.wire = None
        if self.path.exists() or self.path.is_symlink():
            if not self.path.is_file() or self.path.is_symlink() or self.path.stat().st_size > 4096:
                raise PatchError('corrupt automatic committee bootstrap')
            self.accept(json.loads(self.path.read_bytes()))
        elif self.committee_path.exists():
            raise PatchError('automatic committee keys without signed owner roster')

    def accept(self, wire):
        keys = _validate_wire(wire, room=self.room, root_genesis=self.genesis,
                              manifest_sha=self.manifest_sha, owner=self.owner)
        raw = _canonical_json(wire) + b'\n'
        if self.wire is not None:
            if _canonical_json(self.wire) != _canonical_json(wire):
                raise PatchError('conflicting signed automatic committee: refusing a fork')
            return False
        if self.path.exists() and self.path.read_bytes() != raw:
            raise PatchError('signed committee changed on disk')
        if not self.path.exists():
            _atomic_write(self.path, raw)
        expected = _canonical_json({'keys': keys}) + b'\n'
        if self.committee_path.exists():
            if self.committee_path.is_symlink() or self.committee_path.read_bytes() != expected:
                raise PatchError('local pinned committee differs from signed roster')
        else:
            _atomic_write(self.committee_path, expected)
        self.wire = wire
        return True

    def try_create(self, *, myself, peer_pems):
        """Owner signs one immutable roster of 1..3 authenticated peers.

        Call after discovery settles so a two-peer room doesn't accidentally
        pin a solo committee at the moment the first GUI starts. A previously
        signed roster can neither silently grow nor shrink upon disconnect.
        """
        if self.wire is not None or self.private is None:
            return False
        peers = sorted(peer_pems)
        if myself not in peers or not 1 <= len(peers) <= 31:
            return False
        selected = [myself, *[p for p in peers if p != myself][:2]]
        keys = {peer: peer_pems[peer].decode('ascii') for peer in sorted(selected)}
        payload = {'protocol': PROTOCOL_V88 if len(keys) == 3 else PROTOCOL,
                   'room': self.room, 'genesis': self.genesis,
                   'manifest': self.manifest_sha, 'keys': keys}
        wire = {'payload': payload, 'signature': base64.b64encode(
            self.private.sign(_canonical_json(payload))).decode('ascii')}
        return self.accept(wire)
