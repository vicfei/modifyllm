"""ULID-style identifiers: 48-bit ms timestamp + 80-bit randomness, Crockford base32."""

from __future__ import annotations

import os
import time

_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id(prefix: str) -> str:
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big") & ((1 << 80) - 1)
    value = (ms << 80) | rand
    chars = []
    for _ in range(26):
        chars.append(_ENCODING[value & 31])
        value >>= 5
    return f"{prefix}_{''.join(reversed(chars))}"
