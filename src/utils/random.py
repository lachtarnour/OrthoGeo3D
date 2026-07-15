import hashlib
from typing import Any


def stable_seed(seed: int, *parts: Any) -> int:
    text = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % (2**63 - 1)
