"""JuniorStock surface compute. Ticket only. Not an order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "stock_surface.jsonl"


def compute(note: str = "stock") -> dict:
    xs = [((ord(c) % 5) - 2) / 2.0 for c in note[:16]]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = [1 if round(x / gamma) > 1 else (-1 if round(x / gamma) < -1 else int(round(x / gamma))) for x in xs]
    body = {
        "protocol": "goldend-osai-omega/1",
        "port": "JuniorStock",
        "trits": trits,
        "sha3": hashlib.sha3_256(note.encode()).hexdigest()[:16],
        "order": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sha3": body["sha3"], "port": "JuniorStock"}) + "\n")
    return body


if __name__ == "__main__":
    print(json.dumps(compute(), indent=2))
