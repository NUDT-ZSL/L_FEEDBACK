from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: check_binary_determinism.py RUN_DIR_A RUN_DIR_B", file=sys.stderr)
        return 2
    first = Path(argv[1])
    second = Path(argv[2])
    names = ["processed.pcm", "radio.wav", "radio.ogg"]
    mismatches = []
    for name in names:
        left = first / name
        right = second / name
        if not left.is_file() or not right.is_file():
            mismatches.append(f"missing {name}")
            continue
        left_hash = digest(left)
        right_hash = digest(right)
        if left_hash != right_hash:
            mismatches.append(f"{name}: {left_hash} != {right_hash}")
        else:
            print(f"{name}: {left_hash}")
    left_input = digest(first / "logs" / "input_packets.jsonl")
    right_input = digest(second / "logs" / "input_packets.jsonl")
    if left_input != right_input:
        mismatches.append("input_packets.jsonl differs")
    else:
        print(f"input_packets.jsonl: {left_input}")
    result = {"deterministic": not mismatches, "mismatches": mismatches}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not mismatches else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
