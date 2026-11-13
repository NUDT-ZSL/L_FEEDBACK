"""Deterministic before/after PE-style sample generator.

Usage:
    python -m pe_audit.generator --seed 7 --out demo

The generator writes both binaries and a small, audit-ready metadata set:
``patch_before.bin``, ``patch_after.bin``, ``before_manifest.json``,
``after_manifest.json``, ``entries.json`` and ``defects.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

from .binary import (
    ImportSymbol,
    Instruction,
    OP_JCC,
    build_fixture_image,
    call_import,
    call_local,
    cmp_imm,
    jcc,
    nop,
    pad,
    push_imm,
    ret,
)


MASK64 = (1 << 64) - 1

FUNCTIONS = (
    "dispatch",
    "entry_main",
    "guarded_copy",
    "isolated",
    "orphan_copy",
    "unsafe_copy",
    "walk_a",
    "walk_b",
)

IMPORTS = (
    ImportSymbol("kernel32.dll", "ExitProcess"),
    ImportSymbol("msvcrt.dll", "memcpy"),
    ImportSymbol("msvcrt.dll", "strcpy"),
    ImportSymbol("msvcrt.dll", "strncpy"),
)


def _mix_seed(seed: int, salt: int) -> int:
    """SplitMix64, avoiding Python's process-randomized ``hash``."""

    value = (seed & MASK64) ^ (salt & MASK64)
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def _rng(seed: int, version: str) -> random.Random:
    salt = 0xB3F0E17 if version == "before" else 0xA6E47C2
    return random.Random(_mix_seed(seed, salt))


def _insert_padding(
    program: Sequence[Instruction], rng: random.Random
) -> list[Instruction]:
    result: list[Instruction] = []
    for instruction in program:
        for _ in range(rng.randrange(0, 4)):
            result.append(pad())
        result.append(instruction)
    for _ in range(rng.randrange(0, 4)):
        result.append(pad())
    return result


def _serialize_program(instructions: Sequence[Instruction]) -> bytes:
    """Serialize and resolve conditional branch targets to function end.

    Branch targets are offsets only for local control-flow interpretation.
    They are intentionally not signatures or call-graph identities.
    """

    total = len(instructions) * 4
    parts: list[bytes] = []
    for index, instruction in enumerate(instructions):
        if instruction.opcode == OP_JCC:
            assert instruction.branch_number is not None
            parts.append(
                jcc(instruction.branch_number, total - index * 4).encode()
            )
        else:
            parts.append(instruction.encode())
    return b"".join(parts)


def _programs(version: str) -> dict[str, list[Instruction]]:
    local_index = {name: index for index, name in enumerate(sorted(FUNCTIONS))}
    imports = sorted(IMPORTS, key=lambda item: (item.module, item.name))
    import_index = {
        (item.module, item.name): index for index, item in enumerate(imports)
    }

    def local(name: str) -> Instruction:
        return call_local(local_index[name])

    def imported(module: str, name: str) -> Instruction:
        return call_import(import_index[(module, name)])

    common_calls = [
        push_imm(0x11),
        cmp_imm(0x01),
        local("dispatch"),
        local("walk_a"),
        ret(),
    ]
    dispatch = [
        push_imm(0x22),
        local("unsafe_copy"),
        local("guarded_copy"),
        ret(),
    ]
    walk_a = [local("walk_b"), ret()]
    walk_b = [local("walk_a"), ret()]
    guarded = [
        push_imm(0x10),
        cmp_imm(0x08),
        jcc(0),
        imported("msvcrt.dll", "strncpy"),
        ret(),
    ]
    isolated = [push_imm(0x13 if version == "before" else 0x31), nop(), ret()]
    orphan = [imported("msvcrt.dll", "strcpy"), ret()]

    if version == "before":
        unsafe = [
            push_imm(0x20),
            imported("msvcrt.dll", "strcpy"),
            push_imm(0x40),
            imported("msvcrt.dll", "memcpy"),
            ret(),
        ]
    else:
        unsafe = [
            push_imm(0x40),
            cmp_imm(0x08),
            jcc(0),
            imported("msvcrt.dll", "strncpy"),
            ret(),
        ]
    return {
        "dispatch": dispatch,
        "entry_main": common_calls,
        "guarded_copy": guarded,
        "isolated": isolated,
        "orphan_copy": orphan,
        "unsafe_copy": unsafe,
        "walk_a": walk_a,
        "walk_b": walk_b,
    }


def _layout(
    seed: int, version: str, bodies: dict[str, bytes], layout_salt: int = 0
) -> tuple[dict[str, int], dict[str, int], int]:
    rng = _rng(seed ^ layout_salt, version)
    order = sorted(bodies)
    rng.shuffle(order)
    base = 0x400 + rng.randrange(0, 17) * 0x10
    offsets: dict[str, int] = {}
    gaps: dict[str, int] = {}
    cursor = base
    for name in order:
        cursor = (cursor + 0x0F) // 0x10 * 0x10
        offsets[name] = cursor
        gaps[name] = cursor
        cursor += len(bodies[name])
        cursor += rng.randrange(0, 3) * 0x10
    return offsets, gaps, base


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _manifest(
    *,
    version: str,
    seed: int,
    offsets: dict[str, int],
    bodies: dict[str, bytes],
) -> dict[str, object]:
    from .binary import SECTION_RVA

    functions = [
        {
            "name": name,
            "rva": SECTION_RVA + offsets[name],
            "length": len(bodies[name]),
        }
        for name in sorted(bodies)
    ]
    return {
        "schema_version": 1,
        "version": version,
        "seed": seed,
        "entry_points": ["entry_main"],
        "functions": functions,
    }


def default_defects() -> dict[str, object]:
    """Return a fixture that exercises all three audit conclusions."""

    return {
        "schema_version": 1,
        "entry_points": ["entry_main"],
        "rules": [
            {
                "id": "D1",
                "target": "unsafe_copy",
                "dangerous_imports": [
                    {"module": "msvcrt.dll", "name": "strcpy"}
                ],
                "rule": {"type": "guarded_call", "branch_number": 0},
            },
            {
                "id": "D2",
                "target": "orphan_copy",
                "dangerous_imports": [
                    {"module": "msvcrt.dll", "name": "strcpy"}
                ],
                "rule": {"type": "forbid_call"},
            },
            {
                "id": "D3",
                "target": "guarded_copy",
                "dangerous_imports": [
                    {"module": "msvcrt.dll", "name": "strncpy"}
                ],
                "rule": {"type": "forbid_call"},
            },
        ],
    }


def generate_samples(seed: int, out: str | Path) -> dict[str, object]:
    """Generate both images and metadata, returning a path-free summary."""

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    local_names = sorted(FUNCTIONS)
    imports = sorted(IMPORTS, key=lambda item: (item.module, item.name))
    before_rng = _rng(seed, "before")
    after_rng = _rng(seed, "after")
    before_programs = _programs("before")
    after_programs = _programs("after")
    before_bodies = {
        name: _serialize_program(
            _insert_padding(program, before_rng)
        )
        for name, program in before_programs.items()
    }
    after_bodies = {
        name: _serialize_program(_insert_padding(program, after_rng))
        for name, program in after_programs.items()
    }
    before_offsets, _before_gaps, _ = _layout(seed, "before", before_bodies)
    after_offsets, _after_gaps, _ = _layout(seed, "after", after_bodies)

    # The sample contract explicitly requires a different layout between
    # versions. Re-layout with a deterministic salt if two shuffles coincide.
    before_order = sorted(before_offsets, key=lambda name: before_offsets[name])
    after_order = sorted(after_offsets, key=lambda name: after_offsets[name])
    for layout_attempt in range(1, 33):
        if before_order != after_order:
            break
        after_offsets, _after_gaps, _ = _layout(
            seed, "after", after_bodies, layout_salt=0x9E3779B9 * layout_attempt
        )
        after_order = sorted(after_offsets, key=lambda name: after_offsets[name])
    if before_order == after_order:
        raise ValueError("could not produce distinct deterministic layouts")

    before_image = build_fixture_image(
        seed=seed,
        version="before",
        local_names=local_names,
        imports=imports,
        bodies=before_bodies,
        body_offsets=before_offsets,
        entry_name="entry_main",
    )
    after_image = build_fixture_image(
        seed=seed,
        version="after",
        local_names=local_names,
        imports=imports,
        bodies=after_bodies,
        body_offsets=after_offsets,
        entry_name="entry_main",
    )

    before_path = out_path / "patch_before.bin"
    after_path = out_path / "patch_after.bin"
    before_path.write_bytes(before_image)
    after_path.write_bytes(after_image)
    _write_json(
        out_path / "before_manifest.json",
        _manifest(
            version="before",
            seed=seed,
            offsets=before_offsets,
            bodies=before_bodies,
        ),
    )
    _write_json(
        out_path / "after_manifest.json",
        _manifest(
            version="after",
            seed=seed,
            offsets=after_offsets,
            bodies=after_bodies,
        ),
    )
    _write_json(
        out_path / "entries.json",
        {"schema_version": 1, "entries": ["entry_main"]},
    )
    _write_json(out_path / "defects.json", default_defects())
    return {
        "schema_version": 1,
        "seed": seed,
        "before_functions": len(FUNCTIONS),
        "after_functions": len(FUNCTIONS),
        "before_sha256": hashlib.sha256(before_image).hexdigest(),
        "after_sha256": hashlib.sha256(after_image).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic PE32-style patch audit fixtures."
    )
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    parser.add_argument("--out", required=True, help="output directory")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        summary = generate_samples(args.seed, args.out)
    except (OSError, ValueError) as exc:
        error = {
            "error": {
                "code": "GENERATION_FAILED",
                "message": str(exc),
            },
            "schema_version": 1,
        }
        print(json.dumps(error, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
