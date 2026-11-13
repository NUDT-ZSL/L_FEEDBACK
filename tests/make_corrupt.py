#!/usr/bin/env python3
"""Create deterministic malformed PE/PDB inputs for boundary testing."""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def corrupt_pe_section(source: str, destination: str) -> None:
    shutil.copyfile(source, destination)
    data = bytearray(Path(destination).read_bytes())
    pe = _u32(data, 0x3C)
    coff = pe + 4
    optional_size = struct.unpack_from("<H", data, coff + 16)[0]
    table = coff + 20 + optional_size
    struct.pack_into("<I", data, table + 20, 0x7FFFFFFF)
    Path(destination).write_bytes(data)


def _msf_directory(data: bytes):
    block_size = _u32(data, 32)
    block_count = _u32(data, 40)
    directory_bytes = _u32(data, 44)
    block_map_addr = _u32(data, 52)
    map_count = (directory_bytes + block_size - 1) // block_size
    map_offset = block_map_addr * block_size
    directory_blocks = [
        struct.unpack_from("<I", data, map_offset + 4 * index)[0]
        for index in range(map_count)
    ]
    first_block_offset = directory_blocks[0] * block_size
    directory = b"".join(
        data[block * block_size : (block + 1) * block_size]
        for block in directory_blocks
    )[:directory_bytes]
    stream_count = _u32(directory, 0)
    sizes = [
        struct.unpack_from("<I", directory, 4 + 4 * index)[0]
        for index in range(stream_count)
    ]
    cursor = 4 + 4 * stream_count
    blocks = []
    for size in sizes:
        count = (size + block_size - 1) // block_size
        blocks.append(
            [
                struct.unpack_from("<I", directory, cursor + 4 * block_index)[0]
                for block_index in range(count)
            ]
        )
        cursor += 4 * count
    return block_size, first_block_offset, sizes, blocks


def corrupt_pdb_stream_size(source: str, destination: str) -> None:
    shutil.copyfile(source, destination)
    data = bytearray(Path(destination).read_bytes())
    block_size, first_directory_offset, sizes, _blocks = _msf_directory(data)
    if len(sizes) <= 1:
        raise SystemExit("PDB has no stream 1")
    struct.pack_into("<I", data, first_directory_offset + 4 + 4 * 1, 0xFFFFFFFF)
    Path(destination).write_bytes(data)


def corrupt_pdb_dbi_self_reference(source: str, destination: str) -> None:
    shutil.copyfile(source, destination)
    data = bytearray(Path(destination).read_bytes())
    block_size, _first_directory_offset, _sizes, blocks = _msf_directory(data)
    if len(blocks) <= 3 or not blocks[3]:
        raise SystemExit("PDB has no DBI stream")
    dbi_offset = blocks[3][0] * block_size
    module_stream_offset = dbi_offset + 64 + 34
    struct.pack_into("<H", data, module_stream_offset, 3)
    Path(destination).write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    corrupt_pe_section(args.pe, str(outdir / "bad_section.exe"))
    corrupt_pdb_stream_size(args.pdb, str(outdir / "bad_stream.pdb"))
    corrupt_pdb_dbi_self_reference(args.pdb, str(outdir / "bad_dbi.pdb"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
