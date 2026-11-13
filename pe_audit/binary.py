"""The deterministic PE32-style fixture format and its bytecode decoder.

The fixture deliberately does not try to be loadable by Windows.  It does use
real PE headers, a real section table and real import descriptors, while the
function bodies use a small fixed-width, name-indexed instruction set:

* local calls contain an index into a sorted local function-name table;
* import calls contain an index into a sorted import table;
* no instruction contains a function RVA or a file offset.

That separation is what lets an audit align functions by name first and only
then reason about calls.  A changed layout cannot change a call's target.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

from .errors import binary_error


PEAF_MAGIC = b"PEAF"
PEAF_VERSION = 1
INSTRUCTION_SIZE = 4
SECTION_RVA = 0x1000
SECTION_RAW_OFFSET = 0x200
LOCAL_NAME_OFFSET = 0x10
IMPORT_DESCRIPTOR_OFFSET = 0x100

OP_PAD = 0x00
OP_PUSH_IMM = 0x01
OP_CMP_IMM = 0x02
OP_CALL_LOCAL = 0x03
OP_CALL_IMPORT = 0x04
OP_JCC = 0x05
OP_JMP = 0x06
OP_RET = 0x07
OP_NOP = 0x08


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _put_u16(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", data, offset, value)


def _put_u32(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", data, offset, value)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True, order=True)
class ImportSymbol:
    module: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {"module": self.module, "name": self.name}


@dataclass(frozen=True)
class Instruction:
    """One fixed-width bytecode instruction.

    ``value`` is the unsigned payload for most opcodes.  Conditional branches
    use ``branch_number`` and a signed ``target_rel``.  ``PAD`` instructions
    occupy space but are absent from the effective instruction stream.
    """

    opcode: int
    value: int = 0
    branch_number: int | None = None
    target_rel: int = 0

    def __post_init__(self) -> None:
        if self.opcode not in {
            OP_PAD,
            OP_PUSH_IMM,
            OP_CMP_IMM,
            OP_CALL_LOCAL,
            OP_CALL_IMPORT,
            OP_JCC,
            OP_JMP,
            OP_RET,
            OP_NOP,
        }:
            raise ValueError(f"unsupported opcode: {self.opcode!r}")
        if not 0 <= self.value <= 0xFFFFFF:
            raise ValueError("instruction value must fit in 24 bits")
        if self.opcode == OP_JCC:
            if self.branch_number is None or not 0 <= self.branch_number <= 0xFF:
                raise ValueError("JCC branch number must be in range 0..255")
            if not -(1 << 15) <= self.target_rel < (1 << 15):
                raise ValueError("JCC target must fit in a signed 16-bit field")
        elif self.opcode == OP_JMP:
            if not -(1 << 23) <= self.value < (1 << 23):
                raise ValueError("JMP target must fit in a signed 24-bit field")

    def encode(self) -> bytes:
        if self.opcode == OP_JCC:
            assert self.branch_number is not None
            return struct.pack(">BBh", self.opcode, self.branch_number, self.target_rel)
        if self.opcode == OP_JMP:
            signed = self.value & 0xFFFFFF
            return bytes((self.opcode,)) + signed.to_bytes(3, "big")
        return bytes((self.opcode,)) + self.value.to_bytes(3, "big")


def pad() -> Instruction:
    return Instruction(OP_PAD, 0)


def push_imm(value: int) -> Instruction:
    return Instruction(OP_PUSH_IMM, value)


def cmp_imm(value: int) -> Instruction:
    return Instruction(OP_CMP_IMM, value)


def call_local(name_index: int) -> Instruction:
    return Instruction(OP_CALL_LOCAL, name_index)


def call_import(import_index: int) -> Instruction:
    return Instruction(OP_CALL_IMPORT, import_index)


def jcc(branch_number: int, target_rel: int = 4) -> Instruction:
    return Instruction(OP_JCC, 0, branch_number, target_rel)


def jmp(target_rel: int) -> Instruction:
    return Instruction(OP_JMP, target_rel)


def ret() -> Instruction:
    return Instruction(OP_RET, 0)


def nop() -> Instruction:
    return Instruction(OP_NOP, 0)


def noop() -> Instruction:
    """Alias used by callers that think in terms of a non-semantic no-op."""
    return nop()


@dataclass(frozen=True)
class DecodedInstruction:
    opcode: int
    offset: int
    value: int
    branch_number: int | None
    target_rel: int

    @property
    def kind(self) -> str:
        return "branch" if self.opcode in (OP_JCC, OP_JMP) else "insn"


@dataclass
class DecodeResult:
    instructions: list[DecodedInstruction]
    invalid_bytes: int
    parse_issues: list[dict[str, object]]


def decode_instruction(chunk: bytes) -> DecodedInstruction | None:
    """Decode one four-byte instruction, or return ``None`` for illegal bytes."""

    if len(chunk) != INSTRUCTION_SIZE:
        return None
    opcode = chunk[0]
    if opcode == OP_JCC:
        branch_number, target_rel = struct.unpack(">Bh", chunk[1:])
        return DecodedInstruction(opcode, 0, 0, branch_number, target_rel)
    value = int.from_bytes(chunk[1:], "big")
    if opcode == OP_JMP:
        if value & 0x800000:
            value -= 1 << 24
        return DecodedInstruction(opcode, 0, value, None, value)
    if opcode in {
        OP_PAD,
        OP_PUSH_IMM,
        OP_CMP_IMM,
        OP_CALL_LOCAL,
        OP_CALL_IMPORT,
        OP_RET,
        OP_NOP,
    }:
        return DecodedInstruction(opcode, 0, value, None, 0)
    return None


def decode_body(
    body: bytes, function: str, base_offset: int = 0
) -> DecodeResult:
    """Decode a body, count malformed bytes, and continue after illegal words.

    A trailing run of zero bytes is alignment padding, not corruption.  Any
    other truncated tail is counted byte-for-byte as invalid.
    """

    instructions: list[DecodedInstruction] = []
    problems: list[dict[str, object]] = []
    invalid = 0
    pos = 0
    while pos < len(body):
        remaining = len(body) - pos
        if remaining < INSTRUCTION_SIZE:
            tail = body[pos:]
            if all(byte == 0 for byte in tail):
                break
            invalid += remaining
            problems.append(
                {
                    "function": function,
                    "offset": base_offset + pos,
                    "bytes": remaining,
                    "reason": "truncated_instruction",
                }
            )
            break
        chunk = body[pos : pos + INSTRUCTION_SIZE]
        decoded = decode_instruction(chunk)
        if decoded is None:
            invalid += INSTRUCTION_SIZE
            problems.append(
                {
                    "function": function,
                    "offset": base_offset + pos,
                    "bytes": INSTRUCTION_SIZE,
                    "reason": "illegal_opcode",
                    "opcode": chunk[0],
                }
            )
        elif decoded.opcode != OP_PAD:
            instructions.append(
                DecodedInstruction(
                    opcode=decoded.opcode,
                    offset=base_offset + pos,
                    value=decoded.value,
                    branch_number=decoded.branch_number,
                    target_rel=decoded.target_rel,
                )
            )
        pos += INSTRUCTION_SIZE
    return DecodeResult(instructions, invalid, problems)


@dataclass(frozen=True)
class SectionInfo:
    name: str
    virtual_address: int
    virtual_size: int
    raw_offset: int
    raw_size: int


@dataclass
class PeImage:
    data: bytes
    sections: list[SectionInfo]
    imports: list[ImportSymbol]
    local_names: list[str]
    import_rva: int
    entry_rva: int

    def rva_to_offset(self, rva: int) -> int | None:
        for section in self.sections:
            if section.virtual_address <= rva < section.virtual_address + section.raw_size:
                delta = rva - section.virtual_address
                if delta >= section.raw_size:
                    return None
                offset = section.raw_offset + delta
                if offset >= len(self.data):
                    return None
                return offset
        return None

    def interval_bytes(self, rva: int, length: int) -> bytes | None:
        if length <= 0:
            return None
        for section in self.sections:
            start = section.virtual_address
            limit = start + min(section.raw_size, section.virtual_size)
            if rva < start or rva + length > limit:
                continue
            offset = section.raw_offset + (rva - start)
            end = offset + length
            if offset < 0 or end > len(self.data):
                return None
            return self.data[offset:end]
        return None

    def section_overlaps_interval(self, rva: int, length: int) -> bool:
        return self.interval_bytes(rva, length) is not None


def _name_blob(names: Sequence[str]) -> bytes:
    parts: list[bytes] = []
    for name in names:
        encoded = name.encode("ascii")
        if not encoded or len(encoded) > 255:
            raise ValueError(f"name cannot be encoded in one byte: {name!r}")
        parts.append(bytes((len(encoded),)) + encoded)
    return b"".join(parts)


def _write_name_blob(section: bytearray, offset: int, names: Sequence[str]) -> int:
    blob = _name_blob(names)
    end = offset + len(blob)
    if end > len(section):
        raise ValueError("local-name table exceeds section prefix")
    section[offset:end] = blob
    return end


def _write_import_table(
    section: bytearray,
    imports: Sequence[ImportSymbol],
) -> tuple[int, int, int]:
    """Write PE import descriptors and thunk tables into the section prefix."""

    modules: dict[str, list[str]] = {}
    for symbol in sorted(imports, key=lambda item: (item.module, item.name)):
        modules.setdefault(symbol.module, []).append(symbol.name)
    module_names = sorted(modules)

    descriptor_start = IMPORT_DESCRIPTOR_OFFSET
    cursor = descriptor_start + (len(module_names) + 1) * 20
    module_offsets: dict[str, int] = {}
    for module in module_names:
        module_offsets[module] = cursor
        cursor += len(module.encode("ascii")) + 1

    thunk_offsets: dict[str, tuple[int, int]] = {}
    for module in module_names:
        cursor = _align(cursor, 4)
        ilt_offset = cursor
        cursor += (len(modules[module]) + 1) * 4
        cursor = _align(cursor, 4)
        iat_offset = cursor
        cursor += (len(modules[module]) + 1) * 4
        thunk_offsets[module] = (ilt_offset, iat_offset)

    hint_name_offsets: dict[tuple[str, str], int] = {}
    for module in module_names:
        for name in modules[module]:
            cursor = _align(cursor, 2)
            hint_name_offsets[(module, name)] = cursor
            cursor += 2 + len(name.encode("ascii")) + 1

    for index, module in enumerate(module_names):
        offset = descriptor_start + index * 20
        ilt_offset, iat_offset = thunk_offsets[module]
        struct.pack_into(
            "<IIIII",
            section,
            offset,
            SECTION_RVA + ilt_offset,
            0,
            0,
            SECTION_RVA + module_offsets[module],
            SECTION_RVA + iat_offset,
        )
        section[
            module_offsets[module] : module_offsets[module]
            + len(module.encode("ascii")) + 1
        ] = module.encode("ascii") + b"\x00"

    null_descriptor = descriptor_start + len(module_names) * 20
    section[null_descriptor : null_descriptor + 20] = b"\x00" * 20

    for module in module_names:
        ilt_offset, iat_offset = thunk_offsets[module]
        rows = modules[module]
        for index, name in enumerate(rows):
            target = SECTION_RVA + hint_name_offsets[(module, name)]
            struct.pack_into("<I", section, ilt_offset + index * 4, target)
            struct.pack_into("<I", section, iat_offset + index * 4, target)
            hint_name = hint_name_offsets[(module, name)]
            section[hint_name : hint_name + 2] = b"\x00\x00"
            name_end = hint_name + 2 + len(name.encode("ascii"))
            section[hint_name + 2 : name_end + 1] = name.encode("ascii") + b"\x00"

    import_size = (len(module_names) + 1) * 20
    return descriptor_start, import_size, cursor


def build_fixture_image(
    *,
    seed: int,
    version: str,
    local_names: Sequence[str],
    imports: Sequence[ImportSymbol],
    bodies: dict[str, bytes],
    body_offsets: dict[str, int],
    entry_name: str,
) -> bytes:
    """Assemble one deterministic PE32-style image.

    ``body_offsets`` are offsets from the start of the single ``.text``
    section.  The layout is intentionally supplied by the caller so it can
    differ between versions without affecting name-based call encoding.
    """

    if version not in {"before", "after"}:
        raise ValueError("version must be 'before' or 'after'")
    if entry_name not in body_offsets:
        raise ValueError("entry function has no body")
    names = list(local_names)
    sorted_imports = sorted(imports, key=lambda item: (item.module, item.name))
    prefix_padding_base = min(body_offsets.values()) if body_offsets else 0x400
    if prefix_padding_base < 0x400:
        raise ValueError("function bodies must start at or beyond section offset 0x400")

    section_size = max(
        body_offsets[name] + len(body) for name, body in bodies.items()
    )
    section_size = _align(section_size, 0x100)
    section = bytearray(section_size)

    section[0:16] = struct.pack(
        "<4sHHII", PEAF_MAGIC, PEAF_VERSION, len(names), LOCAL_NAME_OFFSET, 0
    )
    names_end = _write_name_blob(section, LOCAL_NAME_OFFSET, names)
    if names_end >= IMPORT_DESCRIPTOR_OFFSET:
        raise ValueError("local-name table overlaps import descriptors")
    descriptor_offset, import_size, metadata_end = _write_import_table(
        section, sorted_imports
    )
    import_rva = SECTION_RVA + descriptor_offset
    if metadata_end >= prefix_padding_base:
        raise ValueError("metadata overlaps the first function body")

    for name, body in bodies.items():
        offset = body_offsets[name]
        section[offset : offset + len(body)] = body

    raw_size = _align(len(section), 0x200)
    image_size = _align(SECTION_RVA + raw_size, 0x1000)
    entry_rva = SECTION_RVA + body_offsets[entry_name]

    dos = bytearray(0x40)
    dos[0:2] = b"MZ"
    comment = f"seed={seed}; deterministic PE32 audit fixture".encode("ascii")[:0x34]
    dos[0x04 : 0x04 + len(comment)] = comment
    _put_u32(dos, 0x3C, 0x40)

    headers = bytearray(raw_size)
    headers[:0x40] = dos
    headers[0x40:0x44] = b"PE\x00\x00"
    coff = 0x44
    _put_u16(headers, coff + 0, 0x014C)  # i386
    _put_u16(headers, coff + 2, 1)  # one section
    _put_u32(headers, coff + 4, 0)  # no timestamp
    _put_u32(headers, coff + 8, 0)
    _put_u32(headers, coff + 12, 0)
    _put_u16(headers, coff + 16, 0xE0)
    _put_u16(headers, coff + 18, 0x0102)

    optional = 0x58
    _put_u16(headers, optional + 0, 0x10B)  # PE32
    headers[optional + 2] = 1
    headers[optional + 3] = 0
    _put_u32(headers, optional + 4, raw_size)
    _put_u32(headers, optional + 8, 0)
    _put_u32(headers, optional + 12, 0)
    _put_u32(headers, optional + 16, entry_rva)
    _put_u32(headers, optional + 20, SECTION_RVA)
    _put_u32(headers, optional + 24, SECTION_RVA)
    _put_u32(headers, optional + 28, 0x10000000)
    _put_u32(headers, optional + 32, 0x1000)
    _put_u32(headers, optional + 36, 0x200)
    _put_u16(headers, optional + 40, 6)  # OS major
    _put_u16(headers, optional + 42, 0)  # OS minor
    _put_u16(headers, optional + 44, 0)  # image major
    _put_u16(headers, optional + 46, 0)  # image minor
    _put_u16(headers, optional + 48, 6)  # subsystem major
    _put_u16(headers, optional + 50, 0)  # subsystem minor
    _put_u32(headers, optional + 52, 0)  # Win32 version
    _put_u32(headers, optional + 56, image_size)
    _put_u32(headers, optional + 60, SECTION_RAW_OFFSET)
    _put_u32(headers, optional + 64, 0)  # checksum
    _put_u16(headers, optional + 68, 3)  # console subsystem
    _put_u16(headers, optional + 70, 0)  # DLL characteristics
    _put_u32(headers, optional + 72, 0x100000)
    _put_u32(headers, optional + 76, 0x1000)
    _put_u32(headers, optional + 80, 0x100000)
    _put_u32(headers, optional + 84, 0x1000)
    _put_u32(headers, optional + 88, 0)  # loader flags
    _put_u32(headers, optional + 92, 16)  # data-directory count
    _put_u32(headers, optional + 96 + 8, import_rva)
    _put_u32(headers, optional + 96 + 12, import_size)

    section_table = optional + 0xE0
    headers[section_table : section_table + 8] = b".text\x00\x00\x00"
    _put_u32(headers, section_table + 8, len(section))
    _put_u32(headers, section_table + 12, SECTION_RVA)
    _put_u32(headers, section_table + 16, raw_size)
    _put_u32(headers, section_table + 20, SECTION_RAW_OFFSET)
    _put_u32(headers, section_table + 36, 0x60000020)

    file_data = bytes(headers[:SECTION_RAW_OFFSET]) + bytes(section).ljust(
        raw_size, b"\x00"
    )
    if len(file_data) != SECTION_RAW_OFFSET + raw_size:
        raise ValueError("internal fixture layout error")
    return file_data


def parse_image(data: bytes) -> PeImage:
    """Parse the PE headers, section table, imports and local name table."""

    if len(data) < 0x200:
        raise binary_error("file is too short to contain PE headers")
    if data[0:2] != b"MZ":
        raise binary_error("missing DOS MZ signature")
    pe_offset = _u32(data, 0x3C)
    if pe_offset + 24 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        raise binary_error("missing PE signature")
    coff = pe_offset + 4
    section_count = _u16(data, coff + 2)
    optional_size = _u16(data, coff + 16)
    optional = coff + 20
    if section_count < 1 or optional + optional_size > len(data):
        raise binary_error("invalid PE optional-header extent")
    if _u16(data, optional) != 0x10B:
        raise binary_error("fixture must use PE32 magic 0x10b")
    entry_rva = _u32(data, optional + 16)
    number_of_directories = _u32(data, optional + 92)
    directories = optional + 96
    if number_of_directories < 1:
        raise binary_error("missing import data directory")
    import_rva = _u32(data, directories + 8)
    import_size = _u32(data, directories + 12)

    section_table = optional + optional_size
    sections: list[SectionInfo] = []
    for index in range(section_count):
        base = section_table + index * 40
        if base + 40 > len(data):
            raise binary_error("truncated section table")
        raw_name = data[base : base + 8].split(b"\x00", 1)[0].decode(
            "ascii", errors="replace"
        )
        virtual_size = _u32(data, base + 8)
        virtual_address = _u32(data, base + 12)
        raw_size = _u32(data, base + 16)
        raw_offset = _u32(data, base + 20)
        if raw_size and raw_offset + raw_size > len(data):
            raise binary_error("section raw data exceeds file size")
        sections.append(
            SectionInfo(
                raw_name,
                virtual_address,
                virtual_size,
                raw_offset,
                raw_size,
            )
        )

    image = PeImage(data, sections, [], [], import_rva, entry_rva)
    text = _find_section(sections, SECTION_RVA)
    if text is None:
        raise binary_error("missing .text section at expected RVA")
    prefix = data[text.raw_offset : text.raw_offset + 16]
    if len(prefix) != 16 or prefix[0:4] != PEAF_MAGIC:
        raise binary_error("missing PEAF name-table header")
    version, count, names_offset, _reserved = struct.unpack_from("<HHII", prefix, 4)
    if version != PEAF_VERSION:
        raise binary_error(f"unsupported PEAF version {version}")
    local_names = _read_local_names(text, names_offset, count, data)
    image.local_names = local_names
    image.imports = _read_imports(image, import_rva, import_size)
    return image


def _find_section(sections: Sequence[SectionInfo], rva: int) -> SectionInfo | None:
    for section in sections:
        start = section.virtual_address
        if start <= rva < start + max(section.virtual_size, section.raw_size):
            return section
    return None


def _read_local_names(
    section: SectionInfo, names_offset: int, count: int, data: bytes
) -> list[str]:
    if count > 4096:
        raise binary_error("unreasonable local-name count")
    position = section.raw_offset + names_offset
    end = section.raw_offset + min(section.raw_size, section.virtual_size)
    names: list[str] = []
    for _ in range(count):
        if position >= end or position >= len(data):
            raise binary_error("truncated local-name table")
        length = data[position]
        position += 1
        name_end = position + length
        if name_end > end or name_end > len(data):
            raise binary_error("truncated local-name entry")
        try:
            name = data[position:name_end].decode("ascii")
        except UnicodeDecodeError as exc:
            raise binary_error("local name is not ASCII") from exc
        if not name:
            raise binary_error("empty local name")
        names.append(name)
        position = name_end
    return names


def _read_imports(image: PeImage, import_rva: int, import_size: int) -> list[ImportSymbol]:
    offset = image.rva_to_offset(import_rva) if import_rva else None
    if offset is None:
        raise binary_error("import directory RVA is outside section data")
    if import_size < 20:
        raise binary_error("import directory is too small")
    descriptors = import_size // 20
    modules: list[tuple[str, int, int]] = []
    for index in range(descriptors):
        position = offset + index * 20
        if position + 20 > len(image.data):
            raise binary_error("truncated import descriptor")
        original_thunk, _stamp, _chain, name_rva, first_thunk = struct.unpack_from(
            "<IIIII", image.data, position
        )
        if original_thunk == 0 and name_rva == 0 and first_thunk == 0:
            break
        if name_rva == 0:
            raise binary_error("import descriptor without module name")
        module = _read_ascii_at(image, name_rva, "module name")
        modules.append((module, original_thunk or first_thunk, first_thunk))

    symbols: list[ImportSymbol] = []
    for module, lookup_rva, _iat_rva in modules:
        offset = image.rva_to_offset(lookup_rva)
        if offset is None:
            raise binary_error("import lookup table is outside section data")
        seen = 0
        while True:
            if offset + 4 > len(image.data):
                raise binary_error("truncated import lookup table")
            entry = _u32(image.data, offset)
            offset += 4
            if entry == 0:
                break
            seen += 1
            if seen > 4096:
                raise binary_error("unreasonable import lookup table")
            if entry & 0x80000000:
                symbols.append(ImportSymbol(module, f"ordinal:{entry & 0xFFFF}"))
                continue
            name_offset = image.rva_to_offset(entry)
            if name_offset is None or name_offset + 2 > len(image.data):
                raise binary_error("import-by-name record is outside section data")
            name = _read_ascii_at(image, entry + 2, "import name")
            symbols.append(ImportSymbol(module, name))
    return sorted(symbols, key=lambda item: (item.module, item.name))


def _read_ascii_at(image: PeImage, rva: int, description: str) -> str:
    offset = image.rva_to_offset(rva)
    if offset is None:
        raise binary_error(f"{description} RVA is outside section data")
    end = image.data.find(b"\x00", offset)
    if end < 0 or end == offset:
        raise binary_error(f"missing or empty {description}")
    try:
        return image.data[offset:end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise binary_error(f"{description} is not ASCII") from exc


def deterministic_sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def active_branches(instructions: Iterable[DecodedInstruction]) -> list[int]:
    """Diagnostic helper: list JCC branch numbers in decoding order."""
    return [
        instruction.branch_number
        for instruction in instructions
        if instruction.opcode == OP_JCC and instruction.branch_number is not None
    ]
