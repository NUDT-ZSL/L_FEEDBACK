from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .pe import PEInfo


MAX_PDB_BYTES = 60 * 1024 * 1024
MSF_MAGIC = b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\x00\x00\x00"


class PDBFormatError(Exception):
    def __init__(self, message: str, code: str = "INVALID_PDB"):
        super().__init__(message)
        self.code = code


class PDBMismatch(Exception):
    def __init__(self, fields: Dict[str, Any]):
        super().__init__("PDB does not match PE")
        self.code = "PDB_MISMATCH"
        self.fields = fields


@dataclass
class FunctionSymbol:
    name: str
    raw_name: str
    rva: int
    size: int
    segment: int
    offset: int
    type_index: int
    param_count: Optional[int] = None
    convention: Optional[str] = None
    module: str = ""
    kind: str = "S_GPROC32"
    end_rva: Optional[int] = None


@dataclass
class PublicSymbol:
    name: str
    raw_name: str
    rva: int
    segment: int
    offset: int
    flags: int


@dataclass
class PDBInfo:
    guid: str
    age: int
    signature: int
    block_size: int
    block_count: int
    stream_count: int
    module_count: int
    functions: List[FunctionSymbol] = field(default_factory=list)
    public_symbols: List[PublicSymbol] = field(default_factory=list)
    module_names: List[str] = field(default_factory=list)
    types: Dict[int, Tuple[int, bytes]] = field(default_factory=dict)
    type_index_begin: int = 0
    type_index_end: int = 0
    dbi_index: int = 3
    sym_record_stream: int = 0


def canonical_name(name: str) -> str:
    value = name
    if value.startswith("?"):
        return value
    if value.startswith("_"):
        value = value[1:]
    if "@" in value:
        head, tail = value.rsplit("@", 1)
        if tail.isdigit():
            value = head
    return value


def _checked_slice(data: bytes, off: int, length: int, what: str) -> bytes:
    if off < 0 or length < 0 or off > len(data) or length > len(data) - off:
        raise PDBFormatError("%s extends beyond its backing data" % what)
    return data[off : off + length]


def _read_cstring(data: bytes, off: int, limit: int, what: str) -> Tuple[str, int]:
    if off < 0 or off >= limit:
        raise PDBFormatError("%s has an invalid string offset" % what)
    end = data.find(b"\x00", off, limit)
    if end < 0:
        raise PDBFormatError("%s string is not terminated" % what)
    try:
        text = data[off:end].decode("utf-8", "replace")
    except Exception as exc:
        raise PDBFormatError("%s string is invalid: %s" % (what, exc))
    return text, end + 1


def _guid(raw_le: bytes) -> str:
    return "{" + str(uuid.UUID(bytes_le=raw_le)).upper() + "}"


class MSF:
    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 56 or data[:32] != MSF_MAGIC:
            raise PDBFormatError("missing MSF 2.0 superblock")
        (
            self.block_size,
            _free_block_map,
            self.block_count,
            self.directory_bytes,
            _unknown,
            self.block_map_addr,
        ) = struct.unpack_from("<IIIIII", data, 32)
        if (
            self.block_size < 512
            or self.block_size > 65536
            or self.block_size & (self.block_size - 1)
        ):
            raise PDBFormatError("invalid MSF block size")
        if self.block_count <= 0 or self.block_count > (len(data) // self.block_size + 1):
            raise PDBFormatError("invalid MSF block count")
        if self.directory_bytes <= 0 or self.directory_bytes > len(data):
            raise PDBFormatError("invalid MSF directory size")
        if self.block_map_addr >= self.block_count:
            raise PDBFormatError("MSF block-map address is outside the file")
        map_count = (self.directory_bytes + self.block_size - 1) // self.block_size
        if map_count > 100000:
            raise PDBFormatError("MSF directory block count is unreasonable")
        map_off = self.block_map_addr * self.block_size
        map_raw = _checked_slice(
            data, map_off, map_count * 4, "MSF directory block map"
        )
        directory_blocks = list(struct.unpack_from("<%dI" % map_count, map_raw))
        parts = []
        for index in directory_blocks:
            if index >= self.block_count:
                raise PDBFormatError("MSF directory references an invalid block")
            off = index * self.block_size
            parts.append(_checked_slice(data, off, self.block_size, "MSF directory block"))
        directory = b"".join(parts)[: self.directory_bytes]
        self.stream_sizes: List[int] = []
        self.stream_blocks: List[Tuple[int, ...]] = []
        self._parse_directory(directory)

    def _parse_directory(self, directory: bytes) -> None:
        if len(directory) < 4:
            raise PDBFormatError("MSF directory is truncated")
        stream_count = struct.unpack_from("<I", directory, 0)[0]
        if stream_count == 0 or stream_count > 1000000:
            raise PDBFormatError("invalid MSF stream count")
        if 4 + stream_count * 4 > len(directory):
            raise PDBFormatError("MSF stream-size table is truncated")
        sizes = list(struct.unpack_from("<%dI" % stream_count, directory, 4))
        cursor = 4 + stream_count * 4
        for size in sizes:
            if size == 0xFFFFFFFF:
                raise PDBFormatError("MSF stream size field is 0xFFFFFFFF")
            block_count = (size + self.block_size - 1) // self.block_size
            needed = block_count * 4
            if cursor + needed > len(directory):
                raise PDBFormatError("MSF stream block table is truncated")
            blocks = (
                tuple(struct.unpack_from("<%dI" % block_count, directory, cursor))
                if block_count
                else tuple()
            )
            for block in blocks:
                if block >= self.block_count:
                    raise PDBFormatError("MSF stream references an invalid block")
            self.stream_sizes.append(size)
            self.stream_blocks.append(blocks)
            cursor += needed
        self.stream_count = stream_count

    def stream(self, index: int) -> bytes:
        if index < 0 or index >= self.stream_count:
            raise PDBFormatError("stream index %d is out of range" % index)
        size = self.stream_sizes[index]
        parts = []
        for block in self.stream_blocks[index]:
            off = block * self.block_size
            parts.append(_checked_slice(self.data, off, self.block_size, "MSF stream block"))
        return b"".join(parts)[:size]


def _parse_types(stream: bytes, info: PDBInfo) -> None:
    if len(stream) < 56:
        raise PDBFormatError("TPI stream is truncated")
    version, header_size, begin, end, record_bytes = struct.unpack_from(
        "<IIIII", stream, 0
    )
    del version
    if header_size < 56 or header_size > len(stream):
        raise PDBFormatError("invalid TPI header size")
    if begin > end or end - begin > 1000000:
        raise PDBFormatError("invalid TPI type-index range")
    if record_bytes > len(stream) - header_size:
        raise PDBFormatError("TPI type-record size exceeds the stream")
    info.type_index_begin = begin
    info.type_index_end = end
    cursor = header_size
    index = begin
    while index < end:
        if cursor + 2 > len(stream):
            raise PDBFormatError("TPI record table is truncated")
        record_len = struct.unpack_from("<H", stream, cursor)[0]
        if record_len < 2 or cursor + 2 + record_len > len(stream):
            raise PDBFormatError("TPI record has an invalid length")
        record = stream[cursor + 2 : cursor + 2 + record_len]
        kind = struct.unpack_from("<H", record, 0)[0]
        info.types[index] = (kind, record)
        cursor += 2 + record_len
        index += 1
    if index != end:
        raise PDBFormatError("TPI type index did not reach its declared end")


def _function_signature(info: PDBInfo, type_index: int) -> Tuple[Optional[int], Optional[str]]:
    if type_index == 0:
        return None, None
    record = info.types.get(type_index)
    if record is None:
        return None, None
    kind, data = record
    if kind != 0x1008 or len(data) < 14:
        return None, None
    call_conv = data[6]
    param_count = struct.unpack_from("<H", data, 8)[0]
    arg_list_index = struct.unpack_from("<I", data, 10)[0]
    arg_record = info.types.get(arg_list_index)
    if arg_record is not None and arg_record[0] == 0x1201 and len(arg_record[1]) >= 6:
        declared = struct.unpack_from("<I", arg_record[1], 2)[0]
        if declared < 100000:
            param_count = declared
    convention = {
        0: "cdecl",
        1: "cdecl",
        2: "pascal",
        3: "fastcall",
        4: "stdcall",
        5: "syscall",
        6: "thiscall",
    }.get(call_conv, "unknown")
    return param_count, convention


def _parse_symbol_stream(
    stream: bytes,
    info: PDBInfo,
    pe: PEInfo,
    module_name: str,
    symbol_bytes: int,
) -> Tuple[List[FunctionSymbol], Optional[str]]:
    if len(stream) < 4:
        raise PDBFormatError("module symbol stream is truncated")
    limit = min(len(stream), max(4, symbol_bytes))
    cursor = 4
    functions: List[FunctionSymbol] = []
    obj_name: Optional[str] = None
    while cursor + 4 <= limit:
        record_len = struct.unpack_from("<H", stream, cursor)[0]
        if record_len < 2:
            raise PDBFormatError("CodeView symbol record has an invalid length")
        total = 2 + record_len
        if cursor + total > limit:
            raise PDBFormatError("CodeView symbol record exceeds the module stream")
        kind = struct.unpack_from("<H", stream, cursor + 2)[0]
        data = stream[cursor + 4 : cursor + total]
        if kind == 0x1101:
            if len(data) < 4:
                raise PDBFormatError("S_OBJNAME record is truncated")
            text, _ = _read_cstring(data, 4, len(data), "S_OBJNAME")
            obj_name = text
        elif kind in (0x1110, 0x110F, 0x1125):
            if len(data) < 35:
                raise PDBFormatError("procedure symbol record is truncated")
            (
                _parent,
                _end,
                _next,
                code_size,
                _dbg_start,
                _dbg_end,
                type_index,
                offset,
                segment,
                _flags,
            ) = struct.unpack_from("<IIIIIIIIHB", data, 0)
            raw_name, _ = _read_cstring(data, 35, len(data), "procedure name")
            if segment == 0:
                continue
            section = next((item for item in pe.sections if item.index == segment), None)
            if section is None:
                raise PDBFormatError("PDB procedure references an unknown PE section")
            if code_size > 0x7FFFFFFF:
                raise PDBFormatError("PDB procedure has an invalid code size")
            rva = section.rva + offset
            if offset + code_size > max(section.virtual_size, section.raw_size) + 0x100:
                raise PDBFormatError("PDB procedure leaves its PE section")
            name = canonical_name(raw_name)
            params, convention = _function_signature(info, type_index)
            functions.append(
                FunctionSymbol(
                    name=name,
                    raw_name=raw_name,
                    rva=rva,
                    size=code_size,
                    segment=segment,
                    offset=offset,
                    type_index=type_index,
                    param_count=params,
                    convention=convention,
                    module=module_name,
                    kind="S_LPROC32" if kind == 0x110F else "S_GPROC32",
                    end_rva=rva + code_size,
                )
            )
        cursor += total
    if cursor != limit and limit == len(stream):
        raise PDBFormatError("module symbol stream has trailing bytes")
    return functions, obj_name


def _parse_public_stream(stream: bytes, info: PDBInfo, pe: PEInfo) -> None:
    cursor = 0
    limit = len(stream)
    while cursor + 4 <= limit:
        record_len = struct.unpack_from("<H", stream, cursor)[0]
        if record_len < 2:
            raise PDBFormatError("public symbol record has an invalid length")
        total = 2 + record_len
        if cursor + total > limit:
            raise PDBFormatError("public symbol record exceeds the symbol stream")
        kind = struct.unpack_from("<H", stream, cursor + 2)[0]
        data = stream[cursor + 4 : cursor + total]
        if kind == 0x110E:
            if len(data) < 10:
                raise PDBFormatError("S_PUB32 record is truncated")
            flags, offset, segment = struct.unpack_from("<IIH", data, 0)
            raw_name, _ = _read_cstring(data, 10, len(data), "S_PUB32 name")
            if segment == 0:
                cursor += total
                continue
            section = next((item for item in pe.sections if item.index == segment), None)
            if section is None:
                cursor += total
                continue
            rva = section.rva + offset
            info.public_symbols.append(
                PublicSymbol(
                    name=canonical_name(raw_name),
                    raw_name=raw_name,
                    rva=rva,
                    segment=segment,
                    offset=offset,
                    flags=flags,
                )
            )
        cursor += total


def load_pdb(path: str, pe: PEInfo, data: Optional[bytes] = None) -> PDBInfo:
    if data is None:
        try:
            with open(path, "rb") as handle:
                data = handle.read(MAX_PDB_BYTES + 1)
        except OSError as exc:
            raise PDBFormatError("cannot read PDB: %s" % exc)
    if len(data) > MAX_PDB_BYTES:
        raise PDBFormatError("PDB exceeds the 60MB limit")
    msf = MSF(data)
    if msf.stream_count <= 3:
        raise PDBFormatError("PDB is missing the PDB/TPI/DBI streams")
    pdb_stream = msf.stream(1)
    if len(pdb_stream) < 28:
        raise PDBFormatError("PDB information stream is truncated")
    version, signature, age = struct.unpack_from("<III", pdb_stream, 0)
    del version
    guid = _guid(pdb_stream[12:28])
    info = PDBInfo(
        guid=guid,
        age=age,
        signature=signature,
        block_size=msf.block_size,
        block_count=msf.block_count,
        stream_count=msf.stream_count,
        module_count=0,
    )

    if (
        pe.codeview_guid is not None
        and (pe.codeview_guid.upper() != guid.upper() or pe.codeview_age != age)
    ):
        raise PDBMismatch(
            {
                "code": "PDB_MISMATCH",
                "message": "PE CodeView GUID/Age does not match the PDB",
                "pe_guid": pe.codeview_guid,
                "pdb_guid": guid,
                "pe_age": pe.codeview_age,
                "pdb_age": age,
            }
        )
    if pe.codeview_guid is None:
        raise PDBMismatch(
            {
                "code": "PDB_MISMATCH",
                "message": "PE has no CodeView debug directory",
                "pe_guid": None,
                "pdb_guid": guid,
                "pe_age": None,
                "pdb_age": age,
            }
        )

    dbi = msf.stream(3)
    if len(dbi) < 64:
        raise PDBFormatError("DBI stream header is truncated")
    dbi_signature = struct.unpack_from("<i", dbi, 0)[0]
    if dbi_signature != -1:
        raise PDBFormatError("DBI stream has an invalid signature")
    sym_record_stream = struct.unpack_from("<H", dbi, 20)[0]
    mod_info_size = struct.unpack_from("<i", dbi, 24)[0]
    info.sym_record_stream = sym_record_stream
    if mod_info_size < 0 or 64 + mod_info_size > len(dbi):
        raise PDBFormatError("DBI module-information substream is invalid")
    cursor = 64
    limit = 64 + mod_info_size
    module_records: List[Dict[str, Any]] = []
    while cursor < limit:
        if cursor + 64 > limit:
            raise PDBFormatError("DBI module record header is truncated")
        module_sym_stream = struct.unpack_from("<H", dbi, cursor + 34)[0]
        sym_bytes = struct.unpack_from("<I", dbi, cursor + 36)[0]
        c11_bytes = struct.unpack_from("<I", dbi, cursor + 40)[0]
        c13_bytes = struct.unpack_from("<I", dbi, cursor + 44)[0]
        module_name, after_name = _read_cstring(dbi, cursor + 64, limit, "module name")
        obj_name, after_obj = _read_cstring(dbi, after_name, limit, "object name")
        if module_sym_stream != 0xFFFF:
            if module_sym_stream == 3:
                raise PDBFormatError("DBI module symbol stream references itself")
            if module_sym_stream >= msf.stream_count:
                raise PDBFormatError("DBI module symbol stream index is out of range")
            if sym_bytes > msf.stream_sizes[module_sym_stream]:
                raise PDBFormatError("DBI module symbol byte size exceeds its stream")
        module_records.append(
            {
                "name": module_name,
                "obj": obj_name,
                "stream": module_sym_stream,
                "sym_bytes": sym_bytes,
                "c11_bytes": c11_bytes,
                "c13_bytes": c13_bytes,
            }
        )
        cursor = (after_obj + 3) & ~3
    info.module_count = len(module_records)
    info.module_names = [item["name"] for item in module_records]

    tpi = msf.stream(2)
    _parse_types(tpi, info)

    for record in module_records:
        stream_index = record["stream"]
        if stream_index == 0xFFFF:
            continue
        module_stream = msf.stream(stream_index)
        functions, _obj = _parse_symbol_stream(
            module_stream, info, pe, record["name"], record["sym_bytes"]
        )
        info.functions.extend(functions)

    if sym_record_stream and sym_record_stream < msf.stream_count:
        if sym_record_stream == 3:
            raise PDBFormatError("DBI symbol-record stream references itself")
        _parse_public_stream(msf.stream(sym_record_stream), info, pe)

    unique: Dict[Tuple[int, int, str], FunctionSymbol] = {}
    for function in info.functions:
        unique[(function.rva, function.size, function.name)] = function
    info.functions = sorted(
        unique.values(), key=lambda item: (item.rva, item.size, item.name)
    )
    info.public_symbols = sorted(
        info.public_symbols, key=lambda item: (item.rva, item.name, item.segment, item.offset)
    )
    return info


def function_by_name(info: PDBInfo) -> Dict[str, FunctionSymbol]:
    result: Dict[str, FunctionSymbol] = {}
    for function in info.functions:
        result.setdefault(function.name, function)
    return result


def function_covering_rva(info: PDBInfo, rva: int) -> Optional[FunctionSymbol]:
    for function in info.functions:
        if function.rva <= rva < function.rva + max(1, function.size):
            return function
    return None

