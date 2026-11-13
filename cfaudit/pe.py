from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


MAX_PE_BYTES = 30 * 1024 * 1024
IMAGE_DIRECTORY_COUNT = 16


class PEFormatError(Exception):
    def __init__(self, message: str, code: str = "INVALID_PE"):
        super().__init__(message)
        self.code = code


@dataclass
class Section:
    name: str
    rva: int
    virtual_size: int
    raw_size: int
    raw_ptr: int
    characteristics: int
    index: int

    def contains_rva(self, rva: int) -> bool:
        span = max(self.virtual_size, self.raw_size)
        return self.rva <= rva < self.rva + span


@dataclass
class PEInfo:
    data: bytes
    machine: int
    magic: int
    is_pe32_plus: bool
    image_base: int
    entry_point: int
    size_of_image: int
    timestamp: int
    sections: List[Section]
    directories: List[Tuple[int, int]]
    imports: List[Dict[str, Any]] = field(default_factory=list)
    exports: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    base_relocations: List[Dict[str, Any]] = field(default_factory=list)
    codeview_guid: Optional[str] = None
    codeview_age: Optional[int] = None
    codeview_path: Optional[str] = None
    import_by_iat: Dict[int, str] = field(default_factory=dict)
    import_thunk_va: Dict[int, str] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def section_for_rva(self, rva: int) -> Optional[Section]:
        for section in self.sections:
            if section.contains_rva(rva):
                return section
        return None

    def rva_to_offset(self, rva: int, length: int = 1) -> Optional[int]:
        if rva < 0:
            return None
        section = self.section_for_rva(rva)
        if section is None:
            if self.sections and rva + length <= self.sections[0].rva:
                return rva
            return None
        delta = rva - section.rva
        if delta + length > section.raw_size:
            return None
        return section.raw_ptr + delta

    def read_rva(self, rva: int, length: int) -> Optional[bytes]:
        off = self.rva_to_offset(rva, length)
        if off is None or off + length > len(self.data):
            return None
        return self.data[off : off + length]


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _u64(data: bytes, off: int) -> int:
    return struct.unpack_from("<Q", data, off)[0]


def _format_guid(raw_le: bytes) -> str:
    return "{" + str(uuid.UUID(bytes_le=raw_le)).upper() + "}"


def _section_name(raw: bytes) -> str:
    end = raw.find(b"\x00")
    if end < 0:
        end = len(raw)
    return raw[:end].decode("latin-1")


def _checked_range(off: int, size: int, total: int, what: str) -> None:
    if off < 0 or size < 0 or off > total or size > total - off:
        raise PEFormatError("%s extends beyond the file" % what)


def load_pe(path: str, data: Optional[bytes] = None) -> PEInfo:
    if data is None:
        try:
            with open(path, "rb") as handle:
                data = handle.read(MAX_PE_BYTES + 1)
        except OSError as exc:
            raise PEFormatError("cannot read PE: %s" % exc)
    if len(data) > MAX_PE_BYTES:
        raise PEFormatError("PE exceeds the 30MB limit")
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise PEFormatError("missing DOS MZ header")

    pe_off_raw = _u32(data, 0x3C)
    if pe_off_raw > len(data) - 4:
        raise PEFormatError("DOS e_lfanew points beyond the file")
    if data[pe_off_raw : pe_off_raw + 4] != b"PE\x00\x00":
        raise PEFormatError("missing PE signature")

    coff = pe_off_raw + 4
    _checked_range(coff, 20, len(data), "COFF header")
    machine, section_count, timestamp, _, _, opt_size, _ = struct.unpack_from(
        "<HHIIIHH", data, coff
    )
    opt = coff + 20
    _checked_range(opt, opt_size, len(data), "optional header")
    if opt_size < 2:
        raise PEFormatError("optional header is too small")
    magic = _u16(data, opt)
    if magic not in (0x10B, 0x20B):
        raise PEFormatError("unsupported optional-header magic 0x%x" % magic)
    is_plus = magic == 0x20B
    min_opt = 112 if is_plus else 96
    if opt_size < min_opt:
        raise PEFormatError("truncated optional header")
    if section_count > 0x2000:
        raise PEFormatError("section count is unreasonable")

    entry_point = _u32(data, opt + 16)
    image_base = _u64(data, opt + 24) if is_plus else _u32(data, opt + 28)
    size_of_image = _u32(data, opt + 56)
    dir_count_off = opt + (108 if is_plus else 92)
    dir_count = _u32(data, dir_count_off) if opt_size >= (dir_count_off - opt + 4) else 0
    dir_count = min(dir_count, IMAGE_DIRECTORY_COUNT)
    dirs_off = opt + (112 if is_plus else 96)
    if opt_size < dirs_off - opt + 8 * dir_count:
        raise PEFormatError("optional data-directory array is truncated")
    directories = [
        struct.unpack_from("<II", data, dirs_off + 8 * i)
        for i in range(dir_count)
    ]
    while len(directories) < IMAGE_DIRECTORY_COUNT:
        directories.append((0, 0))

    sec_off = opt + opt_size
    _checked_range(sec_off, section_count * 40, len(data), "section table")
    sections: List[Section] = []
    for index in range(section_count):
        base = sec_off + index * 40
        name_raw = data[base : base + 8]
        virtual_size, virtual_address, raw_size, raw_ptr = struct.unpack_from(
            "<IIII", data, base + 8
        )
        characteristics = _u32(data, base + 36)
        if raw_size:
            _checked_range(raw_ptr, raw_size, len(data), "section %d raw data" % (index + 1))
        sections.append(
            Section(
                name=_section_name(name_raw),
                rva=virtual_address,
                virtual_size=virtual_size,
                raw_size=raw_size,
                raw_ptr=raw_ptr,
                characteristics=characteristics,
                index=index + 1,
            )
        )
    sections.sort(key=lambda item: (item.rva, item.index))
    if not sections:
        raise PEFormatError("PE contains no sections")

    info = PEInfo(
        data=data,
        machine=machine,
        magic=magic,
        is_pe32_plus=is_plus,
        image_base=image_base,
        entry_point=entry_point,
        size_of_image=size_of_image,
        timestamp=timestamp,
        sections=sections,
        directories=directories,
    )

    _parse_debug(info)
    _parse_imports(info)
    _parse_exports(info)
    _parse_relocations(info)
    _parse_functions(info)
    return info


def _parse_debug(info: PEInfo) -> None:
    rva, size = info.directories[6]
    if not rva or size < 28:
        return
    off = info.rva_to_offset(rva, size)
    if off is None:
        raise PEFormatError("debug directory points outside section data")
    count = size // 28
    for index in range(count):
        base = off + index * 28
        characteristics, timestamp, _major, _minor, dtype, data_size, data_rva, data_ptr = struct.unpack_from(
            "<IIHHIIII", info.data, base
        )
        del characteristics, timestamp
        if dtype != 2 or data_size < 24:
            continue
        if not data_ptr:
            inner = info.read_rva(data_rva, data_size)
            if inner is None:
                continue
            raw = inner
        else:
            if data_ptr > len(info.data) or data_size > len(info.data) - data_ptr:
                raise PEFormatError("CodeView debug data points beyond the file")
            raw = info.data[data_ptr : data_ptr + data_size]
        if raw[:4] != b"RSDS" or len(raw) < 24:
            continue
        info.codeview_guid = _format_guid(raw[4:20])
        info.codeview_age = _u32(raw, 20)
        path_raw = raw[24:]
        end = path_raw.find(b"\x00")
        if end < 0:
            end = len(path_raw)
        info.codeview_path = path_raw[:end].decode("utf-8", "replace")
        break


def _parse_imports(info: PEInfo) -> None:
    rva, size = info.directories[1]
    if not rva or not size:
        return
    if size < 20:
        raise PEFormatError("import directory is too small")
    off = info.rva_to_offset(rva, 20)
    if off is None:
        raise PEFormatError("import directory points outside section data")
    thunk_size = 8 if info.is_pe32_plus else 4
    max_desc = min(size // 20 + 1, 4096)
    found_terminator = False
    for desc_index in range(max_desc):
        base = off + desc_index * 20
        if base + 20 > len(info.data):
            raise PEFormatError("truncated import descriptor")
        ilt, timestamp, forwarder, name_rva, iat = struct.unpack_from(
            "<IIIII", info.data, base
        )
        del timestamp, forwarder
        if ilt == 0 and name_rva == 0 and iat == 0:
            found_terminator = True
            break
        module_raw = info.read_rva(name_rva, 1)
        if module_raw is None:
            raise PEFormatError("import module name is outside the image")
        name_off = info.rva_to_offset(name_rva)
        if name_off is None:
            raise PEFormatError("import module name has no file backing")
        end = info.data.find(b"\x00", name_off)
        if end < 0:
            raise PEFormatError("unterminated import module name")
        module = info.data[name_off:end].decode("ascii", "replace")
        lookup = ilt or iat
        for item_index in range(65536):
            item_rva = lookup + item_index * thunk_size
            raw = info.read_rva(item_rva, thunk_size)
            if raw is None:
                raise PEFormatError("import thunk table leaves its section")
            value = struct.unpack_from("<Q", raw)[0] if thunk_size == 8 else _u32(raw, 0)
            if value == 0:
                break
            ordinal_flag = (1 << (thunk_size * 8 - 1))
            if value & ordinal_flag:
                func_name = "ordinal_%d" % (value & 0xFFFF)
                ordinal = value & 0xFFFF
            else:
                hint_name_rva = value & 0x7FFFFFFF
                hint_off = info.rva_to_offset(hint_name_rva, 2)
                if hint_off is None:
                    raise PEFormatError("import hint/name entry is outside the image")
                name_off = hint_off + 2
                end = info.data.find(b"\x00", name_off)
                if end < 0:
                    raise PEFormatError("unterminated import name")
                func_name = info.data[name_off:end].decode("ascii", "replace")
                ordinal = None
            iat_rva = iat + item_index * thunk_size
            entry = {
                "module": module,
                "name": func_name,
                "ordinal": ordinal,
                "iat_rva": iat_rva,
            }
            info.imports.append(entry)
            info.import_by_iat[iat_rva] = func_name
            info.import_thunk_va[iat_rva] = func_name
        else:
            raise PEFormatError("import thunk table did not terminate")
    if not found_terminator:
        raise PEFormatError("import descriptor array did not terminate")


def _parse_exports(info: PEInfo) -> None:
    rva, size = info.directories[0]
    if not rva or size < 40:
        return
    raw = info.read_rva(rva, 40)
    if raw is None:
        raise PEFormatError("export directory points outside section data")
    (
        _characteristics,
        _timestamp,
        _major,
        _minor,
        _name_rva,
        _base,
        function_count,
        name_count,
        functions_rva,
        names_rva,
        ordinals_rva,
    ) = struct.unpack_from("<IIHHIIIIIII", raw)
    if function_count > 1000000 or name_count > 1000000:
        raise PEFormatError("export tables are unreasonably large")
    for index in range(name_count):
        name_ptr_raw = info.read_rva(names_rva + index * 4, 4)
        ordinal_raw = info.read_rva(ordinals_rva + index * 2, 2)
        if name_ptr_raw is None or ordinal_raw is None:
            raise PEFormatError("export name table leaves section data")
        func_name_rva = _u32(name_ptr_raw, 0)
        ordinal = _u16(ordinal_raw, 0)
        name_off = info.rva_to_offset(func_name_rva)
        if name_off is None:
            raise PEFormatError("export name is outside section data")
        end = info.data.find(b"\x00", name_off)
        if end < 0:
            raise PEFormatError("unterminated export name")
        name = info.data[name_off:end].decode("ascii", "replace")
        if ordinal >= function_count:
            raise PEFormatError("export ordinal is outside the export table")
        func_raw = info.read_rva(functions_rva + ordinal * 4, 4)
        if func_raw is None:
            raise PEFormatError("export function table leaves section data")
        func_rva = _u32(func_raw, 0)
        info.exports.append({"name": name, "rva": func_rva, "ordinal": ordinal + _base})
    info.exports.sort(key=lambda item: (item["rva"], item["name"], item["ordinal"]))


def _parse_relocations(info: PEInfo) -> None:
    rva, size = info.directories[5]
    if not rva or not size:
        return
    if size < 8:
        raise PEFormatError("base relocation directory is too small")
    end_rva = rva + size
    cursor = rva
    while cursor < end_rva:
        raw = info.read_rva(cursor, 8)
        if raw is None:
            raise PEFormatError("base relocation block leaves section data")
        page, block_size = struct.unpack_from("<II", raw)
        if block_size < 8 or block_size > end_rva - cursor:
            raise PEFormatError("invalid base relocation block size")
        count = (block_size - 8) // 2
        entries_raw = info.read_rva(cursor + 8, count * 2)
        if entries_raw is None:
            raise PEFormatError("base relocation entries leave section data")
        entries = []
        for index in range(count):
            value = _u16(entries_raw, index * 2)
            entries.append({"type": value >> 12, "offset": value & 0x0FFF})
        info.base_relocations.append({"page_rva": page, "entries": entries})
        cursor += block_size


def _parse_functions(info: PEInfo) -> None:
    rva, size = info.directories[3]
    has_pdata = any(section.name == ".pdata" for section in info.sections)
    if not info.is_pe32_plus or not rva or not size or not has_pdata:
        info.errors.append(
            {
                "code": "MISSING_PDATA",
                "message": "no usable .pdata RUNTIME_FUNCTION table is present",
            }
        )
        return
    if size % 12:
        raise PEFormatError("exception directory size is not a multiple of 12")
    count = size // 12
    if count > 100000:
        raise PEFormatError("exception directory is unreasonably large")
    raw = info.read_rva(rva, size)
    if raw is None:
        raise PEFormatError("exception directory leaves section data")
    functions = []
    for index in range(count):
        begin, end, unwind = struct.unpack_from("<III", raw, index * 12)
        if end <= begin:
            raise PEFormatError("RUNTIME_FUNCTION has an invalid range")
        functions.append(
            {
                "rva": begin,
                "end_rva": end,
                "size": end - begin,
                "unwind_rva": unwind,
            }
        )
    functions.sort(key=lambda item: (item["rva"], item["end_rva"], item["unwind_rva"]))
    info.functions = functions
    if not functions:
        info.errors.append(
            {
                "code": "MISSING_PDATA",
                "message": ".pdata contains no RUNTIME_FUNCTION entries",
            }
        )


def parse_pe_file(path: str) -> PEInfo:
    return load_pe(path)

