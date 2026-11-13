from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

from .pe import PEInfo


class DecodeError(Exception):
    pass


REG64 = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
]
REG32 = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
]
REG16 = [
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di",
    "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
]
REG8 = [
    "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
]
REG8_LEGACY = ["al", "cl", "dl", "bl", "ah", "ch", "dh", "bh"]

GROUP1 = {0: "add", 1: "or", 2: "adc", 3: "sbb", 4: "and", 5: "sub", 6: "xor", 7: "cmp"}
GROUP2 = {0: "rol", 1: "ror", 2: "rcl", 3: "rcr", 4: "shl", 5: "shr", 6: "sal", 7: "sar"}
JCC = {
    0x70: "jo", 0x71: "jno", 0x72: "jb", 0x73: "jae", 0x74: "je", 0x75: "jne",
    0x76: "jbe", 0x77: "ja", 0x78: "js", 0x79: "jns", 0x7A: "jp", 0x7B: "jnp",
    0x7C: "jl", 0x7D: "jge", 0x7E: "jle", 0x7F: "jg",
}
JCC2 = {
    0x80: "jo", 0x81: "jno", 0x82: "jb", 0x83: "jae", 0x84: "je", 0x85: "jne",
    0x86: "jbe", 0x87: "ja", 0x88: "js", 0x89: "jns", 0x8A: "jp", 0x8B: "jnp",
    0x8C: "jl", 0x8D: "jge", 0x8E: "jle", 0x8F: "jg",
}


@dataclass
class Operand:
    kind: str
    reg: Optional[str] = None
    base: Optional[str] = None
    index: Optional[str] = None
    scale: int = 1
    disp: int = 0
    rip_relative: bool = False
    value: Optional[int] = None


@dataclass
class Instruction:
    rva: int
    size: int
    mnemonic: str
    op1: Optional[Operand] = None
    op2: Optional[Operand] = None
    opcode: int = 0
    group: Optional[int] = None
    rex_w: bool = False
    operand_size: int = 32


def _reg(code: int, width: int, rex_present: bool) -> str:
    if width == 8:
        return (REG8 if rex_present else REG8_LEGACY)[code & 7] if code < 8 else REG8[code]
    if width == 16:
        return REG16[code]
    if width == 32:
        return REG32[code]
    return REG64[code]


def _canonical_register(name: str) -> str:
    mapping = {}
    for index in range(16):
        mapping[REG64[index]] = REG64[index]
        mapping[REG32[index]] = REG64[index]
        mapping[REG16[index]] = REG64[index]
        mapping[REG8[index]] = REG64[index]
    for index, legacy_name in enumerate(REG8_LEGACY):
        mapping[legacy_name] = REG64[index]
    return mapping.get(name, name)


class _Reader:
    def __init__(self, data: bytes, rva: int, is64: bool):
        self.data = data
        self.pos = 0
        self.start_rva = rva
        self.is64 = is64
        self.rex = 0
        self.prefix_66 = False
        self.prefix_67 = False
        self.rex_present = False

    def available(self, count: int) -> bool:
        return self.pos + count <= len(self.data)

    def u8(self) -> int:
        if not self.available(1):
            raise DecodeError("truncated instruction")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def i8(self) -> int:
        value = self.u8()
        return value - 256 if value & 0x80 else value

    def u16(self) -> int:
        if not self.available(2):
            raise DecodeError("truncated instruction")
        value = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return value

    def u32(self) -> int:
        if not self.available(4):
            raise DecodeError("truncated instruction")
        value = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return value

    def i32(self) -> int:
        value = self.u32()
        return value - 0x100000000 if value & 0x80000000 else value

    def u64(self) -> int:
        if not self.available(8):
            raise DecodeError("truncated instruction")
        value = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return value

    @property
    def next_rva(self) -> int:
        return self.start_rva + self.pos

    def modrm(self, operand_width: int) -> Tuple[Operand, int]:
        if not self.available(1):
            raise DecodeError("truncated ModRM")
        byte = self.u8()
        mod = byte >> 6
        reg_code = ((byte >> 3) & 7) | (8 if self.rex & 4 else 0)
        rm_code = (byte & 7) | (8 if self.rex & 1 else 0)
        reg = Operand("reg", reg=_reg(reg_code, operand_width, self.rex_present))
        if mod == 3:
            rm = Operand("reg", reg=_reg(rm_code, operand_width, self.rex_present))
            return rm, reg_code

        addr_width = 16 if (self.prefix_67 and not self.is64) else (32 if self.prefix_67 else 64)
        base = None
        index = None
        scale = 1
        disp = 0
        rip_relative = False
        if addr_width == 16:
            if mod == 0 and rm_code == 6:
                disp = struct.unpack_from("<H", self.data, self.pos)[0]
                self.pos += 2
            else:
                base = _reg(rm_code & 7, 16, self.rex_present)
                if mod == 1:
                    disp = self.i8()
                elif mod == 2:
                    disp = struct.unpack_from("<H", self.data, self.pos)[0]
                    self.pos += 2
        elif rm_code & 7 == 4:
            sib = self.u8()
            sscale = 1 << (sib >> 6)
            sindex = ((sib >> 3) & 7) | (8 if self.rex & 2 else 0)
            sbase = (sib & 7) | (8 if self.rex & 1 else 0)
            if sindex & 7 != 4:
                index = _reg(sindex, 64 if self.is64 else 32, self.rex_present)
                scale = sscale
            if (sbase & 7) == 5 and mod == 0:
                disp = self.i32()
                if self.is64:
                    rip_relative = False
            else:
                base = _reg(sbase, 64 if self.is64 else 32, self.rex_present)
            if mod == 1:
                disp += self.i8()
            elif mod == 2:
                disp += self.i32()
        elif (rm_code & 7) == 5 and mod == 0:
            disp = self.i32()
            if self.is64:
                rip_relative = True
            elif self.rex & 2:
                index = _reg(8, 32, self.rex_present)
            else:
                index = None
        else:
            base = _reg(rm_code, 64 if self.is64 else 32, self.rex_present)
            if mod == 1:
                disp = self.i8()
            elif mod == 2:
                disp = self.i32()
        return (
            Operand(
                "mem",
                base=base,
                index=index,
                scale=scale,
                disp=disp,
                rip_relative=rip_relative,
            ),
            reg_code,
        )


def decode_one(data: bytes, rva: int, is64: bool) -> Instruction:
    reader = _Reader(data, rva, is64)
    while True:
        byte = reader.u8()
        if byte in (0x66, 0x67, 0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65):
            if byte == 0x66:
                reader.prefix_66 = True
            elif byte == 0x67:
                reader.prefix_67 = True
            continue
        if is64 and 0x40 <= byte <= 0x4F:
            reader.rex = byte
            reader.rex_present = True
            continue
        break

    opcode = byte
    rex_w = bool(reader.rex & 8)
    operand_size = 64 if rex_w else (16 if reader.prefix_66 else 32)
    if not is64:
        operand_size = 16 if reader.prefix_66 else 32

    def inst(mnemonic: str, op1=None, op2=None, group=None, opcode_value=None):
        return Instruction(
            rva=rva,
            size=reader.pos,
            mnemonic=mnemonic,
            op1=op1,
            op2=op2,
            opcode=opcode_value if opcode_value is not None else opcode,
            group=group,
            rex_w=rex_w,
            operand_size=operand_size,
        )

    if 0x50 <= opcode <= 0x57:
        reg = _reg((opcode - 0x50) + (8 if reader.rex & 1 else 0), 64 if is64 else 32, reader.rex_present)
        return inst("push", Operand("reg", reg=reg), opcode_value=opcode)
    if 0x58 <= opcode <= 0x5F:
        reg = _reg((opcode - 0x58) + (8 if reader.rex & 1 else 0), 64 if is64 else 32, reader.rex_present)
        return inst("pop", Operand("reg", reg=reg), opcode_value=opcode)
    if opcode == 0x68:
        value = reader.i32()
        return inst("push", Operand("imm", value=value))
    if opcode == 0x6A:
        return inst("push", Operand("imm", value=reader.i8()))
    if opcode == 0x90:
        return inst("nop")
    if opcode == 0xCC:
        return inst("int3")
    if opcode == 0xC3:
        return inst("ret")
    if opcode in (0xC9, 0xC2, 0xCA, 0xCB):
        if opcode == 0xC2:
            reader.u16()
        elif opcode == 0xCA:
            reader.u16()
        return inst("ret")
    if opcode == 0xE8:
        displacement = reader.i32()
        return inst("call", Operand("imm", value=reader.next_rva + displacement))
    if opcode == 0xE9:
        displacement = reader.i32()
        return inst("jmp", Operand("imm", value=reader.next_rva + displacement))
    if opcode == 0xEB:
        displacement = reader.i8()
        return inst("jmp", Operand("imm", value=reader.next_rva + displacement))
    if 0x70 <= opcode <= 0x7F:
        displacement = reader.i8()
        return inst(JCC[opcode], Operand("imm", value=reader.next_rva + displacement))
    if opcode == 0x0F:
        op2 = reader.u8()
        if 0x80 <= op2 <= 0x8F:
            displacement = reader.i32()
            return inst(JCC2[op2], Operand("imm", value=reader.next_rva + displacement))
        if op2 in (0xB6, 0xB7, 0xBE, 0xBF):
            rm, reg_code = reader.modrm(8 if op2 in (0xB6, 0xBE) else 16)
            return inst(
                "movx",
                rm,
                Operand("reg", reg=_reg(reg_code, operand_size, reader.rex_present)),
            )
        if op2 in (0x10, 0x11, 0x28, 0x29, 0x2E, 0x2F, 0x42, 0x43, 0x44, 0x45,
                   0x4A, 0x4B, 0x4C, 0x4D, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                   0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x6E, 0x6F, 0x7E, 0x7F,
                   0xAE, 0xC2, 0xC4, 0xC5, 0xC6, 0xD6, 0xD7,
                   0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0, 0xE1, 0xE2, 0xE3,
                   0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED,
                   0xEE, 0xEF, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
                   0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF):
            reader.modrm(operand_size)
            return inst("sse" if op2 < 0x80 else "simd")
        if op2 == 0x1F:
            rm, _ = reader.modrm(operand_size)
            return inst("nop", rm)
        if op2 in (0x05, 0x0B, 0x31, 0xA2):
            return inst("special")
        if op2 == 0xAF:
            rm, reg_code = reader.modrm(operand_size)
            return inst("imul", rm, Operand("reg", reg=_reg(reg_code, operand_size, reader.rex_present)))
        raise DecodeError("unsupported two-byte opcode 0x0f 0x%02x" % op2)
    if opcode in (0x88, 0x89, 0x8A, 0x8B):
        rm, reg_code = reader.modrm(8 if opcode in (0x88, 0x8A) else operand_size)
        reg = Operand("reg", reg=_reg(reg_code, 8 if opcode in (0x88, 0x8A) else operand_size, reader.rex_present))
        if opcode in (0x88, 0x89):
            return inst("mov", rm, reg)
        return inst("mov", reg, rm)
    if opcode == 0x8D:
        rm, reg_code = reader.modrm(operand_size)
        return inst("lea", Operand("reg", reg=_reg(reg_code, operand_size, reader.rex_present)), rm)
    if opcode in (0x01, 0x03, 0x29, 0x2B, 0x31, 0x33, 0x21, 0x23, 0x09, 0x0B,
                  0x39, 0x3B, 0x85):
        base_mnemonic = {
            0x01: "add", 0x03: "add", 0x29: "sub", 0x2B: "sub",
            0x31: "xor", 0x33: "xor", 0x21: "and", 0x23: "and",
            0x09: "or", 0x0B: "or", 0x39: "cmp", 0x3B: "cmp", 0x85: "test",
        }[opcode]
        rm, reg_code = reader.modrm(operand_size)
        reg = Operand("reg", reg=_reg(reg_code, operand_size, reader.rex_present))
        if opcode in (0x01, 0x29, 0x31, 0x21, 0x09, 0x39, 0x85):
            return inst(base_mnemonic, rm, reg)
        return inst(base_mnemonic, reg, rm)
    if opcode in (0x81, 0x83, 0xC1, 0xD1):
        rm, reg_code = reader.modrm(operand_size)
        value = reader.i32() if opcode == 0x81 else reader.i8()
        group = reg_code & 7
        name = GROUP1.get(group, "group1") if opcode in (0x81, 0x83) else GROUP2.get(group, "group2")
        return inst(name, rm, Operand("imm", value=value), group=group)
    if opcode in (0xC6, 0xC7):
        rm, _ = reader.modrm(8 if opcode == 0xC6 else operand_size)
        value = reader.i8() if opcode == 0xC6 else reader.i32()
        return inst("mov", rm, Operand("imm", value=value))
    if 0xB8 <= opcode <= 0xBF:
        reg = _reg((opcode - 0xB8) + (8 if reader.rex & 1 else 0), operand_size, reader.rex_present)
        value = reader.u64() if rex_w else reader.u32()
        if is64 and not rex_w:
            value &= 0xFFFFFFFF
        return inst("mov", Operand("reg", reg=reg), Operand("imm", value=value))
    if 0xB0 <= opcode <= 0xB7:
        reg = _reg((opcode - 0xB0) + (8 if reader.rex & 1 else 0), 8, reader.rex_present)
        return inst("mov", Operand("reg", reg=reg), Operand("imm", value=reader.u8()))
    if opcode == 0x63:
        rm, reg_code = reader.modrm(operand_size)
        return inst("movsxd", rm, Operand("reg", reg=_reg(reg_code, operand_size, reader.rex_present)))
    if opcode in (0xF7, 0xFF):
        rm, reg_code = reader.modrm(operand_size)
        group = reg_code & 7
        if opcode == 0xFF:
            name = {0: "inc", 1: "dec", 2: "call", 3: "call", 4: "jmp", 5: "jmp", 6: "push"}.get(group, "group5")
        else:
            name = {0: "test", 1: "test", 2: "not", 3: "neg", 4: "mul", 5: "imul", 6: "div", 7: "idiv"}.get(group, "group3")
        return inst(name, rm, group=group)
    if opcode in (0x69, 0x6B):
        rm, reg_code = reader.modrm(operand_size)
        value = reader.i32() if opcode == 0x69 else reader.i8()
        return inst(
            "imul",
            rm,
            Operand("reg", reg=_reg(reg_code, operand_size, reader.rex_present)),
            opcode_value=opcode,
        )
    if opcode in (0xC0, 0xD0, 0xD2, 0xD3):
        rm, reg_code = reader.modrm(8)
        name = {0xC0: "rol", 0xD0: "rol", 0xD2: "rcl", 0xD3: "rcr"}[opcode]
        return inst(name, rm, Operand("imm", value=reader.u8()))
    if opcode == 0xA8:
        return inst("test", Operand("reg", reg="al"), Operand("imm", value=reader.u8()))
    if opcode == 0xA9:
        return inst("test", Operand("reg", reg="rax" if is64 else "eax"), Operand("imm", value=reader.u32()))
    if opcode in (0x98, 0x99):
        return inst("cwd" if opcode == 0x99 else "cbw")
    if opcode in (0x9C, 0x9D):
        return inst("pushf" if opcode == 0x9C else "popf")
    if opcode in (0xFC, 0xFD, 0xF8, 0xF9, 0xF5, 0xF4, 0xFA, 0xFB):
        return inst("flag")
    raise DecodeError("unsupported opcode 0x%02x" % opcode)


def decode_function(pe: PEInfo, rva: int, size: int, is64: bool) -> Tuple[list, Optional[str]]:
    off = pe.rva_to_offset(rva, size)
    if off is None or off + size > len(pe.data):
        return [], "function bytes are not backed by the PE file"
    code = pe.data[off : off + size]
    instructions = []
    cursor = 0
    while cursor < len(code):
        try:
            instruction = decode_one(code[cursor:], rva + cursor, is64)
        except DecodeError as exc:
            return instructions, str(exc)
        if instruction.size <= 0:
            return instructions, "zero-length instruction"
        instructions.append(instruction)
        cursor += instruction.size
        if instruction.mnemonic == "ret":
            break
    return instructions, None


def resolve_call_target(
    pe: PEInfo,
    target_rva: int,
    functions_by_rva,
    depth: int = 0,
):
    if target_rva in functions_by_rva:
        return ("function", functions_by_rva[target_rva])
    if target_rva in pe.import_by_iat:
        return ("import", pe.import_by_iat[target_rva])
    function = None
    for candidate in functions_by_rva.values():
        if candidate.rva <= target_rva < candidate.rva + max(1, candidate.size):
            function = candidate
            break
    if function is not None:
        return ("function", function)
    if depth >= 6:
        return ("unknown", None)
    raw = pe.read_rva(target_rva, 16)
    if raw is None:
        return ("unknown", None)
    if len(raw) >= 6 and raw[0] == 0xE9:
        displacement = struct.unpack_from("<i", raw, 1)[0]
        next_rva = target_rva + 5
        return resolve_call_target(pe, next_rva + displacement, functions_by_rva, depth + 1)
    if len(raw) >= 6 and raw[0] == 0xFF and raw[1] == 0x25:
        displacement = struct.unpack_from("<i", raw, 2)[0]
        slot_rva = target_rva + 6 + displacement
        if slot_rva in pe.import_by_iat:
            return ("import", pe.import_by_iat[slot_rva])
    if len(raw) >= 5 and raw[0] == 0xFF and raw[1] == 0x25:
        slot_rva = struct.unpack_from("<I", raw, 2)[0]
        if slot_rva in pe.import_by_iat:
            return ("import", pe.import_by_iat[slot_rva])
    return ("unknown", None)

