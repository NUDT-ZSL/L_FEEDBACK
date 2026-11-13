from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from pe_audit.binary import (
    OP_CALL_IMPORT,
    OP_CALL_LOCAL,
    OP_JCC,
    OP_NOP,
    OP_PAD,
    OP_PUSH_IMM,
    OP_RET,
    Instruction,
    call_import,
    call_local,
    cmp_imm,
    decode_body,
    decode_instruction,
    jcc,
    pad,
    parse_image,
    push_imm,
    ret,
)
from pe_audit.errors import AuditError
from tests.support import generate


class BytecodeTest(unittest.TestCase):
    def test_fixed_width_instruction_round_trip(self) -> None:
        samples = [
            push_imm(0x123456),
            cmp_imm(0x004242),
            call_local(7),
            call_import(2),
            jcc(3, 12),
            ret(),
            Instruction(OP_NOP),
            pad(),
        ]
        for sample in samples:
            encoded = sample.encode()
            self.assertEqual(len(encoded), 4)
            decoded = decode_instruction(encoded)
            self.assertIsNotNone(decoded)
            assert decoded is not None
            self.assertEqual(decoded.opcode, sample.opcode)

    def test_encoder_rejects_bad_operands(self) -> None:
        with self.assertRaises(ValueError):
            push_imm(1 << 24)
        with self.assertRaises(ValueError):
            jcc(256)
        with self.assertRaises(ValueError):
            Instruction(0xFF)

    def test_padding_is_excluded_from_effective_stream(self) -> None:
        body = pad().encode() * 3 + push_imm(9).encode() + pad().encode()
        decoded = decode_body(body, "f")
        self.assertEqual(decoded.invalid_bytes, 0)
        self.assertEqual(len(decoded.instructions), 1)
        self.assertEqual(decoded.instructions[0].value, 9)

    def test_illegal_words_are_counted_and_decoding_continues(self) -> None:
        body = (
            b"\xff\xff\xff\xff"
            + call_import(0).encode()
            + b"\x99\x00\x00\x00"
            + ret().encode()
        )
        decoded = decode_body(body, "f")
        self.assertEqual(decoded.invalid_bytes, 8)
        self.assertEqual([item.opcode for item in decoded.instructions], [
            OP_CALL_IMPORT,
            OP_RET,
        ])
        self.assertEqual(len(decoded.parse_issues), 2)

    def test_truncated_nonzero_tail_is_invalid(self) -> None:
        body = push_imm(1).encode() + b"\x01\x02\x03"
        decoded = decode_body(body, "f")
        self.assertEqual(decoded.invalid_bytes, 3)
        self.assertEqual(len(decoded.instructions), 1)

    def test_trailing_zero_alignment_is_not_invalid(self) -> None:
        body = ret().encode() + b"\x00\x00"
        decoded = decode_body(body, "f")
        self.assertEqual(decoded.invalid_bytes, 0)
        self.assertEqual(len(decoded.instructions), 1)


class FixtureFormatTest(unittest.TestCase):
    def test_build_parse_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=11)
            image = parse_image((output / "patch_before.bin").read_bytes())
            self.assertEqual(
                image.local_names,
                sorted(
                    [
                        "dispatch",
                        "entry_main",
                        "guarded_copy",
                        "isolated",
                        "orphan_copy",
                        "unsafe_copy",
                        "walk_a",
                        "walk_b",
                    ]
                ),
            )
            self.assertEqual(
                [(item.module, item.name) for item in image.imports],
                sorted((item.module, item.name) for item in image.imports),
            )
            self.assertGreater(image.entry_rva, 0)

    def test_name_table_indices_are_sorted_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate(first, seed=5)
            generate(second, seed=101)
            image_a = parse_image(
                (Path(first) / "patch_before.bin").read_bytes()
            )
            image_b = parse_image(
                (Path(second) / "patch_after.bin").read_bytes()
            )
            self.assertEqual(image_a.local_names, image_b.local_names)

    def test_parse_rejects_bad_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            blob = bytearray((output / "patch_before.bin").read_bytes())
            blob[0:2] = b"ZZ"
            with self.assertRaises(AuditError) as caught:
                parse_image(bytes(blob))
            self.assertEqual(caught.exception.code, "BINARY_PARSE_ERROR")

    def test_parse_rejects_truncated_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            data = (output / "patch_before.bin").read_bytes()
            with self.assertRaises(AuditError) as caught:
                parse_image(data[:100])
            self.assertEqual(caught.exception.code, "BINARY_PARSE_ERROR")


if __name__ == "__main__":
    unittest.main()
