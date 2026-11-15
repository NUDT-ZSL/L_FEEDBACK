from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_event_mixer.errors import BusinessError, ErrorCode
from audio_event_mixer.generator import generate_demo_dataset
from audio_event_mixer.wavio import (
    AtomicPcmWavWriter,
    WavReader,
    inspect_wav,
    read_wav,
    write_pcm_wav,
)


_KSDATAFORMAT_SUFFIX = bytes.fromhex("00001000800000aa00389b71")


def _write_extensible_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    encoding_tag: int,
    bits: int,
) -> None:
    array = np.asarray(samples)
    if array.ndim == 1:
        array = array[:, None]
    channels = array.shape[1]
    block_align = channels * (bits // 8)
    if encoding_tag == 3:
        data = np.ascontiguousarray(array, dtype="<f4").tobytes()
    elif bits == 16:
        data = np.clip(
            np.floor(np.abs(array) * 32767.0 + 0.5) * np.sign(array),
            -32767,
            32767,
        ).astype("<i2").tobytes()
    else:
        raise ValueError("test helper supports float32 or int16")
    subformat = struct.pack("<I", encoding_tag) + _KSDATAFORMAT_SUFFIX
    fmt = struct.pack(
        "<HHIIHHHHI",
        struct.unpack("<H", b"\xfe\xff")[0],
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
        22,
        bits,
        0,
    ) + subformat
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    payload = (
        b"RIFF"
        + struct.pack("<I", 4 + len(fmt_chunk) + len(data_chunk))
        + b"WAVE"
        + fmt_chunk
        + data_chunk
    )
    path.write_bytes(payload)


class WavIoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="aem_wav_test_")
        cls.root = Path(cls.temp.name)
        cls.manifest = generate_demo_dataset(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_reads_all_required_encodings_and_shapes(self) -> None:
        cases = {
            "source_44k_s16_mono.wav": (
                "pcm_s16le",
                44100,
                1,
                16,
                (30870,),
            ),
            "source_48k_s24_stereo.wav": (
                "pcm_s24le",
                48000,
                2,
                24,
                (20160, 2),
            ),
            "source_96k_f32_mono.wav": (
                "float_s32le",
                96000,
                1,
                32,
                (30720,),
            ),
            "source_32k_s32_mono.wav": (
                "pcm_s32le",
                32000,
                1,
                32,
                (6400,),
            ),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                data = read_wav(self.root / "samples" / filename)
                self.assertEqual(data.info.encoding, expected[0])
                self.assertEqual(data.info.sample_rate, expected[1])
                self.assertEqual(data.info.channels, expected[2])
                self.assertEqual(data.info.bits_per_sample, expected[3])
                self.assertEqual(data.samples.shape, expected[4])
                self.assertEqual(data.samples.dtype, np.float64)
                self.assertTrue(np.all(np.isfinite(data.samples)))
                self.assertLessEqual(float(np.max(np.abs(data.samples))), 1.0)

    def test_chunked_reader_matches_read_all(self) -> None:
        path = self.root / "samples" / "source_48k_s24_stereo.wav"
        whole = read_wav(path).samples
        chunks = []
        with WavReader(path) as reader:
            for _, chunk in reader.iter_frames(777):
                chunks.append(chunk)
        assembled = np.concatenate(chunks, axis=0)
        np.testing.assert_array_equal(assembled, whole)

    def test_int24_roundtrip_sign_and_scale(self) -> None:
        source = np.array([-0.75, -0.25, 0.0, 0.25, 0.75])
        path = self.root / "roundtrip24.wav"
        write_pcm_wav(path, source, sample_rate=48000, bit_depth=24)
        decoded = read_wav(path)
        self.assertEqual(decoded.info.encoding, "pcm_s24le")
        np.testing.assert_allclose(decoded.samples, source, atol=2e-7)

    def test_extensible_pcm_and_float(self) -> None:
        source = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
        pcm_path = self.root / "extensible_pcm.wav"
        float_path = self.root / "extensible_float.wav"
        _write_extensible_wav(pcm_path, source, 48000, 1, 16)
        _write_extensible_wav(float_path, source, 48000, 3, 32)
        pcm = read_wav(pcm_path)
        floating = read_wav(float_path)
        self.assertEqual(pcm.info.encoding, "pcm_s16le")
        self.assertEqual(floating.info.encoding, "float_s32le")
        np.testing.assert_allclose(pcm.samples, source, atol=2e-5)
        np.testing.assert_allclose(floating.samples, source, atol=1e-7)

    def test_unsupported_compressed_format(self) -> None:
        path = self.root / "samples" / "source_unsupported_mulaw.wav"
        with self.assertRaises(BusinessError) as caught:
            inspect_wav(path)
        self.assertEqual(caught.exception.code, ErrorCode.UNSUPPORTED_FORMAT)
        with self.assertRaises(BusinessError) as caught:
            read_wav(path)
        self.assertEqual(caught.exception.code, ErrorCode.UNSUPPORTED_FORMAT)

    def test_missing_file(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            inspect_wav(self.root / "does_not_exist.wav")
        self.assertEqual(caught.exception.code, ErrorCode.FILE_NOT_FOUND)

    def test_atomic_pcm_writer_16_and_24(self) -> None:
        source = np.column_stack(
            (
                np.linspace(-0.8, 0.8, 1000),
                np.linspace(0.7, -0.7, 1000),
            )
        )
        for bits in (16, 24):
            with self.subTest(bits=bits):
                path = self.root / f"atomic_{bits}.wav"
                with AtomicPcmWavWriter(path, 48000, bits) as writer:
                    writer.write(source[:333])
                    writer.write(source[333:])
                info = inspect_wav(path)
                self.assertEqual(info.bits_per_sample, bits)
                self.assertEqual(info.frames, 1000)
                decoded = read_wav(path)
                self.assertEqual(decoded.samples.shape, (1000, 2))
                self.assertLess(
                    float(np.max(np.abs(decoded.samples - source))), 1e-3
                )

    def test_atomic_writer_aborts_without_touching_destination(self) -> None:
        destination = self.root / "protected.wav"
        sentinel = b"do-not-overwrite"
        destination.write_bytes(sentinel)
        with self.assertRaises(RuntimeError):
            with AtomicPcmWavWriter(destination, 48000, 16) as writer:
                writer.write(np.zeros((20, 2), dtype=np.float64))
                raise RuntimeError("abort")
        self.assertEqual(destination.read_bytes(), sentinel)

    def test_output_denied(self) -> None:
        parent_file = self.root / "parent_is_file"
        parent_file.write_text("x", encoding="utf-8")
        with self.assertRaises(BusinessError) as caught:
            with AtomicPcmWavWriter(parent_file / "out.wav", 48000, 16):
                pass
        self.assertEqual(caught.exception.code, ErrorCode.OUTPUT_DENIED)


if __name__ == "__main__":
    unittest.main()
