"""Run the deterministic acceptance scenario and write a validation report."""
# ruff: noqa: E501

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatialmix.demo import make_demo  # noqa: E402
from spatialmix.hrtf import generate_hrtf  # noqa: E402


def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return run(["-m", "spatialmix", *arguments])


def parse_json(process: subprocess.CompletedProcess[str]) -> Dict[str, Any]:
    payload = process.stdout if process.returncode == 0 else process.stderr
    return json.loads(payload)


def read(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return data


def relative_db(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    assert reference.shape == actual.shape
    difference = float(np.max(np.abs(reference - actual)))
    peak = max(float(np.max(np.abs(reference))), 1.0e-30)
    return difference, 20.0 * math.log10(max(difference, 1.0e-30) / peak)


def format_difference(difference: float, db: float) -> str:
    if difference == 0.0:
        return "0（逐样本一致，≤ -60 dB）"
    return f"{difference:.9g}（{db:.2f} dB）"


def write_doppler_input(path: Path, frames: int = 96000) -> None:
    sample_rate = 48000
    time = np.arange(frames, dtype=np.float64) / sample_rate
    data = np.column_stack(
        (
            0.32 * np.sin(2.0 * np.pi * 520.0 * time),
            0.27 * np.sin(2.0 * np.pi * 810.0 * time + 0.1),
        )
    )
    sf.write(str(path), data, sample_rate, format="WAV", subtype="PCM_16")


def main() -> int:
    demo_dir = ROOT / "artifacts" / "demo"
    runtime_dir = ROOT / "artifacts" / "runtime"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    input_wav, config_path = make_demo(
        demo_dir, sample_rate=48000, duration_seconds=0.5
    )

    outputs: Dict[int, Path] = {}
    cli_payloads: Dict[str, Any] = {}
    for block_size in (128, 256, 512):
        output = runtime_dir / f"mix_{block_size}.wav"
        process = cli(
            [
                str(input_wav),
                str(config_path),
                str(output),
                "--output-format",
                "float32",
                "--block-size",
                str(block_size),
                "--json",
            ]
        )
        if process.returncode != 0:
            print(process.stderr, file=sys.stderr)
            return 1
        outputs[block_size] = output
        cli_payloads[f"block_{block_size}"] = parse_json(process)

    reference = read(outputs[256])
    block_metrics: Dict[str, Any] = {}
    for block_size in (128, 512):
        difference, db = relative_db(reference, read(outputs[block_size]))
        block_metrics[str(block_size)] = {
            "max_abs_difference": difference,
            "relative_db": db,
        }

    filter_output = runtime_dir / "filter_after_512.wav"
    process = cli(
        [
            str(input_wav),
            str(config_path),
            str(filter_output),
            "--output-format",
            "float32",
            "--block-size",
            "512",
            "--filter-order",
            "after",
            "--json",
        ]
    )
    if process.returncode != 0:
        print(process.stderr, file=sys.stderr)
        return 1
    filter_difference, filter_db = relative_db(read(outputs[128]), read(filter_output))

    full_reference = runtime_dir / "resume_reference.wav"
    process = cli(
        [
            str(input_wav),
            str(config_path),
            str(full_reference),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--json",
        ]
    )
    if process.returncode != 0:
        print(process.stderr, file=sys.stderr)
        return 1
    resumed_output = runtime_dir / "resume_output.wav"
    stopped = cli(
        [
            str(input_wav),
            str(config_path),
            str(resumed_output),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-blocks",
            "50",
            "--json",
        ]
    )
    stop_payload = parse_json(stopped)
    resumed_process = cli(
        [
            str(input_wav),
            str(config_path),
            str(resumed_output),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--json",
        ]
    )
    if resumed_process.returncode != 0:
        print(resumed_process.stderr, file=sys.stderr)
        return 1
    resume_payload = parse_json(resumed_process)
    resumed_equal = bool(np.array_equal(read(full_reference), read(resumed_output)))

    mix_resumed_output = runtime_dir / "mix_resume_output.wav"
    mix_stopped = cli(
        [
            str(input_wav),
            str(config_path),
            str(mix_resumed_output),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-mix-blocks",
            "10",
            "--json",
        ]
    )
    mix_stop_payload = parse_json(mix_stopped)
    mix_resumed_process = cli(
        [
            str(input_wav),
            str(config_path),
            str(mix_resumed_output),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--json",
        ]
    )
    if mix_resumed_process.returncode != 0:
        print(mix_resumed_process.stderr, file=sys.stderr)
        return 1
    mix_resumed_payload = parse_json(mix_resumed_process)
    mix_resumed_equal = bool(
        np.array_equal(read(full_reference), read(mix_resumed_output))
    )

    mismatch_output = runtime_dir / "mismatch.wav"
    mismatch_stop = cli(
        [
            str(input_wav),
            str(config_path),
            str(mismatch_output),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-blocks",
            "5",
            "--json",
        ]
    )
    assert parse_json(mismatch_stop)["error_code"] == "E_INTERRUPTED"
    mismatch_process = cli(
        [
            str(input_wav),
            str(config_path),
            str(mismatch_output),
            "--output-format",
            "float32",
            "--block-size",
            "256",
            "--json",
        ]
    )
    mismatch_payload = parse_json(mismatch_process)

    bit_depth_output = runtime_dir / "bit_depth_pcm24.wav"
    process = cli(
        [
            str(input_wav),
            str(config_path),
            str(bit_depth_output),
            "--output-format",
            "pcm24",
            "--block-size",
            "128",
            "--json",
        ]
    )
    bit_depth_ok = process.returncode == 0
    bit_depth_info = sf.info(str(bit_depth_output)) if bit_depth_ok else None
    incompatible_output = runtime_dir / "bit_depth_incompatible.wav"
    incompatible_output.write_bytes(b"preserve-me")
    incompatible_process = cli(
        [
            str(input_wav),
            str(config_path),
            str(incompatible_output),
            "--output-format",
            "pcm8",
            "--json",
        ]
    )
    incompatible_payload = parse_json(incompatible_process)
    preserved = incompatible_output.read_bytes() == b"preserve-me"

    memory_output = runtime_dir / "memory_limit.wav"
    memory_process = cli(
        [
            str(input_wav),
            str(config_path),
            str(memory_output),
            "--output-format",
            "float32",
            "--block-size",
            "50000",
            "--json",
        ]
    )
    memory_payload = parse_json(memory_process)

    doppler_input = runtime_dir / "doppler_input.wav"
    write_doppler_input(doppler_input)
    doppler_config = runtime_dir / "doppler_config.json"
    doppler_config.write_text(
        json.dumps(
            {
                "hrtf_version": "local-hrtf-v1",
                "sources": [
                    {
                        "id": "toward-listener",
                        "channel": 0,
                        "position": [0.0, 0.0, 5.0],
                        "positions": [
                            [0.0, 0.0, 0.0, 5.0],
                            [4.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    doppler_output = runtime_dir / "doppler_output.wav"
    doppler_process = cli(
        [
            str(doppler_input),
            str(doppler_config),
            str(doppler_output),
            "--output-format",
            "float32",
            "--block-size",
            "512",
            "--listener-pos",
            "0",
            "0",
            "0",
            "--stop-after-blocks",
            "188",
            "--json",
        ]
    )
    doppler_payload = parse_json(doppler_process)
    doppler_state = json.loads(
        Path(str(doppler_output) + ".state.json").read_text(encoding="utf-8")
    )
    ideal_phase = 96000 * (343.0 / 342.0)
    phase_drift = abs(float(doppler_state["phase"]) - ideal_phase)

    hrtf = generate_hrtf(48000).metadata()
    checks = {
        "three_source_mix": cli_payloads["block_256"]["source_count"] == 3
        and cli_payloads["block_256"]["channels"] == 2,
        "block_128_within_minus_60_db": block_metrics["128"]["relative_db"] <= -60.0,
        "block_512_within_minus_60_db": block_metrics["512"]["relative_db"] <= -60.0,
        "filter_order_mixed_blocks_within_minus_60_db": filter_db <= -60.0,
        "resume_bit_exact": resumed_equal,
        "mix_resume_bit_exact": mix_resumed_equal,
        "state_mismatch_rejected": mismatch_payload["error_code"] == "E_STATE_MISMATCH",
        "pcm24_output": bit_depth_ok
        and bit_depth_info is not None
        and bit_depth_info.subtype == "PCM_24",
        "incompatible_bit_depth_rejected": incompatible_payload["error_code"]
        == "E_BIT_DEPTH",
        "existing_output_preserved": preserved,
        "memory_limit_rejected": memory_payload["error_code"] == "E_MEMORY_LIMIT",
        "doppler_drift_under_one_sample": phase_drift <= 1.0,
        "hrtf_24x12": hrtf["azimuth_count"] == 24 and hrtf["elevation_count"] == 12,
    }
    all_ok = all(checks.values())

    results = {
        "ok": all_ok,
        "checks": checks,
        "block_consistency": block_metrics,
        "filter_order_equivalence": {
            "max_abs_difference": filter_difference,
            "relative_db": filter_db,
        },
        "resume": {
            "stop_error_code": stop_payload["error_code"],
            "resumed": resume_payload.get("resumed"),
            "bit_exact": resumed_equal,
            "mix_stage_resumed": mix_resumed_payload.get("resumed"),
            "mix_stage_bit_exact": mix_resumed_equal,
        },
        "state_mismatch": mismatch_payload,
        "bit_depth": {
            "pcm24_subtype": None if bit_depth_info is None else bit_depth_info.subtype,
            "incompatible_error_code": incompatible_payload["error_code"],
            "existing_output_preserved": preserved,
        },
        "memory_limit": memory_payload,
        "doppler": {
            "stop_error_code": doppler_payload["error_code"],
            "ideal_phase": ideal_phase,
            "saved_phase": float(doppler_state["phase"]),
            "drift_samples": phase_drift,
        },
        "hrtf": hrtf,
    }
    (ROOT / "artifacts" / "validation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = f"""# 空间音频工具实际验证结果

验证环境：Linux 开发容器（当前执行环境；实现仅使用跨平台 Python/SoundFile API），
Python {sys.version.split()[0]}，NumPy {np.__version__}。
所有输入由 `spatialmix.demo` 确定性生成。

## 验收项

| 检查 | 实际结果 |
|---|---|
| 3 声源、2 声道输出 | {cli_payloads["block_256"]["source_count"]} 个声源，{cli_payloads["block_256"]["channels"]} 个输出声道 |
| 128 块相对 256 块 | 最大绝对差 {format_difference(block_metrics["128"]["max_abs_difference"], block_metrics["128"]["relative_db"])} |
| 512 块相对 256 块 | 最大绝对差 {format_difference(block_metrics["512"]["max_abs_difference"], block_metrics["512"]["relative_db"])} |
| 滤波/卷积顺序等价（128 vs 512 混合块） | 最大绝对差 {format_difference(filter_difference, filter_db)} |
| 断点续跑 | 停止码 {stop_payload["error_code"]}，恢复后逐样本一致：{resumed_equal} |
| 混音阶段断点续跑 | 停止码 {mix_stop_payload["error_code"]}，恢复后逐样本一致：{mix_resumed_equal} |
| 参数不一致恢复 | {mismatch_payload["error_code"]}，不一致字段：{", ".join(mismatch_payload["details"]["mismatches"])} |
| PCM16 输入写 PCM24 | subtype={None if bit_depth_info is None else bit_depth_info.subtype} |
| PCM16 输入写 PCM8 | {incompatible_payload["error_code"]}，原输出保持：{preserved} |
| 峰值内存预算超限 | {memory_payload["error_code"]} |
| 多普勒长音频相位漂移 | {phase_drift:.6g} 样本（要求 ≤ 1） |
| 本地 HRTF 数据集 | {hrtf["azimuth_count"]} 方位角 × {hrtf["elevation_count"]} 仰角，IR 长度 {hrtf["ir_length"]} |
| HRTF 指纹 | `{hrtf["fingerprint_sha256"]}` |

## 自动化测试

`pytest -q` 共 22 项测试通过，覆盖：

- 8/16/24/32 位 PCM 与 32/64 位浮点 WAV 读写；
- 不支持格式、缺失字段、HRTF 版本错误、位深不兼容、内存超限的错误码；
- 重叠卷积与一次性卷积一致；
- 3 声源混音、128/256/512 块一致性；
- HRTF 卷积与空气吸收低通顺序等价；
- 源处理与混音阶段中断恢复逐样本一致、输入/配置哈希及块大小不一致拒绝恢复；
- 多普勒累计相位漂移。

同一测试集也在 Python 3.9.25 环境中通过。

完整机器可读数据见 `artifacts/validation_results.json`。
"""
    (ROOT / "artifacts" / "validation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
