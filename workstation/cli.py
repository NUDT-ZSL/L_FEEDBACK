"""命令行入口：``python -m workstation <command>``。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .exporter import ExportConfig, ExportManager
from .fixtures import generate_workspace
from .mixer import PreviewEngine
from .project import Project


def _write_json(path: Path, payload: dict | list) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def cmd_init(args: argparse.Namespace) -> int:
    info = generate_workspace(Path(args.workspace), int(args.rate))
    print(f"已生成确定性素材与配置：{info['script_path']}")
    print(f"混音配置：{info['mix_path']}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    project = Project.build(
        Path(args.workspace),
        Path(args.script),
        Path(args.mix),
        default_fade_in=float(args.fade_in),
        default_fade_out=float(args.fade_out),
    )
    report = {
        "project_rate": project.project_rate,
        "timeline_length_frames": project.timeline.length,
        "alignment": project.alignment_report,
        "errors": project.library.errors,
        "error_codes": project.library.error_codes(),
        "tracks": {name: cfg.to_dict() for name, cfg in sorted(project.tracks.items())},
    }
    project.save(Path(args.state))
    _write_json(Path(args.report), report)
    print(f"工程状态：{args.state}")
    print(f"对齐/错误报告：{args.report}")
    print(f"错误码：{project.library.error_codes()}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    restored = Project.restore(Path(args.workspace), Path(args.state))
    project = restored.project
    if restored.verify_errors:
        print(f"恢复校验告警：{restored.verify_errors}", file=sys.stderr)
    engine = PreviewEngine(project.mixer)
    pcm = engine.preview(int(args.start), int(args.frames))
    if args.out:
        pcm.tofile(args.out)
        print(f"PCM 预览（float64 立体声交错）：{args.out}，帧数={pcm.shape[0]}")
    else:
        preview = {
            "start_frame": int(args.start),
            "frame_count": int(args.frames),
            "shape": list(pcm.shape),
            "max_abs": float(np.max(np.abs(pcm))) if pcm.size else 0.0,
            "first_samples": pcm[: min(4, pcm.shape[0])].tolist(),
            "clips_touched": engine.clip_touches,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    restored = Project.restore(Path(args.workspace), Path(args.state))
    project = restored.project
    if restored.verify_errors:
        print(f"恢复校验告警：{restored.verify_errors}", file=sys.stderr)
    total = project.timeline.length if args.frames is None else int(args.frames)
    config = ExportConfig(
        total_frames=total,
        channels=2,
        block_size=int(args.block_size),
        subtype=args.subtype,
        samplerate=project.project_rate,
    )
    manager = ExportManager(
        project.mixer, Path(args.out), Path(args.export_state), config
    )
    result = manager.run()
    _write_json(Path(args.export_report), result)
    print(f"导出完成：{result['output_path']}（resumed={result['resumed']}）")
    print(f"导出状态：{args.export_state}，报告：{args.export_report}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    generate_workspace(workspace, int(args.rate))
    script = workspace / "script.json"
    mix = workspace / "mix.json"
    state = workspace / "project_state.json"
    report = workspace / "report.json"
    build_args = argparse.Namespace(
        workspace=workspace,
        script=script,
        mix=mix,
        state=state,
        report=report,
        fade_in=0.0,
        fade_out=0.0,
    )
    cmd_build(build_args)
    export_args = argparse.Namespace(
        workspace=workspace,
        state=state,
        out=workspace / "master.wav",
        export_state=workspace / "export_state.json",
        export_report=workspace / "export_report.json",
        frames=None,
        block_size=512,
        subtype="PCM_16",
    )
    return cmd_export(export_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m workstation",
        description="离线对白剪辑工作站",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="确定性生成素材、台词表与混音配置")
    init.add_argument("workspace")
    init.add_argument("--rate", default=48000)
    init.set_defaults(func=cmd_init)

    build = sub.add_parser("build", help="解析台词表、对齐时间轴并生成工程状态")
    build.add_argument("workspace")
    build.add_argument("--script", default=None)
    build.add_argument("--mix", default=None)
    build.add_argument("--state", default=None)
    build.add_argument("--report", default=None)
    build.add_argument("--fade-in", dest="fade_in", default=0.0)
    build.add_argument("--fade-out", dest="fade_out", default=0.0)
    build.set_defaults(func=_with_defaults(build, "build"))

    preview = sub.add_parser("preview", help="按起始帧与帧数输出实时混音预览")
    preview.add_argument("workspace")
    preview.add_argument("--state", default=None)
    preview.add_argument("--start", default=0)
    preview.add_argument("--frames", default=512)
    preview.add_argument("--out", default=None)
    preview.set_defaults(func=_with_defaults(preview, "preview"))

    export = sub.add_parser("export", help="可恢复分块导出多声道 WAV 母版")
    export.add_argument("workspace")
    export.add_argument("--state", default=None)
    export.add_argument("--out", default=None)
    export.add_argument("--export-state", dest="export_state", default=None)
    export.add_argument("--export-report", dest="export_report", default=None)
    export.add_argument("--frames", default=None)
    export.add_argument("--block-size", dest="block_size", default=512)
    export.add_argument("--subtype", default="PCM_16")
    export.set_defaults(func=_with_defaults(export, "export"))

    demo = sub.add_parser("demo", help="一键生成并跑通完整流程")
    demo.add_argument("workspace")
    demo.add_argument("--rate", default=48000)
    demo.set_defaults(func=cmd_demo)
    return parser


def _with_defaults(parser: argparse.ArgumentParser, name: str):
    def runner(args: argparse.Namespace) -> int:
        workspace = Path(args.workspace)
        defaults = {
            "build": {
                "script": workspace / "script.json",
                "mix": workspace / "mix.json",
                "state": workspace / "project_state.json",
                "report": workspace / "report.json",
            },
            "preview": {"state": workspace / "project_state.json"},
            "export": {
                "state": workspace / "project_state.json",
                "out": workspace / "master.wav",
                "export_state": workspace / "export_state.json",
                "export_report": workspace / "export_report.json",
            },
        }[name]
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, str(value))
        handler = {
            "build": cmd_build,
            "preview": cmd_preview,
            "export": cmd_export,
        }[name]
        return handler(args)

    return runner


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

