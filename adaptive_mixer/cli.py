"""Command-line entry point: ``python -m adaptive_mixer ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import BatchJob, process_batch
from .config import load_config
from .engine import MixEngine
from .errors import MixerError
from .sample_gen import generate_tracks


def _cmd_render(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    engine = MixEngine(config, base_dir=config_path.parent,
                       output_path=args.output, state_file=args.state)
    result = engine.render()
    report_path = Path(args.report) if args.report \
        else Path(args.output).with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"ok": True, "output": result.output_path,
                      "report": str(report_path),
                      "resumed": result.resumed}, ensure_ascii=False))
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    jobs = [BatchJob(job_id=item["job_id"],
                     config_path=item["config_path"],
                     output_path=item["output_path"],
                     state_file=item.get("state_file"))
            for item in spec["jobs"]]
    summary = process_batch(jobs, max_workers=args.workers)
    Path(args.report).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"ok": summary["failed"] == 0,
                      "succeeded": summary["succeeded"],
                      "failed": summary["failed"]}, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 2


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_tracks(args.out_dir, args.tracks.split(","),
                            args.seconds)
    print(json.dumps(paths, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive_mixer",
        description="离线自适应游戏配乐混音引擎（纯 Python/NumPy）")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="渲染单个场景配置")
    render.add_argument("config", help="场景配置 JSON 路径")
    render.add_argument("output", help="输出 WAV 路径")
    render.add_argument("--report", help="JSON 报告输出路径")
    render.add_argument("--state", help="断点状态文件路径")
    render.set_defaults(func=_cmd_render)

    batch = sub.add_parser("batch", help="并发处理多个场景配置")
    batch.add_argument("jobs", help="批处理任务 JSON 清单")
    batch.add_argument("report", help="汇总报告输出路径")
    batch.add_argument("--workers", type=int, default=4)
    batch.set_defaults(func=_cmd_batch)

    gen = sub.add_parser("generate", help="生成确定性测试分轨")
    gen.add_argument("out_dir")
    gen.add_argument("--tracks",
                     default="strings,percussion,ambience")
    gen.add_argument("--seconds", type=float, default=12.0)
    gen.set_defaults(func=_cmd_generate)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MixerError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
