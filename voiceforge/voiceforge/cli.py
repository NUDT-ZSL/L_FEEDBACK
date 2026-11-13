"""Command line entry point for VoiceForge."""

import argparse
import json
import os
import sys

from . import __version__, OUTPUT_SAMPLE_RATE
from .engine import load_manifest, synthesize
from .generator import DEFAULT_SEED, MANIFEST_NAME, generate_dataset
from .interrupt import InterruptController
from .lock import LockUnavailableError, ProcessLock
from .outputs import write_outputs


def cmd_generate(args) -> int:
    manifest = generate_dataset(args.input_dir, seed=args.seed)
    shard_count = sum(len(player["segments"])
                      for player in manifest["players"])
    print(f"已生成 {len(manifest['players'])} 个玩家、{shard_count} 个分片；"
          f"清单：{os.path.join(args.input_dir, MANIFEST_NAME)}")
    return 0


def cmd_synthesize(args) -> int:
    manifest_path = os.path.join(args.input_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        print(f"错误：找不到事件清单 {manifest_path}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        with ProcessLock(args.output_dir):
            return _run_synthesis(args, manifest_path)
    except LockUnavailableError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 3


def _run_synthesis(args, manifest_path: str) -> int:
    manifest, manifest_text = load_manifest(manifest_path)
    controller = InterruptController()
    controller.install()
    try:
        result = synthesize(
            manifest, manifest_text, args.input_dir, args.output_dir,
            interrupt_controller=controller,
            output_rate=args.sample_rate)
        if result["interrupted"]:
            print("收到中断：已在安全点写出检查点，退出码 0。"
                  "重新运行将从检查点恢复。")
            return 0
        summary = write_outputs(result, manifest, args.input_dir,
                                args.output_dir, args.sample_rate)
        print("合成完成。")
        print(json.dumps({
            "total_samples": summary["total_samples"],
            "event_count": summary["event_count"],
            "clipped_samples": summary["clipped_samples"],
            "missing_source_player_count":
                summary["missing_source_player_count"],
            "missing_source_interval_count":
                summary["missing_source_interval_count"],
        }, indent=2, ensure_ascii=False))
        return 0
    finally:
        controller.restore()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceforge",
        description="离线多玩家语音事件对齐、混音与可恢复合成工具（纯标准库）")
    parser.add_argument("--version", action="version",
                        version=f"voiceforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性样本与事件清单")
    gen.add_argument("input_dir", help="样本输出目录")
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help=f"固定随机种子（默认 {DEFAULT_SEED}）")
    gen.set_defaults(func=cmd_generate)

    syn = sub.add_parser("synthesize", help="从检查点恢复或从头合成统一音轨")
    syn.add_argument("input_dir", help="包含 events.json 与 WAV 分片的目录")
    syn.add_argument("output_dir", help="检查点与最终产物输出目录")
    syn.add_argument("--sample-rate", type=int, default=OUTPUT_SAMPLE_RATE,
                     help=f"统一输出采样率（默认 {OUTPUT_SAMPLE_RATE}）")
    syn.set_defaults(func=cmd_synthesize)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

