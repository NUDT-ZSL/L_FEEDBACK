"""命令行入口：``python -m engine_pipeline.cli [--out DIR] ...``。

业务失败时进程返回对应的整数错误码（1001~1007），成功返回 0。
"""

import argparse
import json
import sys

from .errors import BakeError
from .pipeline import BakeConfig, RpmTable, run_bake


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把单段确定性合成引擎录音烘焙为可无缝循环的 WAV 片段集"
    )
    parser.add_argument("--out", default="out", help="输出目录")
    parser.add_argument("--seed", type=int, default=BakeConfig.seed)
    parser.add_argument("--duration", type=float, default=BakeConfig.duration_s)
    parser.add_argument("--rpm-start", type=float, default=1000.0, help="转速表起始转速")
    parser.add_argument("--rpm-end", type=float, default=2200.0, help="转速表终止转速")
    parser.add_argument("--rpm-step", type=float, default=400.0, help="转速步进")
    parser.add_argument("--peak-ceiling", type=float, default=BakeConfig.peak_ceiling)
    parser.add_argument("--crossfade-ratio", type=float,
                        default=BakeConfig.crossfade_ratio)
    parser.add_argument("--loudness-tol-db", type=float,
                        default=BakeConfig.loudness_tol_db)
    parser.add_argument("--phase-tol-rad", type=float,
                        default=BakeConfig.phase_error_tol_rad)
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = BakeConfig(
        output_dir=args.out,
        seed=args.seed,
        duration_s=args.duration,
        rpm_table=RpmTable(args.rpm_start, args.rpm_end, args.rpm_step),
        peak_ceiling=args.peak_ceiling,
        crossfade_ratio=args.crossfade_ratio,
        loudness_tol_db=args.loudness_tol_db,
        phase_error_tol_rad=args.phase_tol_rad,
    )
    try:
        manifest = run_bake(config)
    except BakeError as exc:
        print(f"[ERROR {int(exc.code)}] {exc}", file=sys.stderr)
        return exc.exit_code

    print(f"烘焙完成：{manifest['segment_count']} 个片段 -> {args.out}")
    print(json.dumps({
        "sample_rate": manifest["sample_rate"],
        "segment_count": manifest["segment_count"],
        "uniform_gain": manifest["uniform_gain"],
        "manifest": args.out + "/manifest.json",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
