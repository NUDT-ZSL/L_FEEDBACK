"""命令行入口：单材质渲染与批量并行渲染。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .errors import ERROR_CODES, MaterialError, emit_error
from .pipeline import process_material


def _worker(args: tuple[str, str, int, int]) -> dict:
    input_path, output_root, base_seed, position = args
    # SeedSequence.spawn 保证批量每项子种子稳定，与调度顺序/并发数无关
    import numpy as np

    child_seed = int(
        np.random.SeedSequence(base_seed).spawn(position + 1)[position].generate_state(1)[0]
    )
    try:
        result = process_material(input_path, output_root, child_seed)
        return {
            "ok": True,
            "input": input_path,
            "position": position,
            "status": "OK",
            "error_code": None,
            "exit_code": 0,
            "seed": child_seed,
            "output_dir": result.output_dir,
            "files": result.files,
        }
    except MaterialError as exc:
        payload = exc.to_dict()
        return {
            "ok": False,
            "input": input_path,
            "position": position,
            "status": exc.code,
            "error_code": exc.code,
            "exit_code": payload["exit_code"],
            "seed": child_seed,
            "message": exc.message,
            "details": exc.details,
        }
    except Exception as exc:  # 防御性兜底，正常业务校验不会走到这里
        return {
            "ok": False,
            "input": input_path,
            "position": position,
            "status": "E_RUNTIME",
            "error_code": "E_RUNTIME",
            "exit_code": ERROR_CODES["E_RUNTIME"],
            "seed": child_seed,
            "message": repr(exc),
            "details": {},
        }


def _write_summary(path: Path, entries: list[dict], seed: int, workers: int) -> None:
    summary = {
        "ok": all(entry["ok"] for entry in entries),
        "base_seed": int(seed),
        "workers": workers,
        "count": len(entries),
        "success_count": sum(1 for e in entries if e["ok"]),
        "failure_count": sum(1 for e in entries if not e["ok"]),
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def run_batch(inputs: list[str], output_root: str, seed: int, jobs: int) -> int:
    tasks = [(str(Path(p).resolve()), output_root, seed, i) for i, p in enumerate(inputs)]
    entries: list[dict | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_worker, task): task[3] for task in tasks}
        for future in as_completed(futures):
            position = futures[future]
            entries[position] = future.result()
    ordered = [entry for entry in entries if entry is not None]
    summary_path = Path(output_root) / "batch_summary.json"
    _write_summary(summary_path, ordered, seed, jobs)

    failed = [entry for entry in ordered if not entry["ok"]]
    if not failed:
        return 0
    if len(failed) == len(ordered):
        # 全部失败：使用与单文件路径一致的业务退出码
        return int(failed[0]["exit_code"])
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matrelight",
        description="分层材质合成、材质缓存与 GGX 方向光反射渲染",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="处理单个材质描述文件")
    render.add_argument("input", help="YAML 材质描述文件")
    render.add_argument("-o", "--output", required=True, help="输出根目录")
    render.add_argument("--seed", type=int, default=20260101, help="确定性种子")

    batch = sub.add_parser("batch", help="并行处理多个材质描述文件")
    batch.add_argument("inputs", nargs="+", help="一个或多个 YAML 材质描述文件")
    batch.add_argument("-o", "--output", required=True, help="输出根目录")
    batch.add_argument("--seed", type=int, default=20260101, help="基础确定性种子")
    batch.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="最大并行进程数",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "render":
        try:
            result = process_material(args.input, args.output, args.seed)
        except MaterialError as exc:
            emit_error(exc)
            return exc.exit_code
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "OK",
                    "input": str(Path(args.input).resolve()),
                    "output_dir": result.output_dir,
                    "seed": result.seed,
                    "files": result.files,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.jobs < 1:
        parser.error("--jobs 必须 >= 1")
    return run_batch(args.inputs, args.output, args.seed, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
