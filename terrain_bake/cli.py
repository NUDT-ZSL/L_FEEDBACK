"""Command-line entry point with business-code exits and stage summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .errors import (
    E_INTERNAL_ERROR,
    E_INVALID_ARGUMENT,
    E_INVALID_HEIGHTFIELD,
    E_WEIGHT_MISMATCH,
    ERROR_EXIT_CODES,
    BakeError,
)
from .pipeline import BakeConfig, bake


class _UsageError(Exception):
    pass


class _BusinessArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _BusinessArgumentParser(
        prog="terrain-bake",
        description=(
            "Deterministic offline terrain LOD, normal-map and preview baker. "
            "All default samples are generated from the supplied seed."
        ),
    )
    parser.add_argument("--size", type=int, default=65, help="square heightfield size")
    parser.add_argument("--seed", type=int, default=20261116, help="uint64 seed")
    parser.add_argument("--lods", type=int, default=3, help="LOD count (>=3)")
    parser.add_argument(
        "--normal-resolution",
        type=int,
        default=512,
        help="output normal map resolution",
    )
    parser.add_argument("--preview-width", type=int, default=800)
    parser.add_argument("--preview-height", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/terrain"),
        help="output directory replaced atomically",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.03,
        help="maximum relative simplification error",
    )
    parser.add_argument(
        "--weight-tolerance",
        type=float,
        default=1e-6,
        help="maximum material weight sum residual",
    )
    parser.add_argument(
        "--preview-backend",
        choices=("auto", "cpu", "opengl"),
        default="auto",
        help="auto tries OpenGL 3.3, then falls back to CPU",
    )
    parser.add_argument(
        "--heightfield",
        type=Path,
        help="optional .npy heightfield for validation/debug runs",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="optional .npy grass/rock/snow weights for validation/debug runs",
    )
    return parser


def _load_npy(path: Path, code: str, label: str) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise BakeError(
            code,
            f"could not load {label} from {path}",
            {"reason": str(exc)},
        ) from exc
    if not isinstance(array, np.ndarray):
        raise BakeError(code, f"{label} file must contain one NumPy array")
    return array


def _error_and_exit(error: BakeError) -> int:
    print(f"[error] {error.code}: {error.message}", file=sys.stderr)
    return ERROR_EXIT_CODES.get(error.code, 8)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"[error] {E_INVALID_ARGUMENT}: {exc}", file=sys.stderr)
        return ERROR_EXIT_CODES[E_INVALID_ARGUMENT]

    config = BakeConfig(
        size=args.size,
        seed=args.seed,
        lod_count=args.lods,
        normal_resolution=args.normal_resolution,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        output_dir=args.output,
        max_relative_error=args.max_error,
        weight_tolerance=args.weight_tolerance,
        preview_backend=args.preview_backend,
    )

    heightfield = None
    weights = None
    try:
        if args.heightfield is not None:
            heightfield = _load_npy(
                args.heightfield,
                E_INVALID_HEIGHTFIELD,
                "heightfield",
            )
        if args.weights is not None:
            weights = _load_npy(
                args.weights,
                E_WEIGHT_MISMATCH,
                "material weights",
            )
    except BakeError as error:
        return _error_and_exit(error)

    try:
        def report(line: str) -> None:
            stream = sys.stderr if line.startswith("[error]") else sys.stdout
            print(line, file=stream)

        result = bake(
            config,
            heightfield=heightfield,
            weights=weights,
            reporter=report,
        )
    except BakeError as error:
        return _error_and_exit(error)
    except Exception as exc:
        print(
            f"[error] {E_INTERNAL_ERROR}: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return ERROR_EXIT_CODES[E_INTERNAL_ERROR]

    if result.exit_code != 0:
        print(
            f"[manifest] status=error error={result.manifest.get('error')} "
            f"path={result.manifest_path}"
        )
    else:
        checksums = result.manifest.get("checksums", {})
        print(
            f"[manifest] status=ok artifacts={len(result.manifest['artifacts'])} "
            f"checksums={len(checksums)} path={result.manifest_path}"
        )
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

