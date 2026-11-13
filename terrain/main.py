"""Command line entry point.

Exit code contract:
* success: 0;
* expected business rejection: 0 with the business code on stderr;
* unexpected exception: non-zero (propagated traceback).
"""

from __future__ import annotations

import argparse
import sys

from .config import PipelineConfig
from .errors import BUSINESS_CODES, PipelineError
from .gen_config import generate
from .pipeline import run_pipeline, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline deterministic terrain tile renderer"
    )
    parser.add_argument(
        "--config", required=True, help="path to the pipeline JSON config"
    )
    parser.add_argument(
        "--generate-configs",
        action="store_true",
        help="regenerate sample configs and exit",
    )
    args = parser.parse_args(argv)

    if args.generate_configs:
        for path in generate("configs"):
            print(path)
        return 0

    try:
        config = PipelineConfig.load(args.config)
        result = run_pipeline(config)
        write_outputs(result, config)
    except PipelineError as exc:
        if exc.code not in BUSINESS_CODES:
            raise
        print(exc.code, file=sys.stderr)
        if exc.code == "E_NO_GL":
            print(f"[fallback] {exc.message}", file=sys.stderr)
        return 0

    print(f"backend={result.backend_used} png={config.output_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
