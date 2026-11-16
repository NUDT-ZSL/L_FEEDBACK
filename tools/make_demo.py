from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.sample_factory import create_scenario  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic offline broadcast test media"
    )
    parser.add_argument("output_dir", help="directory for WAV, MKV and manifest files")
    parser.add_argument(
        "--reference-delay-ms",
        type=float,
        default=30.0,
        help="delay applied to the reference track in the video container",
    )
    parser.add_argument(
        "--no-reference-audio",
        action="store_true",
        help="create a video container with zero audio streams",
    )
    parser.add_argument(
        "--same-duration-silent-track",
        action="store_true",
        help="make the silent Japanese track the same duration (disables fallback fixture)",
    )
    args = parser.parse_args()
    scenario = create_scenario(
        Path(args.output_dir),
        reference_delay_ms=args.reference_delay_ms,
        include_reference=not args.no_reference_audio,
        short_silent_track_seconds=None if args.same_duration_silent_track else 4.0,
    )
    print(
        json.dumps(
            {
                "manifest": str(scenario.manifest_path),
                "video": str(scenario.video_path),
                "languages": {
                    key: str(value) for key, value in scenario.language_paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
