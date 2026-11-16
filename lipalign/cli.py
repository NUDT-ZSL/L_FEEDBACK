"""Command-line interface for lipalign."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .aligner import AlignOptions, align
from .cache import cache_summary
from .errors import LipAlignError, error_payload
from .stretch import StretchSettings
from .synth import DEFAULT_LANGUAGES, DEFAULT_SAMPLE_RATE, generate_samples
from .vad import SplitSettings


class CliParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise LipAlignError("E_CLI", message, exit_code=2)


def _parse_languages(value: str) -> List[str]:
    languages = [item.strip() for item in value.split(",") if item.strip()]
    if not languages:
        raise LipAlignError("E_CLI", "--languages cannot be empty", exit_code=2)
    return languages


def _parse_audio_map(value: str) -> Tuple[str, str]:
    language, separator, path = value.partition("=")
    if not separator or not language.strip() or not path:
        raise LipAlignError(
            "E_CLI",
            "--audio must use LANG=PATH syntax: {}".format(value),
            exit_code=2,
        )
    return language.strip(), path


def _collect_audio_inputs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in args.audio or []:
        pairs.append(_parse_audio_map(item))
    if args.audio_dir:
        if pairs:
            raise LipAlignError(
                "E_CLI", "Use either --audio or --audio-dir, not both", exit_code=2
            )
        audio_dir = Path(args.audio_dir)
        if not audio_dir.is_dir():
            raise LipAlignError(
                "E_INPUT", "Audio directory does not exist: {}".format(audio_dir), exit_code=2
            )
        for path in sorted(audio_dir.glob("*.wav")):
            if path.name.lower() == "background.wav":
                continue
            pairs.append((path.stem, str(path)))
    if not pairs:
        raise LipAlignError(
            "E_CLI",
            "Provide at least one --audio LANG=PATH or --audio-dir",
            exit_code=2,
        )
    seen = set()
    for language, _ in pairs:
        if language in seen:
            raise LipAlignError(
                "E_CLI", "Duplicate language: {}".format(language), exit_code=2
            )
        seen.add(language)
    return pairs


def _build_parser() -> CliParser:
    parser = CliParser(
        prog="lipalign",
        description=(
            "Offline deterministic lip-sync dubbing alignment and multitrack WAV packaging"
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=CliParser)

    generate = subparsers.add_parser(
        "generate", help="Generate deterministic timeline, language WAVs, and pink noise"
    )
    generate.add_argument("--out", required=True, help="Output directory")
    generate.add_argument(
        "--seed",
        type=int,
        default=20251117,
        help="Deterministic synthesis seed (default: 20251117)",
    )
    generate.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated built-in languages (default: en,zh,es)",
    )
    generate.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate for generated WAVs (default: 48000)",
    )

    align_parser = subparsers.add_parser(
        "align", help="Split, stretch, cache, assemble, and package language dubs"
    )
    align_parser.add_argument("--timeline", required=True, help="Lip timeline JSON")
    align_parser.add_argument(
        "--audio",
        action="append",
        metavar="LANG=PATH",
        help="Language WAV; repeat for multiple languages",
    )
    align_parser.add_argument(
        "--audio-dir",
        help="Directory of language WAVs, using the filename stem as language",
    )
    align_parser.add_argument(
        "--background",
        help="Optional mono background WAV; default is deterministic built-in pink noise",
    )
    align_parser.add_argument("--output", required=True, help="Multitrack WAV output path")
    align_parser.add_argument(
        "--report",
        help="Alignment report JSON path (default: OUTPUT.alignment.json)",
    )
    align_parser.add_argument(
        "--cache-dir",
        default=".lipalign-cache",
        help="Incremental fragment cache directory (default: .lipalign-cache)",
    )
    align_parser.add_argument(
        "--algorithm",
        choices=("wsola", "phase_vocoder"),
        default="wsola",
        help="Time-stretch algorithm (default: wsola)",
    )
    align_parser.add_argument(
        "--silence-rms",
        type=float,
        default=0.006,
        help="RMS silence threshold in normalized amplitude (default: 0.006)",
    )
    align_parser.add_argument(
        "--silence-zcr",
        type=float,
        default=0.080,
        help="Zero-crossing-rate silence threshold (default: 0.080)",
    )
    align_parser.add_argument(
        "--silence-frame-ms", type=float, default=20.0, help="VAD frame length"
    )
    align_parser.add_argument(
        "--silence-hop-ms", type=float, default=5.0, help="VAD hop length"
    )
    align_parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=80.0,
        help="Merge activity runs separated by less than this silence",
    )
    align_parser.add_argument(
        "--min-active-ms",
        type=float,
        default=60.0,
        help="Discard activity runs shorter than this duration",
    )
    align_parser.add_argument(
        "--boundary-guard-ms",
        type=float,
        default=2.0,
        help="Boundary guard around detected speech",
    )
    align_parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=10.0,
        help="Fragment edge crossfade duration (default: 10 ms)",
    )
    align_parser.add_argument(
        "--wsola-frame-ms", type=float, default=40.0, help="WSOLA frame length"
    )
    align_parser.add_argument(
        "--wsola-overlap-ms", type=float, default=20.0, help="WSOLA overlap length"
    )
    align_parser.add_argument(
        "--wsola-search-ms", type=float, default=10.0, help="WSOLA similarity search radius"
    )
    align_parser.add_argument(
        "--wsola-search-step", type=int, default=2, help="WSOLA search stride in samples"
    )
    align_parser.add_argument(
        "--pv-frame-size", type=int, default=2048, help="Phase-vocoder FFT frame size"
    )
    align_parser.add_argument(
        "--pv-overlap", type=int, default=4, help="Phase-vocoder overlap factor"
    )

    report = subparsers.add_parser(
        "report", help="Show an alignment report and cache summary"
    )
    report.add_argument("--report", help="Alignment report JSON path")
    report.add_argument("--cache-dir", help="Fragment cache directory")
    report.add_argument(
        "--json", action="store_true", help="Emit a machine-readable summary to stdout"
    )
    return parser


def _run_generate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result = generate_samples(
        out_dir=args.out,
        seed=args.seed,
        languages=_parse_languages(args.languages),
        sample_rate=args.sample_rate,
    )
    elapsed = time.perf_counter() - started
    print(
        'generated sentences={} languages={} seed={} duration_sec={:.3f} '
        'elapsed_sec={:.3f} output="{}"'.format(
            result["sentence_count"],
            len(result["languages"]),
            result["seed"],
            result["duration_sec"],
            elapsed,
            result["output_dir"],
        )
    )
    return 0


def _run_align(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    report_path = (
        Path(args.report)
        if args.report
        else output_path.with_name(output_path.stem + ".alignment.json")
    )
    split_settings = SplitSettings(
        silence_rms=args.silence_rms,
        silence_zcr=args.silence_zcr,
        frame_ms=args.silence_frame_ms,
        hop_ms=args.silence_hop_ms,
        min_silence_ms=args.min_silence_ms,
        min_active_ms=args.min_active_ms,
        boundary_guard_ms=args.boundary_guard_ms,
    )
    stretch_settings = StretchSettings(
        algorithm=args.algorithm,
        wsola_frame_ms=args.wsola_frame_ms,
        wsola_overlap_ms=args.wsola_overlap_ms,
        wsola_search_ms=args.wsola_search_ms,
        wsola_search_step=args.wsola_search_step,
        pv_frame_size=args.pv_frame_size,
        pv_overlap=args.pv_overlap,
    )
    result = align(
        AlignOptions(
            timeline_path=args.timeline,
            audio_inputs=_collect_audio_inputs(args),
            output_path=output_path,
            report_path=report_path,
            cache_dir=args.cache_dir,
            algorithm=args.algorithm,
            split_settings=split_settings,
            stretch_settings=stretch_settings,
            background_path=args.background,
            crossfade_ms=args.crossfade_ms,
        )
    )
    totals = result.stats
    print(
        "aligned sentences={} failed={} cache_hits={} cache_recomputes={} "
        'elapsed_sec={:.3f} output="{}" report="{}"'.format(
            totals["sentences_processed"],
            totals["sentences_failed"],
            totals["cache_hits"],
            totals["cache_recomputes"],
            totals["elapsed_sec"],
            output_path,
            report_path,
        )
    )
    if result.partial:
        raise LipAlignError(
            "E_PARTIAL_FAILURE",
            "One or more sentences failed; successful output and cache were retained",
            exit_code=6,
            details={
                "sentences_failed": totals["sentences_failed"],
                "report": str(report_path),
                "output": str(output_path),
            },
        )
    return 0


def _load_report(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LipAlignError(
            "E_INPUT", "Cannot read report: {}".format(path), exit_code=2
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LipAlignError(
            "E_JSON_INVALID",
            "Invalid report JSON at line {} column {}: {}".format(
                exc.lineno, exc.colno, exc.msg
            ),
            exit_code=2,
        ) from exc
    if not isinstance(payload, dict):
        raise LipAlignError(
            "E_REPORT_INVALID", "Report root must be a JSON object", exit_code=2
        )
    return payload


def _run_report(args: argparse.Namespace) -> int:
    if not args.report and not args.cache_dir:
        raise LipAlignError(
            "E_CLI", "report requires --report and/or --cache-dir", exit_code=2
        )
    report_payload: Optional[Dict[str, Any]] = None
    if args.report:
        report_payload = _load_report(Path(args.report))

    cache_dir = args.cache_dir
    if cache_dir is None and report_payload is not None:
        settings = report_payload.get("settings")
        if isinstance(settings, dict):
            cache_dir = settings.get("cache_dir")
    cache_payload = cache_summary(cache_dir) if cache_dir else None

    if args.json:
        summary: Dict[str, Any] = {"cache": cache_payload}
        if report_payload is not None:
            totals = report_payload.get("totals") or {}
            timeline = report_payload.get("timeline") or {}
            output = report_payload.get("output") or {}
            languages = report_payload.get("languages") or []
            summary["report"] = {
                "path": str(args.report),
                "status": report_payload.get("status"),
                "sentence_count": timeline.get("sentence_count"),
                "duration_sec": timeline.get("duration_sec"),
                "languages": [
                    {
                        "language": item.get("language"),
                        "cache_hits": item.get("cache_hits"),
                        "cache_recomputes": item.get("cache_recomputes"),
                    }
                    for item in languages
                    if isinstance(item, dict)
                ],
                "totals": totals,
                "output": output,
            }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    if report_payload is not None:
        totals = report_payload.get("totals") or {}
        timeline = report_payload.get("timeline") or {}
        output = report_payload.get("output") or {}
        languages = report_payload.get("languages") or []
        print(
            'report="{}" status={} sentences={}/{} failed={} cache_hits={} '
            "cache_recomputes={} duration_sec={} channels={}".format(
                args.report,
                report_payload.get("status"),
                totals.get("sentences_processed"),
                totals.get("sentences_considered"),
                totals.get("sentences_failed"),
                totals.get("cache_hits"),
                totals.get("cache_recomputes"),
                timeline.get("duration_sec"),
                output.get("channel_count"),
            )
        )
        for item in languages:
            if isinstance(item, dict):
                print(
                    "language={} cache_hits={} cache_recomputes={}".format(
                        item.get("language"),
                        item.get("cache_hits"),
                        item.get("cache_recomputes"),
                    )
                )
    if cache_payload is not None:
        print(
            'cache_dir="{}" entries={} invalid_sidecars={} size_bytes={} '
            "algorithms={} sample_rates={}".format(
                cache_payload["cache_dir"],
                cache_payload["entry_count"],
                cache_payload["invalid_sidecar_count"],
                cache_payload["total_bytes"],
                json.dumps(cache_payload["algorithms"], sort_keys=True),
                json.dumps(cache_payload["sample_rates"], sort_keys=True),
            )
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        parser = _build_parser()
        args = parser.parse_args(argv)
        if args.command == "generate":
            return _run_generate(args)
        if args.command == "align":
            return _run_align(args)
        if args.command == "report":
            return _run_report(args)
        raise LipAlignError("E_CLI", "Unknown command", exit_code=2)
    except LipAlignError as exc:
        print(
            json.dumps(
                error_payload(exc),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return int(exc.exit_code)
    except KeyboardInterrupt:
        payload = {
            "ok": False,
            "code": "E_INTERRUPTED",
            "message": "Interrupted by user",
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True), file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        payload = {
            "ok": False,
            "code": "E_INTERNAL",
            "message": "{}: {}".format(type(exc).__name__, exc),
        }
        print(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            file=sys.stderr,
        )
        return 1

