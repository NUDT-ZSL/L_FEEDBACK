"""Command line entry point for the squad voice replay mixer."""

import argparse
import json
import os
import signal
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_mixer.errors import (  # noqa: E402
    EXIT_INTERRUPTED,
    EXIT_IO,
    EXIT_OK,
    EXIT_VALIDATION,
    ProjectValidationError,
)
from voice_mixer.exporter import export_project  # noqa: E402
from voice_mixer.parser import load_project  # noqa: E402
from voice_mixer.sample_project import (  # noqa: E402
    build_clipping_project,
    build_mono_project,
    build_sample_project,
)
from voice_mixer.timeline import TimelineError, build_timeline  # noqa: E402


def _write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _failure_manifest(path, errors):
    payload = {"status": "failed", "errors": errors}
    try:
        _write_json(path, payload)
    except OSError:
        pass


def run_export(project_path, simulate_interrupt_after=None):
    project = load_project(project_path)
    try:
        timeline = build_timeline(project)
    except TimelineError as exc:
        _failure_manifest(project.export.manifest_path, exc.errors)
        print(json.dumps({"status": "failed", "errors": exc.errors}, indent=2))
        return EXIT_VALIDATION

    interrupt_state = {"requested": False, "chunks_remaining": simulate_interrupt_after}

    def _handle_signal(signum, frame):
        interrupt_state["requested"] = True
        print(f"\n[signal] received {signum}; finishing current chunk before shutdown")

    def _interrupt_flag():
        if interrupt_state["requested"]:
            return True
        if interrupt_state["chunks_remaining"] is not None:
            if interrupt_state["chunks_remaining"] <= 0:
                return True
            interrupt_state["chunks_remaining"] -= 1
        return False

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, _handle_signal)
    try:
        code, manifest = export_project(project, timeline, interrupt_flag=_interrupt_flag)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if code == EXIT_INTERRUPTED:
        print("[exit] clean interrupt: snapshot and chunks persisted; rerun to resume")
    return code


def run_demo(base_dir):
    paths = {
        "sample": os.path.join(base_dir, "sample_project.json"),
        "clipping": os.path.join(base_dir, "clipping_project.json"),
        "mono": os.path.join(base_dir, "mono_project.json"),
    }
    _write_json(paths["sample"], build_sample_project(
        output_wav=os.path.join(base_dir, "replay.wav"),
        manifest=os.path.join(base_dir, "manifest.json"),
        work_dir=os.path.join(base_dir, "render_work"),
    ))
    _write_json(paths["clipping"], build_clipping_project(
        output_wav=os.path.join(base_dir, "clipping.wav"),
        manifest=os.path.join(base_dir, "clipping_manifest.json"),
        work_dir=os.path.join(base_dir, "clipping_work"),
    ))
    _write_json(paths["mono"], build_mono_project(
        output_wav=os.path.join(base_dir, "mono.wav"),
        manifest=os.path.join(base_dir, "mono_manifest.json"),
        work_dir=os.path.join(base_dir, "mono_work"),
    ))
    print(f"[demo] generated projects in {base_dir}")
    for name, path in paths.items():
        print(f"[demo] exporting {name}")
        code = run_export(path)
        if code not in (EXIT_OK, EXIT_INTERRUPTED):
            return code
    return EXIT_OK


def run_selftest():
    loader = unittest.TestLoader()
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_mixer", "test")
    suite = loader.discover(
        start_dir=test_dir,
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return EXIT_OK if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Squad voice communication replay mixer")
    parser.add_argument("project", nargs="?", help="path to the JSON project file")
    parser.add_argument("--demo", metavar="DIR", help="generate deterministic sample projects and export them")
    parser.add_argument("--selftest", action="store_true", help="run the standard-library unittest suite")
    parser.add_argument(
        "--simulate-interrupt-after",
        type=int,
        default=None,
        help="simulate a clean interrupt after N chunks (for resume verification)",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()
    if args.demo:
        return run_demo(args.demo)
    if not args.project:
        parser.error("a project path is required unless --demo or --selftest is used")
    try:
        return run_export(args.project, args.simulate_interrupt_after)
    except ProjectValidationError as exc:
        errors = getattr(exc, "errors", None) or [exc.to_obj()]
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return EXIT_VALIDATION
    except OSError as exc:
        print(json.dumps({"status": "failed", "errors": [{"code": "E_IO", "message": str(exc)}]}, indent=2))
        return EXIT_IO


if __name__ == "__main__":
    sys.exit(main())
