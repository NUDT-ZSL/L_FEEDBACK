#!/usr/bin/env python3
"""Local acceptance tests for material-verify.

This script uses only Python's standard library. It is not a runtime
dependency of the verifier.
"""

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    if len(sys.argv) < 3:
        print("usage: acceptance.py <material-verify> <work-dir>", file=sys.stderr)
        return 2

    binary = pathlib.Path(sys.argv[1]).resolve()
    work = pathlib.Path(sys.argv[2]).resolve()
    root = pathlib.Path(__file__).resolve().parent.parent
    samples = root / "tests" / "samples"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    def invoke(name, material, extra=None, output_name=None):
        output = work / (output_name or (name + ".json"))
        command = [
            str(binary),
            "--material",
            str(samples / material),
            "--output",
            str(output),
            "--deterministic",
        ]
        if extra:
            command.extend(extra)
        completed = subprocess.run(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(output.exists(), "missing report for {}: {}".format(name, completed.stderr))
        with output.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        return completed, report, output

    print("[1/8] multi-attachment render and repeated-run determinism")
    first, report1, path1 = invoke(
        "multi-a", "multi_attach/material.json", output_name="multi-a.json"
    )
    second, report2, path2 = invoke(
        "multi-b", "multi_attach/material.json", output_name="multi-b.json"
    )
    require(first.returncode == 0, first.stderr)
    require(second.returncode == 0, second.stderr)
    require(report1["summary"]["validation_passed_pixels"] == 3072, "pixel pass count")
    require(report1["summary"]["validation_failed_pixels"] == 0, "pixel fail count")
    require(report1["summary"]["attachment_count"] == 6, "attachment count")
    require(report1["report_fingerprint"] == report2["report_fingerprint"], "fingerprint")
    require(path1.read_bytes() == path2.read_bytes(), "deterministic reports differ")

    print("[2/8] cache miss/hit equivalence")
    cache = work / "cache"
    cache.mkdir()
    miss_run, miss, _ = invoke(
        "cache-miss",
        "multi_attach/material.json",
        ["--cache-dir", str(cache)],
        "cache-miss.json",
    )
    hit_run, hit, _ = invoke(
        "cache-hit",
        "multi_attach/material.json",
        ["--cache-dir", str(cache)],
        "cache-hit.json",
    )
    require(miss_run.returncode == 0 and hit_run.returncode == 0, "cache run failed")
    require(not miss["variants"][0]["cache_hit"], "first run unexpectedly hit cache")
    require(hit["variants"][0]["cache_hit"], "second run did not hit cache")
    require(
        miss["report_fingerprint"] == hit["report_fingerprint"],
        "cache changed stable fingerprint",
    )
    require(
        [item["sha256"] for item in miss["variants"][0]["attachments"]]
        == [item["sha256"] for item in hit["variants"][0]["attachments"]],
        "cache changed attachment pixels",
    )
    require(
        list(cache.glob("program-cache-v1/*/*.program")),
        "program binary was not written to cache",
    )

    print("[3/8] material variants and macro-specific cache keys")
    variant_run, variants, _ = invoke(
        "variants",
        "multi_attach/material.json",
        ["--variants", "default,tinted"],
        "variants.json",
    )
    require(variant_run.returncode == 0, variant_run.stderr)
    require(len(variants["variants"]) == 2, "expected two variants")
    require(
        variants["variants"][0]["cache_key"] != variants["variants"][1]["cache_key"],
        "macro variants shared a cache key",
    )
    require(variants["variants"][1]["macros"] == ["TINTED_PIPELINE"], "variant macro")

    print("[4/8] procedural texture and constant-buffer overrides")
    override_run, override, _ = invoke(
        "override",
        "multi_attach/material.json",
        ["--texture-size", "13x7", "--const", "tint=0.1,0.2,0.3,0.4"],
        "override.json",
    )
    require(override_run.returncode == 0, override_run.stderr)
    textures = override["material"]["dependencies"]["textures"]
    require(all(t["width"] == 13 and t["height"] == 7 for t in textures), "texture size")
    tint = override["material"]["dependencies"]["constants"]["values"]["tint"]
    require(abs(tint[0] - 0.1) < 1e-6 and abs(tint[3] - 0.4) < 1e-6, "constant override")
    require(
        override["variants"][0]["cache_key"] != report1["variants"][0]["cache_key"],
        "dependency overrides did not change the program cache key",
    )

    print("[5/8] overlapping geometry depth/stencil verification")
    depth_run, depth, _ = invoke("depth", "depth_stencil/material.json")
    require(depth_run.returncode == 0, depth_run.stderr)
    rules = {rule["id"]: rule for rule in depth["variants"][0]["rules"]}
    require(rules["stencil-final"]["status"] == "pass", "stencil validation")
    require(rules["depth-nearest-instance"]["status"] == "pass", "depth validation")
    require(depth["summary"]["attachment_count"] == 4, "depth readbacks")

    print("[6/8] classified compile/link/FBO/attachment failures")
    expected_errors = [
        ("bad_compile", 3, "SHADER_COMPILE_FAILED", "line"),
        ("bad_link", 4, "SHADER_LINK_FAILED", "log"),
        ("bad_fbo", 5, "FRAMEBUFFER_INCOMPLETE", "detail"),
        ("bad_attachment", 6, "ATTACHMENT_OUT_OF_BOUNDS", "message"),
    ]
    for sample, exit_code, code, field in expected_errors:
        run, report, _ = invoke(sample, sample + "/material.json")
        require(run.returncode == exit_code, "{} exit {}".format(sample, run.returncode))
        require(report["errors"], "{} produced no error object".format(sample))
        require(report["errors"][0]["code"] == code, "{} error code".format(sample))
        require(field in report["errors"][0], "{} missing {}".format(sample, field))

    compile_lines = set()
    for _ in range(2):
        _, report, _ = invoke(
            "bad-compile-repeat",
            "bad_compile/material.json",
            output_name="bad-compile-repeat.json",
        )
        compile_lines.add(json.dumps(report, sort_keys=True))
    require(len(compile_lines) == 1, "error report is not deterministic")

    print("[7/8] per-variant recovery after shader failure")
    recovery_run, recovery, _ = invoke(
        "recovery",
        "multi_attach/material.json",
        ["--variants", "broken_compile,default"],
        "recovery.json",
    )
    require(recovery_run.returncode == 3, "recovery exit code")
    require(len(recovery["variants"]) == 1, "successful variant was not preserved")
    require(recovery["variants"][0]["status"] == "pass", "recovered variant failed")
    require(recovery["summary"]["variants_failed"] == 1, "failed variant count")
    require(recovery["errors"][0]["variant"] == "broken_compile", "failure localization")

    print("[8/8] raw attachment dumps match reported hashes")
    pixels = work / "pixels"
    dump_run, dump, _ = invoke(
        "pixels",
        "depth_stencil/material.json",
        ["--dump-pixels", str(pixels)],
        "pixels.json",
    )
    require(dump_run.returncode == 0, dump_run.stderr)
    for attachment in dump["variants"][0]["attachments"]:
        path = pathlib.Path(attachment["pixel_file"])
        require(path.exists(), "missing pixel dump")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == attachment["sha256"], "pixel dump hash mismatch")

    print("PASS: all local acceptance checks succeeded")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        sys.exit(1)
