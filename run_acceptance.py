#!/usr/bin/env python3
"""Acceptance suite: full bake, kill/resume, SIGINT semantics, local
reconstruct, version switch, thread-count bitwise determinism, dry-run,
memory budget, error codes, log comparison."""
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, ".work", "acceptance")
MAIN = os.path.join(ROOT, "main.py")
PY = sys.executable

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{PASS if cond else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))


def run(args, env=None, timeout=120):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run([PY, MAIN] + args, capture_output=True, text=True,
                          env=full_env, timeout=timeout)


def launch(args, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.Popen([PY, MAIN] + args, env=full_env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def stderr_code(proc_or_completed):
    err = proc_or_completed.stderr.strip()
    try:
        return json.loads(err).get("error")
    except ValueError:
        return None


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    g = lambda *a: os.path.join(WORK, *a)

    # -- 1. deterministic sample + full bake -------------------------------
    r = run(["gen-sample", "--out", g("base.hf"), "--width", "128",
             "--height", "128", "--seed", "11"])
    check("gen-sample exit 0", r.returncode == 0, r.stderr.strip())
    base_version = json.loads(r.stdout)["version_hash"]

    r = run(["bake", "--heightfield", g("base.hf"), "--state-dir", g("st_full"),
             "--out", g("full.rgba"), "--log", g("full.log.json"),
             "--threads", "4", "--spacing", "2", "--verify-determinism"])
    check("full bake exit 0", r.returncode == 0, r.stderr.strip())
    log_full = json.load(open(g("full.log.json")))
    check("full bake completed all blocks",
          log_full["status"] == "complete"
          and len(log_full["blocks_completed"]) == log_full["blocks_total"],
          f"{len(log_full['blocks_completed'])}/{log_full['blocks_total']}")
    check("log cost is probe-count based (no clock)",
          abs(log_full["estimated_cost"]
              - log_full["probes_processed"] * 1.0e-3) < 1e-12)

    # -- 2. thread-count independence (bitwise) -----------------------------
    for threads in ("1", "8"):
        run(["bake", "--heightfield", g("base.hf"), "--state-dir",
             g(f"st_t{threads}"), "--out", g(f"t{threads}.rgba"),
             "--log", g(f"t{threads}.log.json"), "--threads", threads,
             "--spacing", "2"],
            env={"PROBE_BAKE_THREADS": threads})
    a = open(g("t1.rgba"), "rb").read()
    b = open(g("t8.rgba"), "rb").read()
    ref = open(g("full.rgba"), "rb").read()
    check("threads=1 vs threads=8 bitwise identical", a == b)
    check("threaded bake matches reference bake bitwise", a == ref)

    # -- 3. SIGKILL mid-bake -> resume from persisted blocks ----------------
    env = {"PROBE_BAKE_BLOCK_DELAY_MS": "120", "PROBE_BAKE_THREADS": "1"}
    p = launch(["bake", "--heightfield", g("base.hf"), "--state-dir",
                g("st_kill"), "--out", g("kill.rgba"), "--log",
                g("kill.log.json"), "--threads", "1", "--spacing", "2"],
               env=env)
    time.sleep(2.0)
    p.kill()  # SIGKILL: no exit-code guarantee, only persisted state survives
    p.wait()
    killed_blocks = len([f for f in os.listdir(g("st_kill"))
                         if f.endswith(".pblk")])
    check("SIGKILL leaves persisted partial blocks", 0 < killed_blocks < 64,
          f"{killed_blocks} blocks persisted before kill")
    r = run(["bake", "--heightfield", g("base.hf"), "--state-dir", g("st_kill"),
             "--out", g("kill.rgba"), "--log", g("kill.log.json"),
             "--threads", "4", "--spacing", "2"])
    log_resume = json.load(open(g("kill.log.json")))
    resumed = len(log_resume["blocks_baked_this_run"])
    check("resume skips completed blocks",
          r.returncode == 0 and resumed == 64 - killed_blocks,
          f"rebaked {resumed}, expected {64 - killed_blocks}")
    check("resumed result bitwise equals uninterrupted bake",
          open(g("kill.rgba"), "rb").read() == ref)

    # -- 4. SIGINT (Ctrl+C) -> finish current block, exit 0 -----------------
    p = launch(["bake", "--heightfield", g("base.hf"), "--state-dir",
                g("st_int"), "--out", g("int.rgba"), "--log",
                g("int.log.json"), "--threads", "2", "--spacing", "2"],
               env={"PROBE_BAKE_BLOCK_DELAY_MS": "150"})
    time.sleep(1.2)
    p.send_signal(signal.SIGINT)
    out, err = p.communicate(timeout=60)
    log_int = json.load(open(g("int.log.json")))
    check("SIGINT exits with code 0", p.returncode == 0,
          f"rc={p.returncode}")
    check("SIGINT persists partial state and logs 'interrupted'",
          log_int["status"] == "interrupted"
          and 0 < len(log_int["blocks_completed"]) < 64,
          f"{len(log_int['blocks_completed'])} blocks persisted")
    r = run(["bake", "--heightfield", g("base.hf"), "--state-dir", g("st_int"),
             "--out", g("int.rgba"), "--log", g("int2.log.json"),
             "--threads", "4", "--spacing", "2"])
    check("interrupted bake resumes to identical result",
          open(g("int.rgba"), "rb").read() == ref)

    # -- 5. local reconstruct after edit ------------------------------------
    rect = "40,48,24,16"
    r = run(["gen-sample", "--out", g("edited.hf"), "--width", "128",
             "--height", "128", "--seed", "11", "--edit-rect", rect,
             "--edit-delta", "0.4"])
    edited_version = json.loads(r.stdout)["version_hash"]
    check("edit changes version hash", edited_version != base_version)

    r = run(["reconstruct", "--heightfield", g("edited.hf"), "--state-dir",
             g("st_full"), "--out", g("recon.rgba"), "--log",
             g("recon.log.json"), "--threads", "4", "--spacing", "2",
             "--edit-rect", rect])
    check("reconstruct exit 0", r.returncode == 0, r.stderr.strip())
    log_recon = json.load(open(g("recon.log.json")))
    rebuilt = len(log_recon["blocks_rebuilt_this_run"])
    carried = len(log_recon["blocks_carried_over"])
    check("reconstruct rebuilds only affected blocks",
          0 < rebuilt < 64 and rebuilt + carried == 64,
          f"rebuilt={rebuilt} carried={carried}")

    run(["bake", "--heightfield", g("edited.hf"), "--state-dir",
         g("st_edited_full"), "--out", g("edited_full.rgba"), "--log",
         g("edited_full.log.json"), "--threads", "3", "--spacing", "2"])
    recon = np.fromfile(g("recon.rgba"), dtype="<f4")
    full = np.fromfile(g("edited_full.rgba"), dtype="<f4")
    check("reconstructed array matches full bake within 1e-5",
          np.allclose(recon, full, atol=1e-5, rtol=0.0),
          f"max abs diff={float(np.max(np.abs(recon - full))):.3e}")
    check("reconstructed array bitwise equals full bake",
          recon.tobytes() == full.tobytes())

    # -- 5b. tampered carried block -> E_RECONSTRUCT_MISMATCH ------------------
    run(["bake", "--heightfield", g("edited.hf"), "--state-dir",
         g("st_tamper"), "--out", g("tamper.rgba"), "--log",
         g("tamper.log.json"), "--threads", "2", "--spacing", "2"])
    victim = g("st_tamper", "block_00005_00005.pblk")
    blob = bytearray(open(victim, "rb").read())
    blob[-1] ^= 0x01  # flip one payload bit, header stays valid
    open(victim, "wb").write(bytes(blob))
    r = run(["reconstruct", "--heightfield", g("edited.hf"), "--state-dir",
             g("st_tamper"), "--out", g("tamper2.rgba"), "--log",
             g("tamper2.log.json"), "--threads", "2", "--spacing", "2",
             "--edit-rect", "0,0,8,8"])
    check("tampered carried block -> E_RECONSTRUCT_MISMATCH, exit 2",
          r.returncode == 2 and stderr_code(r) == "E_RECONSTRUCT_MISMATCH",
          f"rc={r.returncode} err={r.stderr.strip()[:120]}")

    # -- 6. version switch on plain bake -> full invalidation ----------------
    r = run(["bake", "--heightfield", g("edited.hf"), "--state-dir",
             g("st_edited_full"), "--out", g("vs.rgba"), "--log",
             g("vs.log.json"), "--threads", "4", "--spacing", "2"])
    log_vs = json.load(open(g("vs.log.json")))
    check("same-version re-bake is a no-op (all blocks skipped)",
          len(log_vs["blocks_baked_this_run"]) == 0)
    run(["gen-sample", "--out", g("other.hf"), "--width", "128",
         "--height", "128", "--seed", "99"])
    r = run(["bake", "--heightfield", g("other.hf"), "--state-dir",
             g("st_edited_full"), "--out", g("vs2.rgba"), "--log",
             g("vs2.log.json"), "--threads", "4", "--spacing", "2"])
    log_vs2 = json.load(open(g("vs2.log.json")))
    check("version change wipes state and rebakes everything",
          len(log_vs2["blocks_baked_this_run"]) == log_vs2["blocks_total"],
          f"{len(log_vs2['blocks_baked_this_run'])}/{log_vs2['blocks_total']}")

    # -- 7. dry-run ----------------------------------------------------------
    before = set(os.listdir(WORK))
    r = run(["bake", "--heightfield", g("base.hf"), "--state-dir",
             g("st_dry"), "--out", g("dry.rgba"), "--log", g("dry.log.json"),
             "--spacing", "2", "--dry-run"])
    plan = json.loads(r.stdout)
    check("dry-run reports plan", r.returncode == 0
          and plan["total_blocks"] == 64 and plan["block_grid"] == [8, 8]
          and plan["estimated_memory_bytes"] > 0)
    check("dry-run writes nothing",
          not os.path.exists(g("st_dry")) and not os.path.exists(g("dry.rgba"))
          and not os.path.exists(g("dry.log.json")))

    # -- 8. memory budget -----------------------------------------------------
    r = run(["bake", "--heightfield", g("base.hf"), "--state-dir", g("st_mem"),
             "--out", g("mem.rgba"), "--log", g("mem.log.json"),
             "--spacing", "2", "--threads", "4", "--max-cached-blocks", "2"])
    check("memory budget exceeded -> E_MEMORY_LIMIT, exit 2",
          r.returncode == 2 and stderr_code(r) == "E_MEMORY_LIMIT",
          f"rc={r.returncode} err={r.stderr.strip()}")

    # -- 9. invalid inputs ----------------------------------------------------
    raw = open(g("base.hf"), "rb").read()
    bad = bytearray(raw); bad[0] ^= 0xFF
    open(g("bad_magic.hf"), "wb").write(bytes(bad))
    r = run(["bake", "--heightfield", g("bad_magic.hf"), "--state-dir",
             g("st_e1"), "--out", g("e1.rgba"), "--log", g("e1.log.json")])
    check("bad magic -> E_HEADER_MISMATCH, exit 2",
          r.returncode == 2 and stderr_code(r) == "E_HEADER_MISMATCH")

    bad = bytearray(raw); bad[-4:] = b"\x00\x00\xc0\x7f"  # NaN payload
    open(g("bad_hash.hf"), "wb").write(bytes(bad))
    r = run(["bake", "--heightfield", g("bad_hash.hf"), "--state-dir",
             g("st_e2"), "--out", g("e2.rgba"), "--log", g("e2.log.json")])
    check("payload/hash mismatch -> E_HEADER_MISMATCH, exit 2",
          r.returncode == 2 and stderr_code(r) == "E_HEADER_MISMATCH")

    import hashlib
    hdr = bytearray(raw[:40])
    payload = bytearray(raw[40:])
    payload[-4:] = b"\x00\x00\xc0\x7f"  # NaN, then fix the hash
    hdr[24:40] = hashlib.blake2b(bytes(payload), digest_size=16).digest()
    open(g("bad_nan.hf"), "wb").write(bytes(hdr) + bytes(payload))
    r = run(["bake", "--heightfield", g("bad_nan.hf"), "--state-dir",
             g("st_e3"), "--out", g("e3.rgba"), "--log", g("e3.log.json")])
    check("NaN sample -> E_HEIGHT_INVALID, exit 2",
          r.returncode == 2 and stderr_code(r) == "E_HEIGHT_INVALID")

    trunc = raw[:-8]
    open(g("bad_len.hf"), "wb").write(trunc)
    r = run(["bake", "--heightfield", g("bad_len.hf"), "--state-dir",
             g("st_e4"), "--out", g("e4.rgba"), "--log", g("e4.log.json")])
    check("truncated payload -> E_HEADER_MISMATCH, exit 2",
          r.returncode == 2 and stderr_code(r) == "E_HEADER_MISMATCH")

    # -- 10. log comparison ----------------------------------------------------
    r = run(["compare-logs", g("full.log.json"), g("vs2.log.json")])
    rep = json.loads(r.stdout)
    check("compare-logs: version change demands full rebake",
          rep["result"] == "full_rebake_required" and "reason" in rep)
    r = run(["compare-logs", g("full.log.json"), g("int.log.json")])
    rep = json.loads(r.stdout)
    check("compare-logs: same version lists block diff",
          rep["result"] == "same_version"
          and rep.get("blocks_only_in_first")
          and not rep.get("blocks_only_in_second"),
          json.dumps(rep))

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
