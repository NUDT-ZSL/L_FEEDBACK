# cfaudit

`cfaudit` is a standard-library-only Python verifier for closed-source Windows
PE/PDB attack surface. It parses the binary formats directly, reconstructs
function ranges and call sites from private CodeView symbols and a small
x86/x64 instruction decoder, propagates declared taint sources, and returns
deterministic sink paths for patch regression gates.

The tool never loads or executes code from the input PE/PDB. Inputs are opened
read-only, bounded to 30 MiB (PE) and 60 MiB (PDB), and malformed structures
return structured `INVALID_PE`/`INVALID_PDB` errors instead of tracebacks.

## Commands

```powershell
.\cfaudit.cmd parse --pe pe64.exe --pdb pe64.pdb --out p64.json
.\cfaudit.cmd analyze --pe sample.exe --pdb sample.pdb --config config.json --out report.json
.\cfaudit.cmd diff --before old.exe --after new.exe --pdb old.pdb --config config.json --out diff.json
.\cfaudit.cmd diff --baseline baseline.json --after new.exe --pdb new.pdb --config config.json
```

`cfaudit.cmd` uses `py -3`. The package can also be installed with
`py -3 -m pip install .`, which creates a `cfaudit` console entry point, or run
with `py -3 -m cfaudit`.

Exit codes:

- `0`: command completed and the gate passed.
- `1`: diff found one or more regressed sinks.
- `2`: parse/config/version error, including `PDB_MISMATCH`, `MISSING_PDATA`,
  `INVALID_PE`, `INVALID_PDB`, and `BASELINE_VERSION_MISMATCH`.
- `3`: the isolated analysis process was terminated by its Windows Job Object
  for `memory` or `time`.

## Source Configuration

```json
{
  "sources": [
    {"function": "source_a", "param_index": 1, "type": "string"}
  ],
  "dangerous_apis": ["optional_extra_sink"],
  "sanitizers": ["sanitize", "safe_copy"],
  "module_hints": ["sample"]
}
```

`param_index` is zero-based. On Microsoft x64 the analyzer seeds
`RCX/RDX/R8/R9` and stack overflow slots; on x86 it seeds cdecl/stdcall stack
offsets. Unknown or unresolved data flow is conservatively made clean and
counted in `analysis_stats.unknown_propagation`.

The built-in dangerous API table includes `memcpy`, `strcpy`,
`CreateProcessW`, `WinExec`, `LoadLibraryW`, and `CreateFileW`. A sink record
uses the enclosing function symbol as its stable call-point identity. This
makes call-points such as `sink_memcpy`, `sink_win_exec`, and
`sink_create_file` stable across relinks while still exposing the underlying
API in `api`/`apis`.

## Analysis Semantics

- Function ranges come from `.pdata` and are augmented with PDB
  `S_GPROC32`/`S_LPROC32` names; `S_PUB32` records are parsed separately.
- Calls through MSVC ILT thunks, direct relative calls, and x86/x64 IAT
  references are resolved without executing the image.
- Taint propagation is limited to three call-graph edges. Cycles are pruned
  and reported as `RECURSION_CUT`.
- `DIRECT` means a tainted dangerous argument is reached by a path of at most
  two function names. Longer paths are `CONDITIONAL`. No tainted path is
  `UNREACHABLE`.
- If multiple sources reach one sink, `tainted_sources` lists all of them.
  `shortest_path` chooses the shortest path, breaking ties by the
  lexicographic function-name sequence.
- Warnings (`NAME_CONFLICT`, `SOURCE_NOT_FOUND`, `CONV_MISMATCH`,
  `RECURSION_CUT`) do not change the exit code unless another error exists.

## Resource Isolation

`analyze` and `diff` run in a child Python process. On Windows the parent:

1. creates a Job Object;
2. sets `JOB_OBJECT_LIMIT_JOB_TIME` and
   `JOB_OBJECT_LIMIT_PROCESS_MEMORY`;
3. assigns the worker process to the Job Object;
4. monitors accounting and uses `TerminateJobObject` on a real limit breach;
5. maps the termination to `RESOURCE_LIMIT` with `resource` equal to
   `"time"` or `"memory"`.

`--mem-limit` and `--cpu-limit-ms` are available on both `analyze` and
`diff`. The POSIX fallback uses `RLIMIT_AS` and `RLIMIT_CPU`; the shipped
acceptance harness exercises the Windows Job Object path.

## Test Fixtures

All binaries used by the verifier are generated at test time from C sources in
`samples/` with `cl.exe` and `link.exe`; no PE/PDB binaries are checked in.
`tests/run_acceptance.ps1` discovers VS via `vswhere`, imports the x64/x86
environment, builds the fixtures, invokes `cfaudit.cmd` through PowerShell,
and asserts JSON and exit-code behavior.

The x64/x86 PE pair test intentionally runs a small post-link section-name
normalization (`tests/patch_pair_sections.py`) so the linker-produced import
section is named `.rdata` for the required IAT location assertion. The binary
contents, imports, unwind data, PDB pairing, and CodeView GUID/Age are otherwise
entirely produced by `cl.exe`/`link.exe`.

Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\run_acceptance.ps1
```

## Actual Verification Record

The full acceptance script was executed on this Windows 10/WSL2 host using
Visual Studio 2019 Build Tools and Python 3.12 (`py -3`). All checks passed:

- x64 `.pdata` count matched `dumpbin /unwindinfo`: `1505`.
- x86 parse exited `2`, returned zero functions, and reported
  `MISSING_PDATA`; section RVAs were sorted and names contained no NUL.
- `CreateFileW.iat_rva` was inside the fixture's `.rdata` range.
- Matched PE/PDB parse returned named RVA ranges; mismatched GUID/Age returned
  `PDB_MISMATCH`, both GUID/Age pairs, exit `2`, and no `sinks`.
- Sample analysis produced `sink_memcpy` `DIRECT` with path
  `source_a -> sink_memcpy`, `sink_win_exec` `CONDITIONAL` with a four-name
  path, `sink_network` `CONDITIONAL` with tainted parameter `0`, and
  `sink_create_file` `UNREACHABLE`.
- `param_index=0` left `sink_network` with no tainted parameters and
  `UNREACHABLE`.
- Repeated analyses produced byte-identical sink path arrays.
- Patch diff output `improved=[sink_memcpy]`, `regressed=[]`, `passed=true`,
  exit `0`; the regression fixture produced a `CONDITIONAL -> DIRECT`
  regression, `passed=false`, exit `1`.
- Baseline schema `0.9` returned `BASELINE_VERSION_MISMATCH`, exit `2`, and
  omitted `improved`/`regressed`.
- The multi-source fixture reported both `source_a` and `source_b` in
  `tainted_sources`, selected `source_b -> sink_memcpy` as the shortest path,
  and emitted `RECURSION_CUT`.
- The warning fixture emitted `SOURCE_NOT_FOUND` for `missing_func`,
  `NAME_CONFLICT` with both names, and `CONV_MISMATCH`, while exiting `0`.
- Three malformed inputs returned `INVALID_PE` or `INVALID_PDB`, exit `2`,
  empty stderr, and no new filesystem entries except `--out`.
- `--mem-limit 16777216` terminated the worker with
  `RESOURCE_LIMIT/resource=memory` and exit `3`.
- A 20,001-function PE/PDB fixture with `--cpu-limit-ms 100` terminated the
  worker with `RESOURCE_LIMIT/resource=time` and exit `3`; no worker process
  remained.

The final acceptance run printed `ALL ACCEPTANCE CHECKS PASSED`.
