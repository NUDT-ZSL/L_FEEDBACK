#!/usr/bin/env python3
"""Generate a C translation unit with more than 20,000 real functions."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--count", type=int, default=20001)
    args = parser.parse_args()
    lines = ["#include <windows.h>", ""]
    for index in range(args.count):
        lines.extend(
            [
                "__declspec(noinline)",
                "int large_function_%05d(int value)" % index,
                "{",
                "    return value + %d;" % index,
                "}",
                "",
            ]
        )
    lines.extend(
        [
            "int main(void)",
            "{",
            "    return large_function_00000(GetCurrentProcessId());",
            "}",
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
