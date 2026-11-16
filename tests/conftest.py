from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rhythmforge.generator import SOURCE_SPECS, generate_project  # noqa: E402


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    base = tmp_path_factory.mktemp("project")
    samples_dir = str(base / "samples")
    chart_path = str(base / "chart.json")
    chart = generate_project(samples_dir, chart_path)
    return {
        "base": str(base),
        "samples_dir": samples_dir,
        "chart_path": chart_path,
        "chart": chart,
        "output_dir": str(base / "output"),
    }


@pytest.fixture(scope="session")
def built_project(project):
    from rhythmforge.pipeline import build

    manifest = build(project["chart_path"], project["samples_dir"],
                     project["output_dir"])
    project["manifest"] = manifest
    return project


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end build tests")


def expected_format_combinations():
    rates = {spec.sample_rate for spec in SOURCE_SPECS}
    channels = {spec.channels for spec in SOURCE_SPECS}
    widths = {spec.sample_width for spec in SOURCE_SPECS}
    return rates, channels, widths
