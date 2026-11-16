import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sample_factory import build_sample_project  # noqa: E402


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    return build_sample_project(tmp_path)


@pytest.fixture()
def corrupt_project(tmp_path: Path) -> Path:
    return build_sample_project(tmp_path, corrupt=True)


@pytest.fixture()
def missing_project(tmp_path: Path) -> Path:
    return build_sample_project(tmp_path, missing=True)
