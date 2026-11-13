"""Business error codes and the single pipeline exception type.

Business rejections (bad user configuration / unavailable backend) raise
``PipelineError`` carrying a stable string code.  The CLI converts these to
exit code 0 with the code written to stderr, while any other exception
propagates and yields a non-zero exit code.
"""

from __future__ import annotations

E_RES_TOO_HIGH = "E_RES_TOO_HIGH"
E_BAD_FRUSTUM = "E_BAD_FRUSTUM"
E_NO_GL = "E_NO_GL"
E_MESH_PATCH = "E_MESH_PATCH"

BUSINESS_CODES = frozenset(
    {E_RES_TOO_HIGH, E_BAD_FRUSTUM, E_NO_GL, E_MESH_PATCH}
)


class PipelineError(Exception):
    """Expected, code-bearing rejection produced by the pipeline."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def error_code(exc: BaseException) -> str | None:
    """Return the business code of ``exc`` or ``None`` for unexpected errors."""
    if isinstance(exc, PipelineError):
        return exc.code
    return None
