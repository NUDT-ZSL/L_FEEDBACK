"""Business error codes (string valued) used across the pipeline and report."""


class ErrorCode:
    OK = "OK"
    E_CONFIG = "E_CONFIG"                  # invalid configuration / CLI usage
    E_HEIGHTFIELD = "E_HEIGHTFIELD"        # heightfield generation / sampling failure
    E_NORMALMAP = "E_NORMALMAP"            # normal map bake failure
    E_MESH = "E_MESH"                      # LOD mesh generation failure
    E_NORMAL_MISMATCH = "E_NORMAL_MISMATCH"  # consistency check above threshold
    E_TASK_EXECUTION = "E_TASK_EXECUTION"  # unexpected exception inside a bake task
    E_STATE_CORRUPT = "E_STATE_CORRUPT"    # state file unreadable / incompatible
    E_IO = "E_IO"                          # output persistence failure


class BakeError(Exception):
    """Exception carrying a business error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
