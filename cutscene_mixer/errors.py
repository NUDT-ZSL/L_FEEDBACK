"""Error codes emitted by the mixer."""

SCHEMA_INVALID = "SCHEMA_INVALID"
AUDIO_LOAD_FAIL = "AUDIO_LOAD_FAIL"


class SchemaError(Exception):
    """Raised when the descriptor fails validation.

    ``paths`` holds the JSON field paths that failed validation.
    """

    def __init__(self, paths):
        self.paths = list(paths)
        super().__init__("%s: %s" % (SCHEMA_INVALID, ", ".join(self.paths)))
