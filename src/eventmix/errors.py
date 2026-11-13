"""Typed failures and exit-code categories."""


class EventMixError(Exception):
    """Base class for expected tool failures."""

    code = "eventmix_error"
    exit_code = 2


class ConfigError(EventMixError):
    code = "config_error"


class WavError(EventMixError):
    code = "wav_error"


class TrackInputError(EventMixError):
    code = "track_input_error"


class OutputCollisionError(ConfigError):
    code = "output_collision"


class CheckpointError(EventMixError):
    code = "checkpoint_error"


class ArtifactError(EventMixError):
    code = "artifact_error"

