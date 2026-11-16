"""Typed business errors for the rhythm mixer.

Every error carries a stable business code. Configuration-style failures use
exit code 1, runtime / IO-style failures use exit code 2 (see ``cli``).
"""


class MixerError(Exception):
    """Base error carrying a machine-readable business code."""

    code = "E_INTERNAL"
    exit_code = 1

    def __init__(self, message, code=None, exit_code=None, detail=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if exit_code is not None:
            self.exit_code = exit_code
        self.message = message
        self.detail = detail

    def to_payload(self):
        payload = {"error": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ConfigError(MixerError):
    """Invalid chart / event configuration (exit 1)."""

    exit_code = 1


class RuntimeMixerError(MixerError):
    """Runtime / IO failure while reading audio or writing outputs (exit 2)."""

    exit_code = 2


# Concrete, documented business codes.
class BadWavError(RuntimeMixerError):
    code = "E_BAD_WAV"


class SampleRateError(RuntimeMixerError):
    code = "E_SAMPLE_RATE"


class UnknownLayerError(ConfigError):
    code = "E_UNKNOWN_LAYER"


class GainError(ConfigError):
    code = "E_GAIN"


class NotOnBeatError(ConfigError):
    code = "E_NOT_ON_BEAT"


class OutputNotWritableError(RuntimeMixerError):
    code = "E_OUTPUT"


class ChartError(ConfigError):
    code = "E_CHART"


class EventError(ConfigError):
    code = "E_EVENT"
