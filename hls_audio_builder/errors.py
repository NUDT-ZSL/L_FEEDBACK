class BuilderError(Exception):
    exit_code = 1


class ConfigError(BuilderError):
    exit_code = 2


class ConsistencyError(BuilderError):
    exit_code = 3


class OverwriteError(BuilderError):
    exit_code = 4

