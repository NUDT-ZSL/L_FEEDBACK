"""spatmix: deterministic offline spatial-audio mixing pipeline."""

SCENE_VERSION = 1
SNAPSHOT_VERSION = 1
EVENT_LOG_VERSION = 1

EXIT_OK = 0
EXIT_ERROR = {
    "E_SCHEMA": 2,
    "E_TRACK_ORDER": 3,
    "E_RANGE": 4,
    "E_SNAPSHOT_VERSION": 5,
    "E_SNAPSHOT_MISMATCH": 6,
    "E_IO": 7,
}
