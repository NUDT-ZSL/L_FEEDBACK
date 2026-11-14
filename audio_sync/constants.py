"""Shared audio / protocol constants."""

SAMPLE_RATE = 48000
FRAME_DURATION_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 960 samples
SAMPLE_BYTES = 2  # 16-bit PCM
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_BYTES  # 1920 bytes

# Custom binary header: uint16 seq, uint32 timestamp, uint16 payload length.
HEADER_FORMAT = "<HIH"
HEADER_SIZE = 8

SEQ_MOD = 1 << 16  # 16-bit sequence number wrap
TS_MOD = 1 << 32  # 32-bit timestamp wrap

MIN_STREAMS = 2
MAX_STREAMS = 4

MIN_BUFFER_DEPTH = 2
MAX_BUFFER_DEPTH = 10

# Resource constraint: per-stream jitter buffer memory must stay below 256 KB.
MAX_BUFFER_BYTES = 256 * 1024

# PLC: at most this many consecutive frames are actively concealed; longer
# loss runs are replaced by silence until a recovery point is observed.
MAX_PLC_FRAMES = 3

# Business error codes emitted to stderr.
ERR_TRUNCATED = "E_TRUNCATED"
ERR_BAD_LENGTH = "E_BAD_LENGTH"
ERR_DUPLICATE = "E_DUPLICATE"
ERR_SEQ_GAP = "W_SEQ_GAP"
ERR_SEQ_REORDER = "I_SEQ_REORDER"
ERR_TRAILING = "E_TRAILING_BYTES"

# Buffer / queue events recorded on frames and the summary.
EVENT_OVERFLOW = "buffer_overflow"
EVENT_UNDERFLOW = "buffer_underflow"
EVENT_BACKPRESSURE = "backpressure"
EVENT_LATE = "late_frame_dropped"
EVENT_RECOVERY = "plc_recovery"

# Mixer activity gate: normalized RMS below this is treated as silence.
ACTIVITY_RMS_THRESHOLD = 0.01
