"""Console interrupt handling.

Signal handlers only flip a flag; the synthesis loop checks that flag at safe
points (between intervals), flushes a complete checkpoint and exits with code
0.  Hard process kills cannot run cleanup, so recovery must rely solely on the
last durable checkpoint rather than any in-memory or exit-report state.
"""

import os
import signal


class InterruptController:
    def __init__(self):
        self.requested = False
        self.signal_number = None
        self._previous = {}

    def _handler(self, signum, frame):
        self.requested = True
        self.signal_number = signum

    def install(self) -> None:
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for name in signals:
            self._previous[name] = signal.getsignal(name)
            signal.signal(name, self._handler)

    def restore(self) -> None:
        for name, previous in self._previous.items():
            signal.signal(name, previous)
        self._previous.clear()


def simulated_interrupt_interval() -> int:
    """Optional test hook: request an interrupt after N processed intervals.

    Disabled unless VOICEFORGE_INTERRUPT_AFTER is a positive integer.
    """
    raw = os.environ.get("VOICEFORGE_INTERRUPT_AFTER", "")
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if value > 0 else 0

