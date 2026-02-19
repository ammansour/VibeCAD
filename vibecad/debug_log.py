"""In-memory debug log capture for the KiCad UI.

KiCad's Python console/log output isn't always visible to plugin users.
This module provides a small ring buffer and a logging.Handler so the UI
can show recent debug output in a dedicated Debug tab.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, Optional


class InMemoryLogBuffer:
    def __init__(self, max_lines: int = 800):
        self._max_lines = int(max(50, max_lines))
        self._lines: Deque[str] = deque(maxlen=self._max_lines)
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        if not line:
            return
        with self._lock:
            self._lines.append(str(line))

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    def get_text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)


class _VibeCADOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return bool(record.name and record.name.startswith("vibecad"))
        except Exception:
            return False


class BufferLogHandler(logging.Handler):
    """Logging handler that writes formatted records into an InMemoryLogBuffer."""

    def __init__(self, buffer: InMemoryLogBuffer, level: int = logging.INFO):
        super().__init__(level=level)
        self._buffer = buffer
        self.addFilter(_VibeCADOnlyFilter())
        self.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        # Marker so we don't install multiple times across reloads.
        self._vibecad_debug_handler = True

    def set_buffer(self, buffer: InMemoryLogBuffer) -> None:
        """Rebind the handler to a new buffer (useful across plugin reloads)."""
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._buffer.append(msg)
        except Exception:
            # Never let debug capture break the plugin.
            pass


def install_debug_log_capture(
    buffer: InMemoryLogBuffer,
    level: int = logging.INFO,
    logger_name: str = "",
) -> Optional[BufferLogHandler]:
    """Install a BufferLogHandler on the given logger (default: root).

    Returns the active handler. If an existing handler was found, it is
    rebound to the provided buffer (so the Debug tab always reflects the
    current plugin instance).
    """

    target_logger = logging.getLogger(logger_name)
    for h in list(getattr(target_logger, "handlers", []) or []):
        if getattr(h, "_vibecad_debug_handler", False):
            try:
                if hasattr(h, 'set_buffer'):
                    h.set_buffer(buffer)
                else:
                    setattr(h, '_buffer', buffer)
                try:
                    h.setLevel(level)
                except Exception:
                    pass
            except Exception:
                pass
            return h

    handler = BufferLogHandler(buffer, level=level)
    try:
        target_logger.addHandler(handler)
    except Exception:
        return None

    # Ensure we actually receive child logs.
    try:
        if logger_name and hasattr(target_logger, "propagate"):
            target_logger.propagate = True
    except Exception:
        pass

    return handler
