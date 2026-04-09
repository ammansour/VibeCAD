"""Sleep prevention helpers for long-running agent sessions."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger("vibecad")


class SleepGuardMixin:
    """Manage a best-effort OS sleep guard process."""

    _sleep_guard_proc: Optional[Any]

    def _start_sleep_guard(self) -> None:
        """Prevent macOS sleep while an agent run is actively executing."""
        try:
            proc = self._sleep_guard_proc
            if proc is not None and proc.poll() is None:
                return
        except Exception:
            self._sleep_guard_proc = None

        if sys.platform != "darwin":
            return
        exe = shutil.which("caffeinate")
        if not exe:
            return

        try:
            self._sleep_guard_proc = subprocess.Popen(
                [exe, "-dimsu"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            logger.info("Sleep guard enabled via caffeinate")
        except Exception:
            self._sleep_guard_proc = None
            logger.exception("Failed to start sleep guard")

    def _stop_sleep_guard(self) -> None:
        """Stop the sleep-prevention helper process if it is running."""
        proc = self._sleep_guard_proc
        self._sleep_guard_proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=1.0)
            logger.info("Sleep guard disabled")
        except Exception:
            logger.exception("Failed to stop sleep guard")
