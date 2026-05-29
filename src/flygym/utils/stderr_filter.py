"""Filter noisy MuJoCo-Warp CCD iteration warnings from stderr."""

from __future__ import annotations

import os
import re
import sys
import threading

CCD_ITERATIONS_WARNING_RE = re.compile(
    r"Warning:\s+opt\.ccd_iterations,\s+currently set to \d+,\s+needs to be increased\."
)

_FILTER_INSTALLED = False
_FILTER_LOCK = threading.Lock()


def should_filter_stderr_line(line: str) -> bool:
    """Return True if ``line`` is a MuJoCo-Warp CCD iteration warning."""
    return bool(CCD_ITERATIONS_WARNING_RE.search(line))


def install_ccd_iterations_warning_filter() -> None:
    """Drop MuJoCo-Warp ``opt.ccd_iterations`` warnings from process stderr.

    MJWarp emits these through ``wp.printf`` on file descriptor 2, so a Python
    ``sys.stderr`` wrapper alone is not enough. This installs a line filter on
    the underlying stderr fd and is safe to call multiple times.
    """
    global _FILTER_INSTALLED
    with _FILTER_LOCK:
        if _FILTER_INSTALLED:
            return
        _FILTER_INSTALLED = True
        _install_fd_stderr_filter()


def _install_fd_stderr_filter() -> None:
    read_fd, write_fd = os.pipe()
    saved_stderr_fd = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _forward() -> None:
        reader = os.fdopen(read_fd, "rb", closefd=True)
        writer = os.fdopen(saved_stderr_fd, "wb", closefd=False)
        pending = b""
        try:
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    _write_filtered_line(writer, line + b"\n")
        finally:
            if pending:
                _write_filtered_line(writer, pending)
            writer.flush()
            reader.close()

    threading.Thread(
        target=_forward,
        name="flygym-ccd-warning-filter",
        daemon=True,
    ).start()

    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)  # type: ignore[assignment]


def _write_filtered_line(writer, line: bytes) -> None:
    text = line.decode("utf-8", errors="replace")
    if should_filter_stderr_line(text):
        return
    writer.write(line)
    writer.flush()
