"""Centralized Numba JIT compilation helper with PyInstaller fallback.

This module exports ``njit`` and ``jit`` decorators.  If Numba is installed and
the application is not running in a frozen PyInstaller state, it uses Numba's
JIT.  Otherwise it falls back to a zero-overhead pass-through decorator.

``JIT_ENABLED`` and ``JIT_BACKEND`` are public diagnostics.  Performance tests
must only enforce compiled-kernel timing expectations when ``JIT_ENABLED`` is
true; correctness tests continue to run with the Python fallback.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _passthrough_decorator(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Return the decorated function unchanged when Numba is unavailable."""

    def decorator(func: F) -> F:
        return func

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return decorator


_use_numba = False
_numba_import_error: str | None = None
if not getattr(sys, "frozen", False):
    try:
        from numba import jit as _jit
        from numba import njit as _njit
        from numba import get_num_threads as _numba_get_num_threads
        from numba import prange as _numba_prange
        from numba import set_num_threads as _numba_set_num_threads

        _use_numba = True
    except ImportError as exc:
        _numba_import_error = str(exc)

if _use_numba:
    njit = _njit
    jit = _jit
    prange = _numba_prange
else:
    njit = _passthrough_decorator
    jit = _passthrough_decorator
    prange = range

JIT_ENABLED: bool = bool(_use_numba)
JIT_BACKEND: str = "numba" if JIT_ENABLED else "python"
JIT_DISABLED_REASON: str | None
if JIT_ENABLED:
    JIT_DISABLED_REASON = None
elif getattr(sys, "frozen", False):
    JIT_DISABLED_REASON = "frozen_application"
elif _numba_import_error:
    JIT_DISABLED_REASON = f"numba_import_failed: {_numba_import_error}"
else:
    JIT_DISABLED_REASON = "numba_not_installed"


def jit_diagnostics() -> dict[str, Any]:
    """Return the active kernel backend and fallback reason."""
    thread_count = None
    requested_threads = os.environ.get("FE_SOLVER_NUMBA_THREADS")
    if JIT_ENABLED:
        try:
            thread_count = int(_numba_get_num_threads())
        except Exception:
            thread_count = None
    return {
        "enabled": JIT_ENABLED,
        "backend": JIT_BACKEND,
        "disabled_reason": JIT_DISABLED_REASON,
        "frozen": bool(getattr(sys, "frozen", False)),
        "num_threads": thread_count,
        "requested_threads_env": requested_threads,
    }


def set_numba_threads(thread_count: int | None) -> int | None:
    """Set the active Numba worker count and return the previous count."""
    if not JIT_ENABLED or thread_count is None:
        return None
    value = int(thread_count)
    if value <= 0:
        raise ValueError("Numba thread count must be positive")
    previous = int(_numba_get_num_threads())
    _numba_set_num_threads(value)
    return previous


@contextlib.contextmanager
def numba_thread_scope(thread_count: int | None):
    """Temporarily set Numba threads for one solver phase."""
    previous = set_numba_threads(thread_count)
    try:
        yield
    finally:
        if previous is not None:
            _numba_set_num_threads(previous)
