"""Centralized Numba JIT compilation helper with PyInstaller fallback.

This module exports `njit` and `jit` decorators. If Numba is installed and the
application is not running in a frozen PyInstaller state, it uses Numba's JIT.
Otherwise, it falls back to a zero-overhead pass-through decorator.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Define pass-through fallback decorators
def _passthrough_decorator(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        return func
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return decorator

# Check if running under PyInstaller freeze or if numba is absent
_use_numba = False
if not getattr(sys, "frozen", False):
    try:
        from numba import njit as _njit, jit as _jit
        _use_numba = True
    except ImportError:
        pass

if _use_numba:
    njit = _njit
    jit = _jit
else:
    njit = _passthrough_decorator
    jit = _passthrough_decorator
