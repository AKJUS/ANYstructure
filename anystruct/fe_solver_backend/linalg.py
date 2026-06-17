"""Sparse linear algebra backend for FE analyses.

The first production backend is intentionally small and conservative: SciPy's
SuperLU handles all matrix classes, while the interface records matrix class,
ordering, timings and failure reasons.  Optional native backends can be added
behind the same API later without changing analysis modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu

try:
    from pypardiso import PyPardisoSolver
    _HAS_PYPARDISO = True
except ImportError:
    _HAS_PYPARDISO = False

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper
    prange = range
    _HAS_NUMBA = False

class MatrixClass(str, Enum):
    """Declared numerical class of a sparse matrix."""

    SPD = "spd"
    SYMMETRIC_SEMIDEFINITE = "symmetric_semidefinite"
    SYMMETRIC_INDEFINITE = "symmetric_indefinite"
    GENERAL = "general"


@dataclass
class FactorizationHandle:
    """Reusable sparse factorization and solve diagnostics."""

    matrix_shape: tuple[int, int]
    matrix_class: MatrixClass
    backend_name: str
    ordering: str
    signature: Optional[str]
    factorization_time: float
    factorization_count: int = 1
    solve_count: int = 0
    solve_time: float = 0.0
    status: str = "ok"
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _solver: Any = field(default=None, repr=False)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return solve(self, rhs)

    def solve_many(self, rhs_matrix: np.ndarray) -> np.ndarray:
        return solve_many(self, rhs_matrix)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "matrix_class": self.matrix_class.value,
            "ordering": self.ordering,
            "signature": self.signature,
            "shape": list(self.matrix_shape),
            "status": self.status,
            "failure_reason": self.failure_reason,
            "factorization_time": self.factorization_time,
            "factorization_count": self.factorization_count,
            "solve_count": self.solve_count,
            "solve_time": self.solve_time,
            **self.metadata,
        }


class SparseSolverBackend:
    """SciPy/SuperLU sparse backend with a stable FE-facing interface."""

    name = "scipy_superlu"

    def factorize(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> FactorizationHandle:
        options = dict(options or {})
        ordering = str(options.get("ordering", "COLAMD"))
        start = time.time()
        try:
            csc = sparse.csc_matrix(matrix)
            solver = splu(csc, permc_spec=ordering)
        except Exception as exc:
            return FactorizationHandle(
                matrix_shape=tuple(int(v) for v in matrix.shape),
                matrix_class=matrix_class,
                backend_name=self.name,
                ordering=ordering,
                signature=signature,
                factorization_time=time.time() - start,
                status="failed",
                failure_reason=str(exc),
            )
        return FactorizationHandle(
            matrix_shape=tuple(int(v) for v in matrix.shape),
            matrix_class=matrix_class,
            backend_name=self.name,
            ordering=ordering,
            signature=signature,
            factorization_time=time.time() - start,
            _solver=solver,
        )


class PyPardisoSolverBackend:
    """Intel MKL PARDISO backend using pypardiso."""

    name = "pypardiso"

    def factorize(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> FactorizationHandle:
        start = time.time()
        try:
            csc = sparse.csc_matrix(matrix)
            
            class PyPardisoWrapper:
                def __init__(self, mat):
                    self.ps = PyPardisoSolver()
                    self.mat = mat
                    self.ps.factorize(self.mat)
                def solve(self, rhs):
                    return self.ps.solve(self.mat, rhs)

            wrapper = PyPardisoWrapper(csc)
            
        except Exception as exc:
            return FactorizationHandle(
                matrix_shape=tuple(int(v) for v in matrix.shape),
                matrix_class=matrix_class,
                backend_name=self.name,
                ordering="MKL PARDISO",
                signature=signature,
                factorization_time=time.time() - start,
                status="failed",
                failure_reason=str(exc),
            )
        return FactorizationHandle(
            matrix_shape=tuple(int(v) for v in matrix.shape),
            matrix_class=matrix_class,
            backend_name=self.name,
            ordering="MKL PARDISO",
            signature=signature,
            factorization_time=time.time() - start,
            _solver=wrapper,
        )


DEFAULT_BACKEND = PyPardisoSolverBackend() if _HAS_PYPARDISO else SparseSolverBackend()


def _options_signature(options: Optional[Mapping[str, Any]]) -> str:
    return json.dumps(dict(options or {}), sort_keys=True, default=str, separators=(",", ":"))


def sparse_matrix_signature(matrix: sparse.spmatrix) -> str:
    """Content signature for a sparse matrix used in local factorization caches."""

    csr = sparse.csr_matrix(matrix)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in csr.shape)).encode("ascii"))
    digest.update(str(int(csr.nnz)).encode("ascii"))
    digest.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass
class FactorizationCache:
    """Explicit local cache for sparse factorizations.

    The cache is intentionally not global.  Analyses that can safely reuse a
    matrix factorization own the cache and therefore own its lifetime.
    """

    name: str = "factorization_cache"
    max_entries: int = 8
    backend: Optional[SparseSolverBackend] = None
    _handles: Dict[Tuple[str, str, str, str], FactorizationHandle] = field(default_factory=dict, init=False, repr=False)
    hits: int = 0
    misses: int = 0
    factorization_failures: int = 0

    def key(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass | str,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[str, str, str, str]:
        matrix_key = str(signature) if signature is not None else sparse_matrix_signature(matrix)
        return (
            matrix_key,
            _coerce_matrix_class(matrix_class).value,
            _options_signature(options),
            tuple(int(v) for v in matrix.shape).__repr__(),
        )

    def factorize(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass | str,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> FactorizationHandle:
        cache_key = self.key(matrix, matrix_class, signature=signature, options=options)
        if cache_key in self._handles:
            self.hits += 1
            handle = self._handles[cache_key]
            handle.metadata["cache_name"] = self.name
            handle.metadata["cache_hit"] = True
            return handle
        self.misses += 1
        handle = factorize(
            matrix,
            matrix_class,
            signature=signature or cache_key[0],
            options=options,
            backend=self.backend,
        )
        handle.metadata["cache_name"] = self.name
        handle.metadata["cache_hit"] = False
        if handle.status == "ok":
            if self.max_entries > 0 and len(self._handles) >= self.max_entries:
                oldest = next(iter(self._handles))
                self._handles.pop(oldest, None)
            if self.max_entries != 0:
                self._handles[cache_key] = handle
        else:
            self.factorization_failures += 1
        return handle

    def linear_operator(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass | str,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> tuple[LinearOperator, FactorizationHandle]:
        """Return a LinearOperator that applies the cached inverse."""

        handle = self.factorize(matrix, matrix_class, signature=signature, options=options)
        if handle.status != "ok":
            raise RuntimeError(f"Cannot create inverse operator from failed factorization: {handle.failure_reason}")

        def matvec(rhs: np.ndarray) -> np.ndarray:
            return handle.solve(rhs)

        operator = LinearOperator(matrix.shape, matvec=matvec, dtype=float)
        return operator, handle

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_entries": int(self.max_entries),
            "entries": int(len(self._handles)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "factorization_failures": int(self.factorization_failures),
            "backend": (self.backend or DEFAULT_BACKEND).name,
        }

    def clear(self) -> None:
        self._handles.clear()


def _coerce_matrix_class(matrix_class: MatrixClass | str) -> MatrixClass:
    if isinstance(matrix_class, MatrixClass):
        return matrix_class
    return MatrixClass(str(matrix_class))


def factorize(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    backend: Optional[SparseSolverBackend] = None,
) -> FactorizationHandle:
    """Factorize a sparse matrix and return a reusable handle."""
    backend = backend or DEFAULT_BACKEND
    return backend.factorize(matrix, _coerce_matrix_class(matrix_class), signature=signature, options=options)


def factorize_cached(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    cache: Optional[FactorizationCache] = None,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    backend: Optional[SparseSolverBackend] = None,
) -> FactorizationHandle:
    """Factorize through a local cache when supplied."""

    if cache is None:
        return factorize(matrix, matrix_class, signature=signature, options=options, backend=backend)
    if backend is not None and cache.backend is None:
        cache.backend = backend
    return cache.factorize(matrix, matrix_class, signature=signature, options=options)


def cached_inverse_operator(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    cache: Optional[FactorizationCache] = None,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> tuple[LinearOperator, FactorizationHandle]:
    """Build a sparse inverse operator, using a local cache if supplied."""

    local_cache = cache or FactorizationCache(name="single_inverse_operator", max_entries=1)
    return local_cache.linear_operator(matrix, matrix_class, signature=signature, options=options)


def solve(handle: FactorizationHandle, rhs: np.ndarray) -> np.ndarray:
    """Solve one right-hand side using an existing factorization."""
    if handle.status != "ok" or handle._solver is None:
        raise RuntimeError(f"Cannot solve with failed factorization: {handle.failure_reason}")
    rhs = np.asarray(rhs, dtype=float)
    start = time.time()
    result = np.asarray(handle._solver.solve(rhs), dtype=float)
    handle.solve_time += time.time() - start
    handle.solve_count += 1 if rhs.ndim == 1 else int(rhs.shape[1])
    if np.any(~np.isfinite(result)):
        raise RuntimeError("Sparse solve produced NaN/Inf values")
    return result


def solve_many(handle: FactorizationHandle, rhs_matrix: np.ndarray) -> np.ndarray:
    """Solve one or more right-hand sides with one numerical factorization."""
    rhs = np.asarray(rhs_matrix, dtype=float)
    if rhs.ndim == 1:
        return solve(handle, rhs)
    if rhs.ndim != 2:
        raise ValueError("rhs_matrix must be one- or two-dimensional")
    return solve(handle, rhs)
