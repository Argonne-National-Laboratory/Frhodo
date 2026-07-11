"""Long-lived multiprocessing pool reused across optimization runs.

The pool's workers each hold a frozen ``ChemicalMechanism`` built once
at spawn time. On Windows the per-worker import + Cantera Solution
build is ~10–20s; reusing the pool across opt runs amortizes that to
one-per-app-session.

The pool is recreated lazily when any of these change:

  * the worker payload's CONTENT changed (mechanism structure or
    coefficients differ from what the workers were spawned with).
    Content equality — not ``mech.gas`` object identity — is the reuse
    test: the Plog→Troe recast rebuilds the Solution object every
    optimization run, but its deterministic fits produce an identical
    payload, and workers are a pure function of the payload.
  * requested worker count exceeds the current size (the pool grows;
    it never shrinks since idle workers are cheap)

Lifecycle:
  * created lazily on first :meth:`acquire`
  * closed by :meth:`close` (called from ``Main.closeEvent``)
  * registered with ``atexit`` so a crash before ``closeEvent`` still
    terminates the worker processes
"""
from __future__ import annotations

import atexit
import hashlib
import multiprocessing as mp
import pickle
import threading
from typing import Any, Callable

from frhodo.optimize._worker_context import MechBuildPayload
from frhodo.optimize.cost.fit_fcn import (
    _pool_reinit_worker,
    initialize_parallel_worker,
)



try:
    import psutil
except ImportError:
    psutil = None

_LogFn = Callable[[str], None]


def default_worker_count(n_shocks=None):
    """Bench-derived pool size: ~2/3 of logical processors.

    CVODE-bound simulations contend for FP units, so hyperthread
    oversubscription costs throughput — on a 12-physical/24-logical
    host, 16 workers evaluated ~7% faster than 26 while spawning ~40%
    fewer processes. Physical-core counts are preferred when psutil is
    available (physical + 1/3 margin matches the same optimum).
    """
    physical = None
    if psutil is not None:
        physical = psutil.cpu_count(logical=False)
    if physical:
        workers = max(2, physical + physical // 3)
    else:
        workers = max(2, (mp.cpu_count() * 2) // 3)
    if n_shocks is not None:
        workers = min(workers, max(1, int(n_shocks)))

    return workers

# In-place worker re-init barrier timeout; past this a worker is
# presumed dead and the pool respawns instead.
_REINIT_TIMEOUT_S = 120.0


def _payload_fingerprint(payload: MechBuildPayload) -> str:
    """Content hash of the worker payload; equal fingerprints mean the
    spawned workers' mechanisms are identical."""
    digest = hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()

    return digest


class PersistentWorkerPool:
    """``mp.Pool`` whose workers survive across optimization runs.

    Caller flow:

        pool_mgr = PersistentWorkerPool()
        ...
        pool = pool_mgr.acquire(workers=6, payload=payload)
        # use pool …
        # do NOT call pool.close() — pool_mgr owns its lifetime
        ...
        pool_mgr.close()  # on app exit
    """

    def __init__(self) -> None:
        self._pool: Any = None
        self._size: int = 0
        self._payload_hash: str | None = None
        # "spawned" | "reinit" | "reused" — what the last acquire did.
        self.last_acquire_action = "spawned"
        # True once a staged numba warmup has run for this pool
        # generation; respawns reset it (fresh processes, fresh race).
        self.warmed = False
        # Serializes acquire/close across threads (the load-time
        # pre-spawn runs off the GUI thread).
        self._lock = threading.Lock()
        # One Manager for the pool's lifetime: spawning one per re-init
        # costs a fresh process each time on Windows.
        self._manager = None
        atexit.register(self.close)

    @property
    def running(self) -> bool:
        """Whether a worker fleet is currently up (acquire would reuse
        or re-initialize it rather than spawn one)."""
        return self._pool is not None

    def acquire(
        self,
        *,
        workers: int,
        payload: MechBuildPayload,
    ) -> Any:
        """Return a ready pool: reused as-is on identical payload
        content, re-initialized in place on changed content, and
        respawned only when grown or when in-place re-init fails."""
        with self._lock:
            return self._acquire_locked(workers, payload)

    def _acquire_locked(self, workers: int, payload: MechBuildPayload) -> Any:
        payload_hash = _payload_fingerprint(payload)
        if self._needs_respawn(workers):
            self._respawn(workers, payload, payload_hash)
            self.last_acquire_action = "spawned"
            self.warmed = False
        elif payload_hash != self._payload_hash:
            try:
                self._reinit_workers(payload)
                self._payload_hash = payload_hash
                self.last_acquire_action = "reinit"
            except Exception:
                # A worker died or the barrier timed out; a fresh spawn
                # is the safe fallback.
                self._respawn(workers, payload, payload_hash)
                self.last_acquire_action = "spawned"
                self.warmed = False
        else:
            self.last_acquire_action = "reused"

        return self._pool

    def _respawn(self, workers, payload, payload_hash):
        # Called with self._lock held — must use the unlocked teardown.
        self._close_locked()
        self._pool = mp.Pool(
            processes=workers,
            initializer=initialize_parallel_worker,
            initargs=(payload,),
        )
        self._size = workers
        self._payload_hash = payload_hash

    def _reinit_workers(self, payload: MechBuildPayload) -> None:
        """Rebuild every live worker's mechanism in place — far cheaper
        than respawning processes (imports dominate spawn cost)."""
        if self._manager is None:
            self._manager = mp.Manager()
        barrier = self._manager.Barrier(self._size)
        args = [(payload, barrier, _REINIT_TIMEOUT_S)] * self._size
        self._pool.map(_pool_reinit_worker, args, chunksize=1)

    def _needs_respawn(self, workers: int) -> bool:
        if self._pool is None:
            return True
        if self._size < workers:
            return True

        return False

    def close(self) -> None:
        """Tear the pool down. Idempotent; safe to call multiple times."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
        if self._pool is None:
            return

        pool = self._pool
        self._pool = None
        self._size = 0
        self._payload_hash = None
        self.warmed = False

        try:
            # terminate, not close+join: close waits for workers that
            # may still be importing (a growth respawn can hit a pool
            # whose spawn hasn't finished), and nothing is ever
            # mid-task at teardown time.
            pool.terminate()
            pool.join()
        except Exception:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
