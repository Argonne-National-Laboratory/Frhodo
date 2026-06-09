"""Long-lived multiprocessing pool reused across optimization runs.

The pool's workers each hold a frozen ``ChemicalMechanism`` built once
at spawn time. On Windows the per-worker import + Cantera Solution
build is ~10–20s; reusing the pool across opt runs amortizes that to
one-per-app-session.

The pool is recreated lazily when any of these change:

  * the Cantera ``Solution`` was replaced (``mech.gas`` identity diff).
    Coefficient edits don't replace ``mech.gas`` — they mutate the
    existing reactions in place — so the persistent workers stay valid
    across normal opt iterations and only get rebuilt on structural
    rebuilds (mech file load, Plog→Troe recast, etc.).
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
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Callable

from frhodo.optimize._worker_context import MechBuildPayload
from frhodo.optimize.cost.fit_fcn import initialize_parallel_worker

if TYPE_CHECKING:
    from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism


_LogFn = Callable[[str], None]


class PersistentWorkerPool:
    """``mp.Pool`` whose workers survive across optimization runs.

    Caller flow:

        pool_mgr = PersistentWorkerPool()
        ...
        pool = pool_mgr.acquire(workers=6, mech=mech, payload=payload, log=log)
        # use pool …
        # do NOT call pool.close() — pool_mgr owns its lifetime
        ...
        pool_mgr.close()  # on app exit
    """

    def __init__(self) -> None:
        self._pool: Any = None
        self._size: int = 0
        # Strong reference to the Cantera Solution the workers were
        # spawned from. Identity check on ``mech.gas`` detects structural
        # rebuilds without needing a separate version counter; holding a
        # ref also prevents stale-id collisions if the old Solution were
        # freed and a new one happened to land at the same address.
        self._frozen_gas: Any = None
        atexit.register(self.close)

    def acquire(
        self,
        *,
        workers: int,
        mech: "ChemicalMechanism",
        payload: MechBuildPayload,
    ) -> Any:
        """Return a ready pool, growing or rebuilding it as needed."""
        if self._needs_rebuild(workers, mech):
            self.close()
            self._pool = mp.Pool(
                processes=workers,
                initializer=initialize_parallel_worker,
                initargs=(payload,),
            )
            self._size = workers
            self._frozen_gas = mech.gas

        return self._pool

    def _needs_rebuild(self, workers: int, mech: "ChemicalMechanism") -> bool:
        if self._pool is None:
            return True
        if self._size < workers:
            return True
        if mech.gas is not self._frozen_gas:
            return True

        return False

    def close(self) -> None:
        """Tear the pool down. Idempotent; safe to call multiple times."""
        if self._pool is None:
            return

        pool = self._pool
        self._pool = None
        self._size = 0
        self._frozen_gas = None

        try:
            pool.close()
            pool.join()
        except Exception:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
