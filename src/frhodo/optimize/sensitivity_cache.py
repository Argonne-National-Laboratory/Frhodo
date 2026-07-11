"""Shared in-memory cache of start-mechanism solves and sensitivities.

One process-wide, byte-capped LRU store shared by every consumer that
solves at the start mechanism: campaign screening (GUI ranking runs and
the optimizer's uniqueness-weighting pass) and the Sim Explorer
sensitivity views. Entries are keyed on a content fingerprint of the
mechanism coefficients plus everything else the solves depend on, so a
result computed by one consumer is a cache hit for the others and any
coefficient change invalidates naturally.

Optimizer inner sweeps are excluded: they evaluate thousands of distinct
perturbed mechanisms, each visited once, so caching them only pays the
fingerprint and eviction overhead without hits.
"""
import hashlib
import pickle
import threading
from collections import OrderedDict

import numpy as np



DEFAULT_MAX_BYTES = 250 * 2**20


def mech_fingerprint(mech) -> bytes:
    """Content hash of the mechanism's rate coefficients."""
    try:
        blob = pickle.dumps(mech.coeffs, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        version = int(getattr(mech, "coeffs_version", 0))

        return b"version:%d" % version

    return hashlib.sha256(blob).digest()


def grid_fingerprint(t_grid) -> bytes | None:
    """Content hash of an explicit output time grid; ``None`` passes
    through (solver-native grid)."""
    if t_grid is None:
        return None

    arr = np.ascontiguousarray(t_grid, dtype=np.float64)

    return hashlib.sha256(arr.tobytes()).digest()


def sim_cache_key(kind, mech_fp, reactor_state, shock, observable,
                  species_idx, method, grid_fp=None) -> tuple:
    """Everything the trajectory + sensitivity solves depend on.

    ``kind`` separates entry layouts ("screen" bundles vs "sens" pairs)
    that would otherwise collide on identical solve inputs.
    """
    mix = getattr(shock, "thermo_mix", None) or {}
    key = (
        kind,
        mech_fp,
        int(getattr(shock, "num", 0) or 0),
        float(shock.T_reactor), float(shock.P_reactor),
        tuple(sorted((str(k), float(v)) for k, v in dict(mix).items())),
        repr(reactor_state), observable, species_idx, method, grid_fp,
    )

    return key


def _value_nbytes(value) -> int:
    total = 0
    for item in value:
        if isinstance(item, np.ndarray):
            total += item.nbytes

    return total


class SensitivityCache:
    """Thread-safe, byte-capped LRU store of solve results.

    Values are tuples of ndarrays (``None`` members allowed); the byte
    cap counts array payloads only. Inserting a value larger than the
    cap stores nothing.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple, tuple] = OrderedDict()
        self._bytes = 0
        self._max_bytes = int(max_bytes)

    def get(self, key: tuple):
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)

        return value

    def put(self, key: tuple, value: tuple) -> None:
        nbytes = _value_nbytes(value)
        if nbytes > self._max_bytes:
            return

        with self._lock:
            old = self._store.pop(key, None)
            if old is not None:
                self._bytes -= _value_nbytes(old)
            self._store[key] = value
            self._bytes += nbytes
            while self._bytes > self._max_bytes:
                _, evicted = self._store.popitem(last=False)
                self._bytes -= _value_nbytes(evicted)

    def prune_stale(self, mech_fp: bytes) -> None:
        """Drop entries keyed to any other mechanism fingerprint."""
        with self._lock:
            stale = [k for k in self._store if k[1] != mech_fp]
            for key in stale:
                self._bytes -= _value_nbytes(self._store.pop(key))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._bytes = 0

    def set_max_bytes(self, max_bytes: int) -> None:
        with self._lock:
            self._max_bytes = int(max_bytes)
            while self._bytes > self._max_bytes and self._store:
                _, evicted = self._store.popitem(last=False)
                self._bytes -= _value_nbytes(evicted)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._store)


shared_cache = SensitivityCache()
