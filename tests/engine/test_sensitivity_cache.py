"""Shared sensitivity cache: keys, fingerprints, and LRU byte-capping."""
from types import SimpleNamespace

import numpy as np

from frhodo.optimize.sensitivity_cache import (
    SensitivityCache,
    grid_fingerprint,
    mech_fingerprint,
    sim_cache_key,
)



def _arr(kb):
    return np.zeros(kb * 128, dtype=np.float64)  # 1 KB per 128 float64


def _shock(num=1, T=1500.0, P=2e4):
    shock = SimpleNamespace(
        num=num, T_reactor=T, P_reactor=P, thermo_mix={"Kr": 0.96},
    )

    return shock


class TestMechFingerprint:
    def test_equal_coeffs_give_equal_fingerprints(self):
        a = SimpleNamespace(coeffs=[{"A": 1.0, "b": 0.5}], coeffs_version=0)
        b = SimpleNamespace(coeffs=[{"A": 1.0, "b": 0.5}], coeffs_version=7)
        assert mech_fingerprint(a) == mech_fingerprint(b), (
            "fingerprint must depend on coefficient content, not the "
            "version counter"
        )

    def test_changed_coefficient_changes_fingerprint(self):
        mech = SimpleNamespace(coeffs=[{"A": 1.0}], coeffs_version=0)
        before = mech_fingerprint(mech)
        mech.coeffs[0]["A"] = 1.5
        assert mech_fingerprint(mech) != before

    def test_unpicklable_coeffs_fall_back_to_version(self):
        mech = SimpleNamespace(coeffs=[lambda: None], coeffs_version=3)
        assert mech_fingerprint(mech) == b"version:3"


class TestGridFingerprint:
    def test_none_passes_through(self):
        assert grid_fingerprint(None) is None

    def test_content_keyed(self):
        t = np.linspace(0.0, 1.0, 50)
        assert grid_fingerprint(t) == grid_fingerprint(t.copy())
        assert grid_fingerprint(t) != grid_fingerprint(t * 2.0)


class TestSimCacheKey:
    def test_kind_separates_entry_layouts(self):
        args = (b"fp", "reactor", _shock(), "T", None, "auto")
        assert sim_cache_key("screen", *args) != sim_cache_key("sens", *args)

    def test_condition_fields_enter_the_key(self):
        base = sim_cache_key(
            "screen", b"fp", "reactor", _shock(T=1500.0), "T", None, "auto",
        )
        hotter = sim_cache_key(
            "screen", b"fp", "reactor", _shock(T=1600.0), "T", None, "auto",
        )
        assert base != hotter


class TestSensitivityCache:
    def test_miss_returns_none_and_hit_returns_value(self):
        cache = SensitivityCache(max_bytes=10_000)
        value = (_arr(1), _arr(1))
        assert cache.get(("k",)) is None
        cache.put(("k",), value)
        assert cache.get(("k",)) is value

    def test_byte_cap_evicts_least_recently_used(self):
        cache = SensitivityCache(max_bytes=3 * 1024)
        for i in range(3):
            cache.put(("k", i), (_arr(1),))
        cache.get(("k", 0))  # refresh: k1 becomes the LRU entry
        cache.put(("k", 3), (_arr(1),))
        assert cache.get(("k", 1)) is None, "LRU entry must be evicted"
        assert cache.get(("k", 0)) is not None
        assert cache.get(("k", 3)) is not None
        assert cache.nbytes <= 3 * 1024

    def test_value_larger_than_cap_is_not_stored(self):
        cache = SensitivityCache(max_bytes=1024)
        cache.put(("big",), (_arr(2),))
        assert cache.get(("big",)) is None
        assert cache.nbytes == 0

    def test_replacing_a_key_updates_byte_accounting(self):
        cache = SensitivityCache(max_bytes=10_000)
        cache.put(("k",), (_arr(4),))
        cache.put(("k",), (_arr(1),))
        assert cache.nbytes == 1024
        assert len(cache) == 1

    def test_none_members_count_zero_bytes(self):
        cache = SensitivityCache(max_bytes=10_000)
        cache.put(("k",), (_arr(1), None, None, _arr(1)))
        assert cache.nbytes == 2 * 1024

    def test_prune_stale_drops_other_fingerprints_only(self):
        cache = SensitivityCache(max_bytes=10_000)
        cache.put(("screen", b"old", 1), (_arr(1),))
        cache.put(("screen", b"new", 1), (_arr(1),))
        cache.put(("sens", b"old", 2), (_arr(1),))
        cache.prune_stale(b"new")
        assert len(cache) == 1
        assert cache.get(("screen", b"new", 1)) is not None
        assert cache.nbytes == 1024

    def test_set_max_bytes_shrinks_immediately(self):
        cache = SensitivityCache(max_bytes=10_000)
        for i in range(4):
            cache.put(("k", i), (_arr(1),))
        cache.set_max_bytes(2 * 1024)
        assert len(cache) == 2
        assert cache.get(("k", 3)) is not None, "newest entries survive"

    def test_clear_empties_store_and_bytes(self):
        cache = SensitivityCache(max_bytes=10_000)
        cache.put(("k",), (_arr(1),))
        cache.clear()
        assert len(cache) == 0
        assert cache.nbytes == 0
