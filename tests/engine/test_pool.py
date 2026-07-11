"""Persistent worker-pool reuse semantics."""
import copy

import numpy as np
import pytest

from frhodo.optimize._worker_context import MechBuildPayload
from frhodo.optimize.pool import PersistentWorkerPool, _payload_fingerprint


def _payload(scale=1.0):
    payload = MechBuildPayload(
        reset_mech=[{"rxnType": "Arrhenius", "A": 1.0e10 * scale}],
        thermo_coeffs=[np.arange(3.0)],
        coeffs=[{"pre_exponential_factor": 1.0e10 * scale}],
        coeffs_bnds=[{}],
        rate_bnds=[{}],
    )

    return payload


class TestPayloadFingerprint:
    def test_equal_content_equal_fingerprint(self):
        assert _payload_fingerprint(_payload()) == _payload_fingerprint(
            _payload())

    def test_different_content_different_fingerprint(self):
        assert _payload_fingerprint(_payload()) != _payload_fingerprint(
            _payload(scale=2.0))


class TestAcquireTiers:
    def _manager_with_fake_pool(self):
        mgr = PersistentWorkerPool()
        mgr._pool = object()
        mgr._size = 4
        mgr._payload_hash = _payload_fingerprint(_payload())

        return mgr

    def test_reuses_untouched_on_identical_payload_content(self):
        """A structurally-identical payload (e.g. a deterministic recast
        that rebuilt the Solution object) neither respawns nor
        re-initializes."""
        mgr = self._manager_with_fake_pool()
        calls = []
        mgr._reinit_workers = lambda payload: calls.append("reinit")
        mgr._respawn = lambda *a: calls.append("respawn")
        pool = mgr.acquire(workers=4, payload=_payload())
        assert pool is mgr._pool
        assert calls == [], f"identical payload triggered {calls}"

    def test_reinits_in_place_on_changed_payload_content(self):
        mgr = self._manager_with_fake_pool()
        calls = []
        mgr._reinit_workers = lambda payload: calls.append("reinit")
        mgr._respawn = lambda *a: calls.append("respawn")
        mgr.acquire(workers=4, payload=_payload(scale=2.0))
        assert calls == ["reinit"], (
            f"changed payload must re-init in place, got {calls}"
        )
        assert mgr._payload_hash == _payload_fingerprint(_payload(scale=2.0))

    def test_respawns_on_reinit_failure(self):
        mgr = self._manager_with_fake_pool()

        def boom(payload):
            raise RuntimeError("worker died")

        calls = []
        mgr._reinit_workers = boom
        mgr._respawn = lambda *a: calls.append("respawn")
        mgr.acquire(workers=4, payload=_payload(scale=2.0))
        assert calls == ["respawn"]

    def test_respawns_when_more_workers_requested(self):
        mgr = self._manager_with_fake_pool()
        assert mgr._needs_respawn(8)
        assert not mgr._needs_respawn(4)


@pytest.mark.slow
class TestRealPoolReinit:
    def test_reinit_updates_every_worker_mechanism(self, loaded_cycloheptane):
        """After an in-place re-init with modified coefficients, every
        worker must see the new value (no silently stale workers)."""
        from frhodo.optimize._worker_context import MechBuildPayload
        from frhodo.optimize.cost.fit_fcn import _pool_worker_probe

        mech = loaded_cycloheptane
        rxn_idx = next(
            i for i, entry in enumerate(mech.coeffs)
            if not isinstance(entry, dict) and len(entry) == 1
            and "pre_exponential_factor" in entry[0]
        )

        def payload():
            built = MechBuildPayload(
                reset_mech=copy.deepcopy(mech.reset_mech),
                thermo_coeffs=mech.thermo_coeffs,
                coeffs=copy.deepcopy(mech.coeffs),
                coeffs_bnds=mech.coeffs_bnds,
                rate_bnds=mech.rate_bnds,
            )

            return built

        base = payload()
        changed = payload()
        changed.coeffs[rxn_idx][0]["pre_exponential_factor"] *= 2.0

        mgr = PersistentWorkerPool()
        try:
            pool_a = mgr.acquire(workers=2, payload=base)
            probes = pool_a.map(
                _pool_worker_probe,
                [(rxn_idx, None, "pre_exponential_factor")] * 2,
                chunksize=1,
            )
            baseline = base.coeffs[rxn_idx][0]["pre_exponential_factor"]
            assert all(p == pytest.approx(baseline) for p in probes)

            pool_b = mgr.acquire(workers=2, payload=changed)
            assert pool_b is pool_a, "changed payload must not respawn"
            probes = pool_b.map(
                _pool_worker_probe,
                [(rxn_idx, None, "pre_exponential_factor")] * 2,
                chunksize=1,
            )
            expected = changed.coeffs[rxn_idx][0]["pre_exponential_factor"]
            assert all(p == pytest.approx(expected) for p in probes), (
                f"stale worker mechanism after re-init: {probes} "
                f"vs expected {expected}"
            )
        finally:
            mgr.close()
