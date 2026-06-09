"""``ChemicalMechanism.exclusive()`` serializes cross-thread entry.

The context manager wraps a re-entrant lock: the same thread may
acquire it nested; a second thread blocks until the holder releases.
"""
import threading
import time

import cantera as ct
import pytest

from frhodo.simulation.mechanism import ChemicalMechanism


def _make_mech():
    mech = ChemicalMechanism()
    mech.gas = ct.Solution("h2o2.yaml")
    mech.set_rate_expression_coeffs()
    mech.set_thermo_expression_coeffs()
    mech.isLoaded = True

    return mech


class TestExclusive:
    def test_lock_is_initialized(self):
        mech = _make_mech()
        assert mech._lock is not None

    def test_reentrant_on_same_thread(self):
        mech = _make_mech()
        with mech.exclusive():
            with mech.exclusive():
                pass

    def test_serial_calls_release(self):
        mech = _make_mech()
        for _ in range(3):
            with mech.exclusive():
                pass

    def test_second_thread_blocks_until_release(self):
        """A second thread entering ``exclusive()`` while another holds it
        waits rather than raising or running concurrently."""
        mech = _make_mech()
        events = []
        holding = threading.Event()
        release = threading.Event()

        def hold_lock():
            with mech.exclusive():
                events.append("holder_in")
                holding.set()
                release.wait(timeout=2.0)
                events.append("holder_out")

        def try_acquire():
            holding.wait(timeout=2.0)
            events.append("contender_waiting")
            with mech.exclusive():
                events.append("contender_in")

        holder = threading.Thread(target=hold_lock)
        contender = threading.Thread(target=try_acquire)
        holder.start()
        contender.start()
        time.sleep(0.05)
        release.set()
        holder.join(timeout=3.0)
        contender.join(timeout=3.0)

        assert events.index("holder_out") < events.index("contender_in"), (
            f"contender entered before holder released: {events}"
        )
