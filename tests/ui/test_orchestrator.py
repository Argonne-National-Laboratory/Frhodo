"""``Multithread_Optimize`` (orchestrator) construction + early-return guards.

These tests pin what PR 23b's deeper decoupling must preserve: the
optimizer never starts when the directory is invalid, when an opt run
is already flagged, or when no reactions/coefs are marked optimizable.
"""
import pytest

from scipy import stats


pytestmark = pytest.mark.gui


class TestOrchestratorWiring:
    def test_attached_to_main(self, main_window):
        assert hasattr(main_window, "optimize")
        assert main_window.optimize.parent is main_window

    def test_dist_is_gennorm(self, main_window):
        assert main_window.optimize.dist is stats.gennorm

    def test_initial_flags(self, main_window):
        assert main_window.run_control.optimize_running is False
        # multiprocessing toggle defaults to True (matches GUI checkbox)
        assert main_window.run_control.multiprocessing is True

    def test_action_run_connected(self, main_window):
        """The Run toolbar action triggers the orchestrator. Qt does not
        let us read connected slots back; we assert receiver count > 0."""
        assert main_window.action_Run.receivers(
            main_window.action_Run.triggered
        ) >= 1

    def test_action_abort_connected(self, main_window):
        assert main_window.action_Abort.receivers(
            main_window.action_Abort.triggered
        ) >= 1


class TestStartThreadsGuards:
    """``start_threads`` returns early without spawning a worker when the
    pre-conditions don't hold. Reachable only after ``parent.path["mech"]``
    is set (Cycloheptane fixture)."""

    def _captured_log(self, main_window):
        captured: list[str] = []
        original = main_window.log.append

        def capture(msg, **kwargs):
            captured.append(str(msg))

            return original(msg, **kwargs)

        main_window.log.append = capture

        return captured

    def test_invalid_directory_aborts(self, main_with_loaded_mech):
        main = main_with_loaded_mech
        captured = self._captured_log(main)
        main.directory.invalid = ["exp_main"]

        main.optimize.start_threads()

        assert any("Invalid directory" in m for m in captured), (
            f"expected 'Invalid directory' message; got {captured}"
        )
        assert main.run_control.optimize_running is False

    def test_already_running_aborts(self, main_with_loaded_mech):
        main = main_with_loaded_mech
        captured = self._captured_log(main)
        main.directory.invalid = []
        main.run_control.optimize_running = True
        try:
            main.optimize.start_threads()
        finally:
            main.run_control.optimize_running = False

        assert any("already set to True" in m for m in captured), (
            f"expected 'already set to True' message; got {captured}"
        )

    def test_no_optimizable_reactions_aborts(self, main_with_loaded_mech):
        main = main_with_loaded_mech
        captured = self._captured_log(main)
        main.directory.invalid = []
        main.run_control.optimize_running = False

        main.optimize.start_threads()

        assert any("No reactions" in m for m in captured), (
            f"expected 'No reactions' message; got {captured}"
        )
        assert main.run_control.optimize_running is False


class TestHasOptRxns:
    def test_default_is_false(self, main_with_loaded_mech):
        """A freshly-loaded mech has nothing toggled in the builder."""
        main = main_with_loaded_mech
        assert main.optimizables.build(main.mech).is_empty()

    def test_true_when_rxn_and_coef_opted_in(self, main_with_loaded_mech):
        main = main_with_loaded_mech
        bnds_key = next(iter(main.mech.coeffs_bnds[0]))
        coef_name = next(iter(main.mech.coeffs_bnds[0][bnds_key]))
        main.optimizables.set_reaction_optimizable(0, True)
        main.optimizables.set_coefficient_optimizable(0, bnds_key, coef_name, True)
        try:
            assert not main.optimizables.build(main.mech).is_empty()
        finally:
            main.optimizables.reset()
