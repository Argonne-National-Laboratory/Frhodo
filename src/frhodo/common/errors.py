"""Typed errors for engine failure modes.

The integrator writes ``_failure_reason`` to a side-channel attribute
on the reactor object before raising; the catch site reads it back and
wraps in ``IntegrationError``. Cantera's C boundary does not preserve
custom Python exception classes, so the side channel is required.
"""
import enum


class FailureReason(str, enum.Enum):
    """Categorical reason an integrator or jump solver gave up.

    String-backed so error reasons survive serialization and remain
    comparable across process boundaries.
    """
    TEMPERATURE_INVALID = "temperature_invalid"
    PRESSURE_INVALID = "pressure_invalid"
    DENSITY_INVALID = "density_invalid"
    INPUT_INVALID = "input_invalid"
    SOLVER_FAILURE = "solver_failure"
    PERFECT_GAS_NOT_CONVERGED = "perfect_gas_not_converged"
    FROSH_NOT_CONVERGED = "frosh_not_converged"


class IntegrationError(RuntimeError):
    """Reactor ODE integration failed with a typed reason.

    Attributes:
        reason: The :class:`FailureReason` set by the reactor's
            side-channel before raising.
    """

    def __init__(self, reason: FailureReason, message: str):
        super().__init__(message)
        self.reason = reason


class ShockJumpError(RuntimeError):
    """Raised when the normal-shock jump solver cannot produce zone-2/zone-5 state.

    Attributes:
        reason: The :class:`FailureReason` describing where the solve
            failed (input validation, perfect-gas warm-start, or the
            Frosh root-find).
    """

    def __init__(self, reason: FailureReason, message: str):
        super().__init__(message)
        self.reason = reason


class MechanismLoadError(RuntimeError):
    """Raised when ``MechanismLoader`` cannot produce a valid Solution."""


class SchemaVersionError(RuntimeError):
    """Raised when a persisted config has an unsupported ``schema_version``."""
