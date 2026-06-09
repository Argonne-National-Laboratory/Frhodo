"""``OptimizableSet`` / ``OptimizableCoefficient`` data-model tests.

The selection of which coefficients to optimize is built by
``OptimizableSetBuilder.build`` (see ``test_optimizable_builder.py``).
This file pins the immutable model's invariants.
"""
import pytest

from frhodo.optimize.parameters import (
    OptimizableCoefficient,
    OptimizableSet,
)


class TestOptimizableSetModel:
    def test_default_is_empty(self):
        opt = OptimizableSet()
        assert opt.is_empty()
        assert opt.optimizable_reactions == ()
        assert opt.coefficients == ()

    def test_frozen(self):
        c = OptimizableCoefficient(
            rxn_idx=0, coef_name="A", coef_idx=0,
            coeffs_key=0, bnds_key="rate",
        )
        with pytest.raises(Exception):
            c.rxn_idx = 1

    def test_extra_keys_rejected(self):
        with pytest.raises(Exception):
            OptimizableCoefficient(
                rxn_idx=0, coef_name="A", coef_idx=0,
                coeffs_key=0, bnds_key="rate",
                surprise="boom",
            )

    def test_negative_rxn_idx_rejected(self):
        with pytest.raises(Exception):
            OptimizableCoefficient(
                rxn_idx=-1, coef_name="A", coef_idx=0,
                coeffs_key=0, bnds_key="rate",
            )
