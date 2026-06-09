"""Per-reaction uncertainty + optimizable-state snapshot for mech reloads.

When the user loads a new mechanism the reaction-tree's box state is
normally lost — every coefficient's uncertainty value/type drops back
to its default and every "optimizable" flag clears. This module
snapshots and restores per-reaction state, keyed by reaction signature
(reactants, products, reversibility) so matching reactions across the
reload are recognized even when their index changes.

Duplicate reactions (multiple rxns sharing the same signature) are
stacked in iteration order. On restore, prior states are consumed in
the same order, but every duplicate-bearing rxn is flagged as
partial-match so the user verifies which duplicate received which
state — order matching is brittle once the file has been re-saved.

Three outcomes per reaction:

* No matching signature in the snapshot — leave defaults.
* Signature matches and every prior coefficient state transfers
  cleanly — full restore (rate + coefs + optimizable flags).
* Signature matches but at least one prior coefficient state can't be
  transferred (different rate type, value mismatch, missing coef slot)
  *or* the signature has duplicates on either side of the reload —
  restore what we can; caller flags the reaction so the GUI can
  paint it for review.
"""
from copy import deepcopy

import numpy as np


def rxn_signature(rxn):
    """Hashable identity tuple for one Cantera ``Reaction``.

    Built from reactants, products, and reversibility — coefficient
    values are excluded so a rxn matches across rate-type changes
    (e.g. Arrhenius → Plog with the same stoichiometry).
    """
    reactants = tuple(sorted(rxn.reactants.items()))
    products = tuple(sorted(rxn.products.items()))

    return (reactants, products, bool(rxn.reversible))


def signatures_for_gas(gas):
    """Per-reaction signatures for a Cantera gas.

    Returns a list of length ``gas.n_reactions``. Entries may repeat
    when the mechanism contains duplicate reactions.
    """
    return [rxn_signature(gas.reaction(i)) for i in range(gas.n_reactions)]


def coeffs_equal(a, b, rtol=1e-9, atol=0.0):
    """Deep numerical equality across the nested rxn-coef structures
    Frhodo stores in ``mech.coeffs[i]``.

    Handles dicts, lists, numpy arrays, and scalar floats (including
    NaN-equal-NaN). Returns False on the first mismatch.
    """
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return False

        return all(coeffs_equal(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False

        return all(coeffs_equal(ai, bi, rtol, atol) for ai, bi in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if isinstance(a, (int, float, np.floating)):
        if not isinstance(b, (int, float, np.floating)):
            return False
        if (
            isinstance(a, float)
            and isinstance(b, float)
            and np.isnan(a)
            and np.isnan(b)
        ):
            return True

        return abs(a - b) <= max(atol, rtol * max(abs(float(a)), abs(float(b)), 1.0))

    return a == b


def _has_user_state(coef_state):
    """True iff the user set non-default unc value/type or flagged the
    coef as optimizable. The default state ("F" type, NaN value, not
    optimizable) means the user never touched this box, so a mismatch
    on this coef isn't visible to them and doesn't merit a partial flag.
    """
    if coef_state["optimizable"]:
        return True
    if coef_state["type"] != "F":
        return True
    value = coef_state["value"]
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False

    return True


def _capture_one(mech, optimizables, rxnIdx):
    coef_state = {}
    for bnds_key, sub in mech.coeffs_bnds[rxnIdx].items():
        for coef_name, coef_dict in sub.items():
            coef_state[(bnds_key, coef_name)] = {
                "value": coef_dict["value"],
                "type": coef_dict["type"],
                "resetVal": coef_dict["resetVal"],
                "optimizable": optimizables.is_coefficient_optimizable(
                    rxnIdx, bnds_key, coef_name,
                ),
            }

    return {
        "coeffs": deepcopy(mech.coeffs[rxnIdx]),
        "rate_unc": {
            "value": mech.rate_bnds[rxnIdx]["value"],
            "type": mech.rate_bnds[rxnIdx]["type"],
        },
        "rate_optimizable": optimizables.is_reaction_optimizable(rxnIdx),
        "coef_state": coef_state,
    }


def capture_state(mech, optimizables):
    """Snapshot per-rxn engine state from a loaded mech.

    Returns a ``dict`` keyed by :func:`rxn_signature`. The value is a
    ``list[state]`` — one entry per matching rxn in iteration order.
    Most signatures map to a one-element list; duplicate reactions
    produce multi-element lists.
    """
    snapshot: dict[tuple, list[dict]] = {}
    for i in range(mech.gas.n_reactions):
        sig = rxn_signature(mech.gas.reaction(i))
        snapshot.setdefault(sig, []).append(_capture_one(mech, optimizables, i))

    return snapshot


def restore_state(mech, optimizables, snapshot):
    """Apply a prior :func:`capture_state` snapshot to a freshly-loaded mech.

    For each new rxn, looks up the matching prior signature. Prior
    states stored under one signature are consumed in iteration order
    — the N-th occurrence of a sig in the new mech consumes the N-th
    state in the snapshot's list.

    * Rate-level uncertainty + reaction-optimizable flag always restore.
    * Per-coef state restores only when the slot exists in the new
      mech **and** the new coefficient value matches the prior value
      numerically.
    * A coefficient-state mismatch on a user-touched slot flags the
      reaction as partial-match.
    * A signature with duplicate occurrences (on either side of the
      reload) flags every restored rxn under it as partial-match
      regardless of coefficient match — order-based correlation is
      brittle once the file has been re-saved, so the user verifies.

    Args:
        mech: Freshly-loaded ``ChemicalMechanism``.
        optimizables: ``OptimizableSetBuilder`` to receive restored
            flags. Pre-reset is the caller's responsibility.
        snapshot: Output of :func:`capture_state` (may be ``None`` or
            empty; both are handled).

    Returns:
        ``(restored, partial)`` — ``restored`` is the set of rxn
        indices that found a matching prior signature. ``partial`` is
        the subset that had a coef mismatch or a duplicate-bearing
        signature.
    """
    restored: set = set()
    partial: set = set()
    if not snapshot:
        return restored, partial

    new_sig_counts: dict[tuple, int] = {}
    for i in range(mech.gas.n_reactions):
        sig = rxn_signature(mech.gas.reaction(i))
        new_sig_counts[sig] = new_sig_counts.get(sig, 0) + 1

    cursor: dict[tuple, int] = {}
    for i in range(mech.gas.n_reactions):
        sig = rxn_signature(mech.gas.reaction(i))
        prior_list = snapshot.get(sig)
        if not prior_list:
            continue
        idx = cursor.get(sig, 0)
        if idx >= len(prior_list):
            continue
        cursor[sig] = idx + 1
        prior = prior_list[idx]

        restored.add(i)
        mech.rate_bnds[i]["value"] = prior["rate_unc"]["value"]
        mech.rate_bnds[i]["type"] = prior["rate_unc"]["type"]
        optimizables.set_reaction_optimizable(i, prior["rate_optimizable"])

        new_coef_keys = {
            (bnds_key, coef_name)
            for bnds_key, sub in mech.coeffs_bnds[i].items()
            for coef_name in sub
        }

        any_orphan = False
        for old_key, old in prior["coef_state"].items():
            if old_key in new_coef_keys:
                continue
            if _has_user_state(old):
                any_orphan = True
                break

        any_mismatch = False
        for bnds_key, sub in mech.coeffs_bnds[i].items():
            for coef_name, coef_dict in sub.items():
                key = (bnds_key, coef_name)
                old = prior["coef_state"].get(key)
                if old is None:
                    continue

                if not coeffs_equal(old["resetVal"], coef_dict["resetVal"]):
                    if _has_user_state(old):
                        any_mismatch = True
                    continue

                coef_dict["value"] = old["value"]
                coef_dict["type"] = old["type"]
                optimizables.set_coefficient_optimizable(
                    i, bnds_key, coef_name, old["optimizable"],
                )

        is_duplicate = len(prior_list) > 1 or new_sig_counts[sig] > 1
        if is_duplicate and prior["rate_optimizable"]:
            partial.add(i)
        elif (any_orphan or any_mismatch) and prior["rate_optimizable"]:
            partial.add(i)

    return restored, partial
