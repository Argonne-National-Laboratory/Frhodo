# Incident-shock reactor — Jacobian and parameter gradient derivation

Analytical derivatives `∂f/∂y` and `∂f/∂p` used by sensitivity analyses
(adjoint and forward) for the Goldsmith / Speth incident-shock reactor.
Validated column-by-column against central finite differences in
[tests/engine/test_adjoint.py](../tests/engine/test_adjoint.py); these
tests are the audit trail for every formula in this document. The
analytical Jacobian is consumed by [frhodo/simulation/shock/adjoint.py](../frhodo/simulation/shock/adjoint.py)
as the user-supplied `jac` callback for the backward solve, and the
parameter RHS gradient feeds the adjoint quadrature.

---

## 1. State vector and RHS

The reactor integrates lab-frame time `t` over the state vector

```
y = [z, A, ρ, v, T, t_shock, Y_0, …, Y_{K-1}]      length n = 6 + K
```

Index abbreviations: `i_z=0, i_A=1, i_ρ=2, i_v=3, i_T=4, i_ts=5`,
species at `6..6+K-1`. `K = gas.n_species`.

The RHS, from [`_shock_derivatives`](../frhodo/simulation/shock/incident_shock_reactor.py),
factors into a geometric scale `s = ρA/(ρ_1 A_1)` times an unscaled
field `ġ`:

```
ġ_z   = v
ġ_A   = dA_dt                                       0 if area_change=False
ġ_ρ   = (1/(1+β)) · (φ − ρ β dA_dt / A)             chem + geom split below
ġ_v   = −v · (ġ_ρ/ρ + dA_dt/A)
ġ_T   = −(ψ + v · ġ_v) / cp
ġ_ts  = 1
ġ_{Y_i} = wdot_i W_i / ρ

f = s · ġ
```

with intermediates

```
β   = v² · (1/(cp T) − W/(R_u T))
φ_h = Σ_k h_k · wdot_k                              kinetic enthalpy term
φ_n = Σ_k wdot_k                                    net mole production
φ   = φ_h/(cp T) − W · φ_n                          species term in ġ_ρ
ψ   = (Σ_k wdot_k · h_k) / ρ = φ_h / ρ              heat-release term in ġ_T
```

`cp = gas.cp_mass`, `W = gas.mean_molecular_weight`,
`h_k = gas.partial_molar_enthalpies` (J/kmol),
`wdot = gas.net_production_rates` (kmol/m³/s),
`W_k = gas.molecular_weights`.

For the area-change Mirels formulation with exponent `n = 0.5` and
`ξ = max(z/L, 10⁻¹⁰)`:

```
dA_dt = v · As · n / L · ξ^{n-1} / (1−ξⁿ)²
∂(dA_dt)/∂v = dA_dt / v
∂(dA_dt)/∂z = (v As n / L²) · ξ^{n-2} · [(n−1)(1 − ξⁿ) + 2n ξⁿ] / (1 − ξⁿ)³
```

With `area_change=False` (the typical case), `dA_dt = 0` and all its
derivatives vanish.

## 2. Cantera derivative conventions and the "DP correction"

Cantera's kinetics derivatives are computed at specific fixed states.
Per the Cantera 3.x documentation:

| Cantera property | Variables held constant |
|---|---|
| `net_production_rates_ddT` | **pressure**, molar concentration, mole fractions |
| `net_production_rates_ddCi` | temperature, **pressure**, other species concentrations |
| `net_production_rates_ddP` | temperature, molar concentration, mole fractions |

The key observation is that **`ddT` and `ddCi` are at constant
pressure**, while the chain rule we need is at fixed `(ρ, Y)` for T
perturbations and fixed `(T, Y, other c_j)` for ρ perturbations —
neither of which holds pressure constant.

This was confirmed empirically by FD: at fixed `(T, Y)`, varying `ρ`
changes `wdot` more than `DC @ (Y/W)` alone predicts. The discrepancy
is exactly `DP · (R_u T / W)`, which is the contribution of the
pressure change implicit in the `ρ → ρ + Δρ` perturbation. The same
correction applies to `T` and `Y_i` perturbations.

The corrected chain-rule formulas — the ones the code actually uses —
are:

```
∂wdot/∂T |_{ρ, Y fixed}    = DT + DP · (ρ R_u / W)
∂wdot/∂ρ |_{T, Y fixed}    = DC @ (Y/W) + DP · (R_u T / W)
∂wdot_k/∂Y_j |_{T, ρ, other Y} = (DC[:, j] + DP · R_u T) · (ρ/W_j)
```

The DP term is required for falloff and three-body reactions where
the rate depends on the bath-gas concentration / pressure. Without
it, the Jacobian misses these contributions and the adjoint diverges
from native sensitivity by tens of percent on mechs with non-trivial
pressure-dependent kinetics.

This is not optional. **The DP correction is the difference between
"works on H2/O2" and "matches native on cycloheptane and GRI 3.0."**

## 3. Building blocks (per Jacobian call)

```
DC      = gas.net_production_rates_ddCi          shape (K, K)
DT      = gas.net_production_rates_ddT           shape (K,)
DP      = gas.net_production_rates_ddP           shape (K,)

cp_k_mole = gas.partial_molar_cp                 J/kmol/K
cp_k_mass = cp_k_mole / Wk                       J/kg/K, per-species
hk      = gas.partial_molar_enthalpies           J/kmol
wdot    = gas.net_production_rates               kmol/m³/s
cp      = gas.cp_mass                            J/kg/K
W       = gas.mean_molecular_weight              kg/kmol

Y_over_W   = Y / Wk
rho_over_W = ρ / Wk

DP_RuT  = DP · (R_u · T)

dwdot_dT   = DT + DP · (ρ R_u / W)                              shape (K,)
dwdot_drho = DC @ Y_over_W + DP · (R_u T / W)                   shape (K,)
dwdot_dY   = (DC + DP_RuT[:, None]) · rho_over_W[None, :]       shape (K, K)
```

`dcp_dT` is the one quantity Cantera does not expose directly; it is
computed by central-style finite difference on `cp_mass`:

```
ε_T   = T · 10⁻⁷
gas.TD = T + ε_T, ρ    →    cp_plus = gas.cp_mass
gas.TD = T, ρ          →    restored
dcp_dT = (cp_plus − cp) / ε_T
```

For ideal gas, ∂cp/∂Y_i = cp_mass_i — already available as
`cp_k_mass[i]`:

```
dcp_dY = cp_k_mass                shape (K,)
dW_dY  = −W² / Wk                 from W = 1/Σ(Y_k/W_k)
```

## 4. Decomposition of `J = ∂f/∂y` via the scale factor

Since `f = s · ġ`, the product rule gives

```
∂f_a/∂y_k = (∂s/∂y_k) · ġ_a  +  s · (∂ġ_a/∂y_k)
```

`∂s/∂y_k` is non-zero only for `k = i_ρ` (returns `A/(ρ_1 A_1)`) and
`k = i_A` (returns `ρ/(ρ_1 A_1)`). So:

```
∂f_a/∂y_k = s · (∂ġ_a/∂y_k)                                            k ∉ {i_A, i_ρ}
∂f_a/∂i_A = (ρ/(ρ_1 A_1)) · ġ_a  +  s · (∂ġ_a/∂i_A)
∂f_a/∂i_ρ = (A/(ρ_1 A_1)) · ġ_a  +  s · (∂ġ_a/∂i_ρ)
```

The rest of the derivation focuses on `∂ġ_a/∂y_k`; the scale-product
correction is applied as a final step in code.

## 5. Intermediate derivatives

Composition derivatives of the kinetic-enthalpy sum `φ_h = Σ h_k · wdot_k`
(using `∂h_k/∂T = cp_k_mole`, independent of composition for ideal gas):

```
∂φ_h/∂T   = Σ_k (cp_k_mole · wdot_k + h_k · dwdot_dT_k)
∂φ_h/∂ρ   = Σ_k h_k · dwdot_drho_k
∂φ_h/∂Y_i = Σ_k h_k · dwdot_dY_{k, i}                  shape (K,)
```

Mole-production sum `φ_n = Σ wdot_k`:

```
∂φ_n/∂T   = Σ_k dwdot_dT_k
∂φ_n/∂ρ   = Σ_k dwdot_drho_k
∂φ_n/∂Y_i = Σ_k dwdot_dY_{k, i}                        shape (K,)
```

Combined `φ = φ_h/(cp T) − W · φ_n`:

```
∂φ/∂T   = ∂φ_h/∂T / (cp T)  −  φ_h · (dcp_dT/cp + 1/T) / (cp T)  −  W · ∂φ_n/∂T
∂φ/∂ρ   = ∂φ_h/∂ρ / (cp T)  −  W · ∂φ_n/∂ρ
∂φ/∂Y_i = ∂φ_h/∂Y_i / (cp T) − φ_h · dcp_dY_i / (cp² T) − dW_dY_i · φ_n − W · ∂φ_n/∂Y_i
```

`β = v² · (1/(cp T) − W/(R_u T))`:

```
∂β/∂v   = 2 β / v
∂β/∂T   = −v² · (dcp_dT/(cp² T) + 1/(cp T²) − W/(R_u T²))
∂β/∂ρ   = 0
∂β/∂Y_i = v² · (−dcp_dY_i/(cp² T) + W²/(W_i R_u T))
```

Heat-release sum `ψ = φ_h / ρ` (note the difference from φ, which has
a `cp T` weighting):

```
∂ψ/∂T   = ∂φ_h/∂T / ρ
∂ψ/∂ρ   = ∂φ_h/∂ρ / ρ  −  φ_h / ρ²
∂ψ/∂Y_i = ∂φ_h/∂Y_i / ρ
∂ψ/∂v   = ∂ψ/∂A = ∂ψ/∂z = 0
```

## 6. The six shock-specific rows of `∂ġ/∂y`

### Row z (`ġ_z = v`):

```
∂ġ_z/∂v = 1;  others = 0.
```

### Row A (`ġ_A = dA_dt`):

For `area_change=False`: all zero.

For `area_change=True`:

```
∂ġ_A/∂v = dA_dt / v
∂ġ_A/∂z = (v As n / L²) · ξ^{n-2} · [(n−1)(1 − ξⁿ) + 2 n ξⁿ] / (1 − ξⁿ)³
```

### Row ρ (`ġ_ρ = ġ_ρ^chem + ġ_ρ^geom`):

The chemistry part `ġ_ρ^chem = φ/(1+β)`:

```
∂ġ_ρ^chem/∂y_k = (∂φ/∂y_k)/(1+β)  −  ġ_ρ^chem · (∂β/∂y_k)/(1+β)
```

For `area_change=False`, the geometric piece vanishes. For
`area_change=True`, define `μ = β/(1+β)` so `∂μ/∂x = ∂β/∂x / (1+β)²`,
then:

```
ġ_ρ^geom        = −ρ μ dA_dt / A

∂ġ_ρ^geom/∂z    = −ρ μ/A · ∂(dA_dt)/∂z
∂ġ_ρ^geom/∂A    = ρ μ dA_dt / A²
∂ġ_ρ^geom/∂ρ    = −μ dA_dt / A
∂ġ_ρ^geom/∂v    = −ρ dA_dt/A · ∂μ/∂v  −  ρ μ/A · (dA_dt / v)
∂ġ_ρ^geom/∂T    = −ρ dA_dt/A · ∂μ/∂T
∂ġ_ρ^geom/∂Y_i  = −ρ dA_dt/A · ∂μ/∂Y_i
```

The total row is the sum: `∂ġ_ρ/∂y_k = ∂ġ_ρ^chem/∂y_k + ∂ġ_ρ^geom/∂y_k`.

### Row v (`ġ_v = −v · Q` where `Q = ġ_ρ/ρ + dA_dt/A`):

```
∂Q/∂ρ   = ∂ġ_ρ/∂ρ / ρ  −  ġ_ρ / ρ²
∂Q/∂A   = ∂ġ_ρ/∂A / ρ  −  dA_dt / A²
∂Q/∂z   = ∂ġ_ρ/∂z / ρ  +  ∂(dA_dt)/∂z / A         (area_change=True only)
∂Q/∂v   = ∂ġ_ρ/∂v / ρ  +  ∂(dA_dt)/∂v / A         (area_change=True only)
∂Q/∂T   = ∂ġ_ρ/∂T / ρ
∂Q/∂Y_i = ∂ġ_ρ/∂Y_i / ρ

∂ġ_v/∂y_k = −v · ∂Q/∂y_k                          k ≠ i_v
∂ġ_v/∂v   = −Q  −  v · ∂Q/∂v
```

### Row T (`ġ_T = −(ψ + v · ġ_v) / cp`):

The compact form, derived by applying the quotient rule with respect
to the `1/cp` factor:

```
∂ġ_T/∂y_k = −(∂ψ/∂y_k + v · ∂ġ_v/∂y_k) / cp                       k ∉ {i_v, i_T, species}
∂ġ_T/∂v   = −(ġ_v + v · ∂ġ_v/∂v) / cp
∂ġ_T/∂T   = −∂ψ/∂T / cp  −  v · ∂ġ_v/∂T / cp  −  ġ_T · dcp_dT / cp
∂ġ_T/∂Y_i = −(∂ψ/∂Y_i + v · ∂ġ_v/∂Y_i) / cp  −  ġ_T · dcp_dY_i / cp
```

The `−ġ_T · dcp_dT/cp` and `−ġ_T · dcp_dY_i/cp` terms come from
differentiating the `1/cp` factor; expand to `+(ψ + v·ġ_v)·dcp/cp²`
to verify the sign. This sign was historically a bug in an earlier
draft of the plan — the code [adjoint.py:497-498](../frhodo/simulation/shock/adjoint.py#L497-L498) uses the correct
form.

### Row t_shock (`ġ_ts = 1`):

All zero.

## 7. Species block (`ġ_{Y_i} = wdot_i · W_i / ρ`)

```
∂ġ_{Y_i}/∂T   = dwdot_dT_i · W_i / ρ                                   uses corrected dwdot_dT
∂ġ_{Y_i}/∂ρ   = dwdot_drho_i · W_i / ρ  −  wdot_i · W_i / ρ²           uses corrected dwdot_drho
∂ġ_{Y_i}/∂Y_j = dwdot_dY_{i, j} · W_i / ρ                              uses corrected dwdot_dY
∂ġ_{Y_i}/∂v   = ∂ġ_{Y_i}/∂A = ∂ġ_{Y_i}/∂z = ∂ġ_{Y_i}/∂t_shock = 0
```

These are matrix scalings of the Cantera-derived blocks — no
per-species loop is needed; the entire species sub-block evaluates as
broadcasting in `numpy`.

## 8. Parameter RHS gradient `∂f/∂p_j`

`p_j` is the multiplier on reaction `j`'s rate constant (per
`gas.set_multiplier(value, i_reaction=j)`). Detailed balance is
preserved by Cantera — both forward and reverse rate constants scale
by `p_j` — so the net rate of progress `q_j` is linear in `p_j` at
`p = 1`:

```
∂wdot_k/∂p_j = ν_{k, j} · q_j
```

with `ν_{k, j} = product_stoich − reactant_stoich` (the net
stoichiometric matrix) and `q_j = gas.net_rates_of_progress[j]`.

Chain rule through the shock-specific rows (only via `wdot`, since
`cp`, `W`, `h_k` are not functions of `p`):

```
∂ġ_ρ/∂p_j  = (1/(1+β)) · Σ_k (h_k/(cp T) − W) · ν_{k, j} · q_j
∂ġ_v/∂p_j  = −v · ∂ġ_ρ/∂p_j / ρ
∂ġ_T/∂p_j  = −(Σ_k h_k · ν_{k, j} · q_j / ρ  +  v · ∂ġ_v/∂p_j) / cp
∂ġ_{Y_i}/∂p_j = ν_{i, j} · q_j · W_i / ρ
∂ġ_z/∂p_j  = ∂ġ_A/∂p_j  = ∂ġ_ts/∂p_j  = 0
```

The species-row sparsity is significant — for each reaction `j`, only
species in that reaction (typically 2–6) have non-zero
`∂ġ_{Y_i}/∂p_j`. The code exploits this by using `nu` as a CSR
matrix in the quadrature inner loop.

After scaling: `∂f_a/∂p_j = s · ∂ġ_a/∂p_j`.

## 9. drhodz observable: terminal condition and direct partial

`drhodz_tot = dρ/dz = (dρ/dt) / (dz/dt)`. In the scaled state-vector
representation:

```
drhodz = f_ρ / f_z = (s · ġ_ρ) / (s · v) = ġ_ρ / v
```

The scale factor cancels.

### Terminal condition for the adjoint

The functional `g(y(t_m)) = drhodz(y(t_m))` has gradient

```
∂(drhodz)/∂y_k = (1/f_z) · J[i_ρ, k]  −  (drhodz/f_z) · J[i_z, k]
```

— a linear combination of two rows of the already-cached J. No
separate drhodz-specific Jacobian work is required. The code computes
this at line [adjoint.py:154](../frhodo/simulation/shock/adjoint.py#L154):

```python
lam_T = (J_m[I_RHO] - drhodz_m * J_m[I_Z]) / f_z_m
```

### Direct partial

`drhodz` depends on `p_j` only through `ġ_ρ` (the `v` denominator
is `p`-independent):

```
∂(drhodz)/∂p_j |_y(t_m) = (1/v) · ∂ġ_ρ/∂p_j |_y(t_m)
                        = drhodz_per_rxn[j] from `drhodz_per_rxn_at_state`
```

— the existing helper in [reactor_output.py:103-141](../frhodo/simulation/shock/reactor_output.py)
already returns this exact quantity (when called with
`area_change_term=0`).

### Normalization

Cantera-style normalized sensitivities `∂ln(drhodz)/∂ln(k_j)` are
recovered by dividing the un-normalized result by `drhodz(t_m)`:

```
sens_normalized[m, j] = (quadrature[m, j] + drhodz_per_rxn[m, j]) / drhodz[m]
```

This matches the convention `_drhodz_chain_rule` returns in the
native path ([sensitivity.py:302](../frhodo/simulation/shock/sensitivity.py#L302)).

## 10. References

- Cao, Y., Li, S., Petzold, L., Serban, R. *Adjoint sensitivity
  analysis for differential-algebraic equations: The adjoint DAE
  system and its numerical solution.* SIAM J. Sci. Comput. 24:
  1076–1089 (2003).
- Hindmarsh, A. C., et al. *SUNDIALS: Suite of Nonlinear and
  Differential/Algebraic Equation Solvers.* ACM TOMS 31: 363–396
  (2005).
- Cantera kinetics derivatives:
  <https://cantera.org/3.2/python/kinetics.html#cantera.Kinetics.net_production_rates_ddCi>
- Goldsmith / Speth incident-shock RHS — derived in
  [incident_shock_reactor.py](../frhodo/simulation/shock/incident_shock_reactor.py).
- Sign convention for the adjoint integral (positive with
  `dλ/dt = −Jᵀλ`): Cao et al. 2003 eq. 2.8.
