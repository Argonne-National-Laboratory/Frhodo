# Vendored OpenDSM modules

Upstream: https://github.com/opendsm/opendsm
Commit: 023d72cb4c2c9a464cd0101f4eab68759f15f425 (main, vendored 2026-06-10)
License: Apache-2.0 (headers retained in each file)

## File map

| Vendored file | Upstream path |
| --- | --- |
| `adaptive_loss.py` | `opendsm/common/stats/adaptive_loss.py` |
| `adaptive_loss_Z.py` | `opendsm/common/stats/adaptive_loss_Z.py` |
| `outliers.py` | `opendsm/common/stats/outliers.py` |
| `stats_basic.py` | `opendsm/common/stats/basic.py` |
| `bisymlog.py` | `opendsm/common/stats/distribution_transform/bisymlog.py` |
| `utils.py` | `opendsm/common/utils.py` |

## Local modifications

Every file: imports rewritten `opendsm.common.*` → `frhodo._vendor.opendsm.*`.

- `adaptive_loss.py / get_C`: added `weights=` (per-observation weights threaded
  through the IQR / MAD / stdev scale estimators, numba-safe Optional
  narrowing); default `algo="mad"` (upstream `"iqr_legacy"`).
- `adaptive_loss.py / adaptive_weights`: added `weights=` (weighted median for
  mu, weights into outlier rejection / scale / alpha estimation) and
  `C_scalar=` (post-multiplier on C; 1 / outlier-rejection sensitivity);
  default `C_algo="mad"`.
- `bisymlog.py`: trimmed to the two numba kernels; the `Bisymlog` class and its
  `TransformBase` / `mu_sigma` / scipy dependencies are not vendored.
- `utils.py`: trimmed to `to_np_array` and `OoM_numba`.

## Not re-applied from the previous vendoring

`_fast_weighted_alpha` and `kernel_adaptive_weights` (Frhodo-side additions
with no remaining consumers; upstream's `KernelWeightCache` supersedes them).

## Re-vendoring procedure

Clone upstream, copy the files per the map, rewrite imports, re-apply the
modifications above, update this file's commit SHA, run
`tests/engine/test_adaptive_loss.py` and the engine suite.
