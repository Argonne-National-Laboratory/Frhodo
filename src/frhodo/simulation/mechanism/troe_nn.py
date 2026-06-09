"""Troe inverse-function neural net: 9 rates -> K candidate Troe parameter sets.

Pure-numpy inference. Weights load from an ``.npz`` checkpoint produced by
the slave-side training script in ``development/troe_fit/train_nn.py``.

Each call returns four arrays per example:
  preds_norm:    (B, K, 10) normalized predictions, bounded by tanh/sigmoid
  log_sigmas:    (B, K, 10) heteroscedastic log-σ per parameter
  predicted_rms: (B, K) per-candidate predicted post-fit log-RMS, > 0
  conf_logits:   (B, K) candidate-selection logits
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import expit


K_CANDIDATES = 20
EMBED_DIM = 128
DECODER_HIDDEN = 256
DECODER_PER_CAND = 21       # 10 means + 10 log-σ + 1 predicted-rms

# Output-mapping scales: ``normalized_to_capture_np`` maps the NN's
# tanh/sigmoid output to physical-Troe units. These ranges are sized to
# cover the empirical label distribution from build_synth_v3 + synth_v2.
_EA_SCALE = 3e9            # Ea_0, Ea_inf
_LNA_SCALE = 100.0          # ln A_0, ln A_inf
_N_SCALE = 50.0             # n_0, n_inf
_T3_T1_LOG_RANGE = 15.0     # T3, T1 in [1e-15, 1e15]
_T2_LOG_RANGE = 5.0         # T2 in [-1e5, 1e5] (bisymlog-shaped)

CHECKPOINT_PATH = Path(__file__).with_name("troe_nn_checkpoint.npz")


def _softplus(x):
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))


def _linear(x, weight, bias):
    return x @ weight.T + bias


def _relu(x):
    return np.maximum(x, 0.0)


def raw_to_normalized_np(raw):
    """Bound a raw 10-vec to [-1, 1].

    Slot 6 (A_Fc) uses sigmoid → (0, 1); other slots use tanh → [-1, 1].
    """
    out = np.empty_like(raw)
    out[..., 0] = np.tanh(raw[..., 0])
    out[..., 1] = np.tanh(raw[..., 1])
    out[..., 2] = np.tanh(raw[..., 2])
    out[..., 3] = np.tanh(raw[..., 3])
    out[..., 4] = np.tanh(raw[..., 4])
    out[..., 5] = np.tanh(raw[..., 5])
    out[..., 6] = expit(raw[..., 6])
    out[..., 7] = np.tanh(raw[..., 7])
    out[..., 8] = np.tanh(raw[..., 8])
    out[..., 9] = np.tanh(raw[..., 9])

    return out


def normalized_to_capture_np(norm):
    """Map normalized predictions to capture-form physical Troe params.

    Layout: ``[Ea_0, A_0, n_0, Ea_inf, A_inf, n_inf, A_Fc, T3, T1, T2]``.
    Slots 1, 4 are linear ``A``; slots 0, 3 are ``Ea`` in J/kmol.
    """
    out = np.empty_like(norm)
    out[..., 0] = norm[..., 0] * _EA_SCALE
    out[..., 1] = np.exp(norm[..., 1] * _LNA_SCALE)
    out[..., 2] = norm[..., 2] * _N_SCALE
    out[..., 3] = norm[..., 3] * _EA_SCALE
    out[..., 4] = np.exp(norm[..., 4] * _LNA_SCALE)
    out[..., 5] = norm[..., 5] * _N_SCALE
    out[..., 6] = norm[..., 6]
    out[..., 7] = 10.0 ** (norm[..., 7] * _T3_T1_LOG_RANGE)
    out[..., 8] = 10.0 ** (norm[..., 8] * _T3_T1_LOG_RANGE)
    out[..., 9] = np.sign(norm[..., 9]) * (
        10.0 ** (np.abs(norm[..., 9]) * _T2_LOG_RANGE) - 1.0
    )

    return out


def capture_to_normalized_np(capture):
    """Inverse of :func:`normalized_to_capture_np`.

    Slot 6 (A_Fc) is clipped to a safe ``logit`` range; the linear-A slots
    are passed through ``log`` first. Returns the **normalized** ``[-1, 1]``
    representation, not the unbounded ``raw`` pre-activation.
    """
    norm = np.empty_like(capture, dtype=np.float64)
    norm[..., 0] = capture[..., 0] / _EA_SCALE
    norm[..., 1] = np.log(np.maximum(capture[..., 1], 1e-300)) / _LNA_SCALE
    norm[..., 2] = capture[..., 2] / _N_SCALE
    norm[..., 3] = capture[..., 3] / _EA_SCALE
    norm[..., 4] = np.log(np.maximum(capture[..., 4], 1e-300)) / _LNA_SCALE
    norm[..., 5] = capture[..., 5] / _N_SCALE
    norm[..., 6] = capture[..., 6]
    norm[..., 7] = np.log10(np.maximum(capture[..., 7], 1e-300)) / _T3_T1_LOG_RANGE
    norm[..., 8] = np.log10(np.maximum(capture[..., 8], 1e-300)) / _T3_T1_LOG_RANGE
    T2 = capture[..., 9]
    norm[..., 9] = np.sign(T2) * np.log10(np.abs(T2) + 1.0) / _T2_LOG_RANGE

    return norm


class _Net:
    """Pure-numpy inference of ``TroeInverseKNet``.

    Layer naming matches the torch ``state_dict`` produced by
    ``development/troe_fit/train_nn.py`` so the checkpoint round-trips
    without renaming.
    """

    def __init__(self, weights: dict[str, np.ndarray]):
        self._w = weights

    def forward(self, points: np.ndarray, mask: np.ndarray):
        """Args:
            points: ``(B, n_pts, 3)`` already-normalized feature tensor.
            mask:   ``(B, n_pts)`` boolean mask of valid points.

        Returns ``(preds_norm, log_sigmas, predicted_rms, conf_logits)``
        with shapes ``(B, K, 10)``, ``(B, K, 10)``, ``(B, K)``, ``(B, K)``.
        """
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points must be (B, n_pts, 3); got {points.shape}")
        if mask.shape != points.shape[:-1]:
            raise ValueError(
                f"mask {mask.shape} must align with points {points.shape[:-1]}"
            )

        w = self._w
        x = _relu(_linear(points, w["point_encoder.0.weight"],
                          w["point_encoder.0.bias"]))
        x = _relu(_linear(x, w["point_encoder.3.weight"],
                          w["point_encoder.3.bias"]))
        embeddings = _linear(x, w["point_encoder.6.weight"],
                             w["point_encoder.6.bias"])

        mask_f = mask.astype(embeddings.dtype)[..., None]
        e_masked = embeddings * mask_f
        count = np.maximum(mask_f.sum(axis=1), 1.0)
        mean_p = e_masked.sum(axis=1) / count
        e_for_max = np.where(mask[..., None], e_masked, -1e9)
        max_p = e_for_max.max(axis=1)
        var_p = np.maximum((e_masked ** 2).sum(axis=1) / count - mean_p ** 2, 1e-8)
        std_p = np.sqrt(var_p)
        pooled = np.concatenate([mean_p, max_p, std_p], axis=-1)

        d = _relu(_linear(pooled, w["decoder.0.weight"], w["decoder.0.bias"]))
        d = _relu(_linear(d, w["decoder.3.weight"], w["decoder.3.bias"]))
        raw = _linear(d, w["decoder.6.weight"], w["decoder.6.bias"])
        B = pooled.shape[0]
        raw = raw.reshape(B, K_CANDIDATES, DECODER_PER_CAND)
        means_raw = raw[..., :10]
        log_sigmas = raw[..., 10:20]
        predicted_rms = _softplus(raw[..., 20])
        preds_norm = raw_to_normalized_np(means_raw)

        c = _relu(_linear(pooled, w["confidence_head.0.weight"],
                          w["confidence_head.0.bias"]))
        conf_logits = _linear(c, w["confidence_head.3.weight"],
                              w["confidence_head.3.bias"])

        return preds_norm, log_sigmas, predicted_rms, conf_logits


_STATS_KEYS = ("logT_mean", "logT_std", "logM_mean", "logM_std",
               "lnk_mean", "lnk_std")
_MODEL: _Net | None = None
_STATS: dict | None = None


def get_model() -> tuple[_Net, dict]:
    """Return ``(model, stats)`` from the bundled ``.npz`` checkpoint.

    Lazy-loaded; subsequent calls return the cached objects.
    """
    global _MODEL, _STATS
    if _MODEL is None:
        data = np.load(CHECKPOINT_PATH, allow_pickle=False)
        weights = {
            k[len("weights."):]: data[k]
            for k in data.files if k.startswith("weights.")
        }
        stats = {k: float(data[f"stats.{k}"]) for k in _STATS_KEYS}
        _MODEL = _Net(weights)
        _STATS = stats

    return _MODEL, _STATS


def reset_model_cache() -> None:
    """Test helper: drop the cached model + stats so the next call reloads."""
    global _MODEL, _STATS
    _MODEL = None
    _STATS = None
