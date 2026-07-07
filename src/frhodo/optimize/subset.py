"""Informative-experiment subsetting for optimize-on-subset runs.

Greedy D-optimal selection over the screening capability footprints:
each added experiment maximizes the information-volume gain
log det(GᵀG) of the selected block. Optimization evaluates the k
selected shocks; the full campaign serves as the validation set for
the generalization gap.
"""
import numpy as np



def select_informative_subset(footprints, k) -> list[int]:
    """Indices of the k most jointly-informative experiments.

    Args:
        footprints: ``(n_experiments, n_reactions)`` capability matrix
            (rows: how strongly each experiment's observable responds
            per reaction).
        k: Subset size; clipped to the experiment count.

    Returns:
        Row indices in selection order (most informative first).
    """
    G = np.asarray(footprints, dtype=float)
    n = G.shape[0]
    k = int(min(k, n))
    if k <= 0:
        return []

    # Regularization keeps the determinant defined while the selected
    # block is rank-deficient (fewer rows than reactions).
    eps = 1e-9 * max(float(np.max(np.abs(G))) ** 2, 1e-300)
    d = G.shape[1]
    M = eps * np.eye(d)
    selected: list[int] = []
    remaining = set(range(n))
    for _ in range(k):
        best_idx = -1
        best_gain = -np.inf
        for i in remaining:
            g = G[i]
            # Rank-1 determinant update: det(M + ggᵀ) = det(M)(1 + gᵀM⁻¹g)
            gain = float(g @ np.linalg.solve(M, g))
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        selected.append(best_idx)
        remaining.discard(best_idx)
        M = M + np.outer(G[best_idx], G[best_idx])

    return selected


def subset_report(footprints, selected) -> dict:
    """Coverage summary: how much information volume the subset keeps."""
    G = np.asarray(footprints, dtype=float)
    d = G.shape[1]
    eps = 1e-9 * max(float(np.max(np.abs(G))) ** 2, 1e-300)

    def logdet(rows):
        M = eps * np.eye(d) + rows.T @ rows
        sign, value = np.linalg.slogdet(M)

        return float(value)

    full = logdet(G)
    sub = logdet(G[selected])
    report = {
        "n_full": int(G.shape[0]),
        "n_subset": len(selected),
        "logdet_full": full,
        "logdet_subset": sub,
        "information_retained": float(np.exp((sub - full) / d)),
    }

    return report
