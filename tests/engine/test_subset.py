"""D-optimal experiment subsetting."""
import numpy as np

from frhodo.optimize.subset import select_informative_subset, subset_report



class TestSelectInformativeSubset:
    def test_orthogonal_rows_selected_before_duplicates(self):
        G = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        picked = select_informative_subset(G, 3)
        assert set(picked) == {0, 2, 3} or set(picked) == {1, 2, 3}, (
            f"three orthogonal directions available, picked rows {picked}"
        )

    def test_larger_response_wins_within_a_direction(self):
        G = np.array([
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
        ])
        picked = select_informative_subset(G, 2)
        assert 1 in picked and 2 in picked, (
            f"expected the strong row 1 and orthogonal row 2, got {picked}"
        )

    def test_k_clipped_to_experiment_count(self):
        G = np.eye(2)
        picked = select_informative_subset(G, 10)
        assert sorted(picked) == [0, 1]

    def test_k_zero_returns_empty(self):
        assert select_informative_subset(np.eye(3), 0) == []

    def test_selection_is_deterministic(self):
        rng = np.random.default_rng(4)
        G = rng.random((12, 5))
        a = select_informative_subset(G, 6)
        b = select_informative_subset(G, 6)
        assert a == b


class TestSubsetReport:
    def test_full_selection_retains_all_information(self):
        rng = np.random.default_rng(0)
        G = rng.random((8, 3))
        report = subset_report(G, list(range(8)))
        np.testing.assert_allclose(report["information_retained"], 1.0)

    def test_subset_retention_between_zero_and_one(self):
        rng = np.random.default_rng(1)
        G = rng.random((10, 4))
        picked = select_informative_subset(G, 5)
        report = subset_report(G, picked)
        assert 0.0 < report["information_retained"] < 1.0
