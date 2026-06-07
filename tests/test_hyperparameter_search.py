import unittest

import bootstrap  # noqa: F401

from app.ml.hyperparameter_search import build_candidate_grid, normalize_search_mode


class HyperparameterSearchTestCase(unittest.TestCase):
    def test_normalize_search_mode_falls_back_to_none(self):
        self.assertEqual(normalize_search_mode("fast"), "fast")
        self.assertEqual(normalize_search_mode("unknown"), "none")
        self.assertEqual(normalize_search_mode(None), "none")

    def test_linear_balanced_grid_includes_regularized_models(self):
        candidates = build_candidate_grid(
            model_name="linear",
            mode="balanced",
            options={},
            train_size=30,
        )

        estimators = {candidate["linear_estimator"] for candidate in candidates}
        self.assertIn("linear", estimators)
        self.assertIn("ridge", estimators)
        self.assertNotIn("lasso", estimators)

    def test_knn_grid_keeps_neighbors_within_train_size(self):
        candidates = build_candidate_grid(
            model_name="knn",
            mode="balanced",
            options={"knn_neighbors": 50},
            train_size=6,
        )

        self.assertTrue(candidates)
        self.assertTrue(all(candidate["knn_neighbors"] <= 6 for candidate in candidates))
        self.assertTrue(any(candidate["knn_weights"] == "distance" for candidate in candidates))

    def test_mlp_fast_grid_is_limited(self):
        candidates = build_candidate_grid(
            model_name="mlp",
            mode="fast",
            options={"mlp_epochs": 500, "mlp_patience": 80},
            train_size=100,
        )

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["mlp_epochs"] <= 220 for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
