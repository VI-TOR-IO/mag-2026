import unittest

import bootstrap  # noqa: F401

from app.charts import (
    build_loss_chart,
    build_model_comparison_chart,
    build_prediction_diagnostics,
    build_prediction_history_charts,
)


class ChartsTestCase(unittest.TestCase):
    def test_prediction_diagnostics_builds_scatter_and_histogram(self):
        diagnostics = build_prediction_diagnostics(
            y_true=[100, 200, 300, 400],
            y_pred=[110, 190, 310, 420],
            max_points=3,
            bins_count=4,
        )

        self.assertEqual(diagnostics["summary"]["points_count"], 4)
        self.assertEqual(diagnostics["summary"]["shown_points_count"], 3)
        self.assertEqual(len(diagnostics["scatter"]["points"]), 3)
        self.assertEqual(len(diagnostics["histogram"]["bins"]), 4)

    def test_model_comparison_chart_builds_r2_and_error_bars(self):
        chart = build_model_comparison_chart([
            {"key": "linear", "model": "Linear", "r2": 0.8, "mae": 10, "rmse": 20},
            {"key": "knn", "model": "KNN", "r2": 0.5, "mae": 20, "rmse": 30},
        ])

        self.assertEqual(len(chart["r2_bars"]), 2)
        self.assertEqual(len(chart["error_rows"]), 2)
        self.assertGreater(chart["r2_bars"][0]["width"], chart["r2_bars"][1]["width"])
        self.assertLess(chart["error_rows"][0]["mae"]["width"], chart["error_rows"][1]["mae"]["width"])

    def test_loss_chart_builds_lines(self):
        chart = build_loss_chart({
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
            "best_epoch": 3,
            "stopped_epoch": 3,
        })

        self.assertIsNotNone(chart["train_points"])
        self.assertIsNotNone(chart["val_points"])
        self.assertEqual(chart["best_epoch"], 3)

    def test_prediction_history_charts(self):
        stats = {
            "error_rows": [{"percent_diff": 10}, {"percent_diff": 5}],
            "model_counts": [{"model_name": "linear", "cnt": 2}],
            "avg_errors": [{"model_name": "linear", "avg_percent_diff": 7.5}],
        }

        charts = build_prediction_history_charts(stats, {"linear": "Linear"})

        self.assertIsNotNone(charts["trend"])
        self.assertEqual(charts["model_counts"][0]["label"], "Linear")
        self.assertEqual(charts["avg_errors"][0]["value"], "7.50")


if __name__ == "__main__":
    unittest.main()
