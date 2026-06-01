import unittest

import bootstrap  # noqa: F401

from app.ml.evaluation import (
    build_adequacy_report,
    build_baseline_metrics,
    build_residual_report,
)


class EvaluationTestCase(unittest.TestCase):
    def test_baseline_metrics_use_train_mean(self):
        baseline = build_baseline_metrics(
            y_train=[10, 20, 30],
            y_test=[20, 30],
        )

        self.assertEqual(baseline["prediction_value"], 20)
        self.assertIn("rmse", baseline)
        self.assertIn("mae", baseline)

    def test_residual_report_summarizes_error_distribution(self):
        report = build_residual_report(
            y_true=[100, 120, 140, 160],
            y_pred=[95, 125, 150, 150],
        )

        self.assertEqual(report["residual_mean"], 0)
        self.assertEqual(report["max_abs_error"], 10)
        self.assertIsNotNone(report["mape"])

    def test_adequacy_report_marks_model_better_than_baseline(self):
        report = build_adequacy_report(
            test_metrics={"r2": 0.8, "mae": 5, "rmse": 6},
            train_metrics={"r2": 0.85, "mae": 4, "rmse": 5},
            baseline_metrics={"r2": 0.0, "mae": 20, "rmse": 24},
            cv_metrics={"r2_mean": 0.76, "r2_std": 0.04},
            residual_report={"p90_abs_error_share": 12},
        )

        self.assertEqual(report["verdict"], "good")
        self.assertGreater(report["baseline_improvement_percent"], 0)

    def test_adequacy_report_flags_model_worse_than_baseline(self):
        report = build_adequacy_report(
            test_metrics={"r2": -0.2, "mae": 30, "rmse": 35},
            train_metrics={"r2": 0.95, "mae": 2, "rmse": 3},
            baseline_metrics={"r2": 0.0, "mae": 10, "rmse": 12},
            cv_metrics={"r2_mean": -0.1, "r2_std": 0.5},
            residual_report={"p90_abs_error_share": 60},
        )

        self.assertEqual(report["verdict"], "bad")
        self.assertTrue(any(check["status"] == "error" for check in report["checks"]))


if __name__ == "__main__":
    unittest.main()
