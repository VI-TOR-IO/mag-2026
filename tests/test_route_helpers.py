import os
import tempfile
import unittest

import bootstrap  # noqa: F401

from app.routes import (
    allowed_file,
    build_prediction_error_summary,
    build_prediction_summary,
    calculate_prediction_error,
    model_artifacts_exist,
    parse_float,
    validate_prediction_inputs,
)


class RouteHelpersTestCase(unittest.TestCase):
    def test_allowed_file_accepts_only_csv(self):
        self.assertTrue(allowed_file("data.csv"))
        self.assertTrue(allowed_file("DATA.CSV"))
        self.assertFalse(allowed_file("data.xlsx"))
        self.assertFalse(allowed_file("data"))

    def test_parse_float_accepts_comma_decimal_separator(self):
        self.assertEqual(parse_float("12,5", "value"), 12.5)

    def test_prediction_error_handles_zero_actual_sale(self):
        absolute_diff, percent_diff = calculate_prediction_error(10, 0)

        self.assertEqual(absolute_diff, 10)
        self.assertIsNone(percent_diff)

    def test_prediction_validation_rejects_invalid_ranges(self):
        features = {
            "InStrSpending": 10,
            "Discount": 1.2,
            "TVSpending": 20,
            "StockRate": 0.8,
            "Price": 50,
            "Radio": 5,
            "OnlineAdsSpending": 7,
        }

        with self.assertRaises(ValueError):
            validate_prediction_inputs(features, actual_sale=None)

    def test_prediction_summary_builds_business_values(self):
        summary = build_prediction_summary(
            {
                "InStrSpending": 10,
                "Discount": 0.1,
                "TVSpending": 20,
                "StockRate": 0.8,
                "Price": 100,
                "Radio": 5,
                "OnlineAdsSpending": 15,
            },
            prediction=200,
        )

        self.assertEqual(summary["total_marketing_spend"], 50)
        self.assertEqual(summary["price_after_discount"], 90)
        self.assertEqual(summary["discount_percent"], 10)

    def test_prediction_error_summary_uses_quality_bands(self):
        summary = build_prediction_error_summary(absolute_diff=4, percent_diff=4)

        self.assertEqual(summary["level"], "success")

    def test_model_artifacts_exist_checks_active_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for filename in ["linear_model.pkl", "linear_x_scaler.pkl", "linear_y_scaler.pkl"]:
                open(os.path.join(tmp, filename), "w", encoding="utf-8").close()

            self.assertTrue(model_artifacts_exist("linear", tmp))
            self.assertFalse(model_artifacts_exist("knn", tmp))


if __name__ == "__main__":
    unittest.main()
