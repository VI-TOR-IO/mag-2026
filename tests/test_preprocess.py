import unittest

import bootstrap  # noqa: F401
import pandas as pd

from app.constants import FEATURE_COLUMNS
from app.ml.preprocess import build_dataset_profile, validate_dataset


class PreprocessTestCase(unittest.TestCase):
    def test_validate_dataset_keeps_expected_columns_and_numeric_values(self):
        row = {"Sale": "1000"}
        row.update({feature: "1,5" for feature in FEATURE_COLUMNS})
        row["unused"] = "ignored"

        df = validate_dataset(pd.DataFrame([row]))

        self.assertEqual(list(df.columns), ["Sale"] + FEATURE_COLUMNS)
        self.assertEqual(float(df.loc[0, "Sale"]), 1000.0)
        self.assertEqual(float(df.loc[0, FEATURE_COLUMNS[0]]), 1.5)

    def test_validate_dataset_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "отсутствуют"):
            validate_dataset(pd.DataFrame([{"Sale": 1}]))

    def test_build_dataset_profile_reports_quality_issues(self):
        row = {"Sale": 100}
        row.update({feature: 1 for feature in FEATURE_COLUMNS})
        row["Discount"] = 1.5
        row["Price"] = -10

        profile = build_dataset_profile(pd.DataFrame([row]))
        messages = " ".join(issue["message"] for issue in profile["issues"])

        self.assertFalse(profile["is_valid"])
        self.assertIn("Price", messages)
        self.assertIn("Discount", messages)


if __name__ == "__main__":
    unittest.main()
