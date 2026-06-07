import os
import tempfile
import unittest

import bootstrap  # noqa: F401

from app.database.db import (
    create_user,
    delete_prediction_by_id,
    get_all_users,
    get_prediction_by_id,
    get_training_by_id,
    get_training_history,
    get_user_by_email,
    init_db,
    save_training_run,
    save_prediction,
    update_user_role,
    verify_user_password,
)


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "app.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_db_creates_default_admin(self):
        init_db(self.db_path, default_admin_password="secret123")

        admin = get_user_by_email(self.db_path, "admin@example.com")

        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "admin")
        self.assertTrue(verify_user_password(admin, "secret123"))

    def test_create_user_normalizes_email(self):
        init_db(self.db_path, create_default_admin=False)

        create_user(self.db_path, "analyst", "Analyst@Example.COM", "password1")
        user = get_user_by_email(self.db_path, "analyst@example.com")

        self.assertIsNotNone(user)
        self.assertEqual(user["email"], "analyst@example.com")
        self.assertEqual(user["role"], "analyst")

    def test_invalid_role_is_rejected(self):
        init_db(self.db_path, create_default_admin=False)

        with self.assertRaises(ValueError):
            create_user(self.db_path, "bad", "bad@example.com", "password1", role="owner")

        create_user(self.db_path, "analyst", "analyst@example.com", "password1")
        user = get_all_users(self.db_path)[0]

        with self.assertRaises(ValueError):
            update_user_role(self.db_path, user["id"], "owner")

    def test_save_training_run(self):
        init_db(self.db_path, create_default_admin=False)

        training_id = save_training_run(
            db_path=self.db_path,
            model_name="linear",
            model_title="Linear Regression",
            dataset_name="sample.csv",
            rows_count=100,
            train_size=80,
            test_size=20,
            feature_columns='["Price"]',
            parameters="{}",
            metrics={"r2": 0.8, "mae": 10, "rmse": 12},
            cv_metrics={"r2_mean": 0.75, "r2_std": 0.03},
            artifact_paths={"artifact_dir": "models/versions/linear_1"},
            coefficients="[]",
            loss_history=None,
            dataset_report="{}",
            train_metrics='{"r2": 0.85}',
            baseline_metrics='{"rmse": 20}',
            residual_report='{"p90_abs_error": 5}',
            adequacy_report='{"verdict": "good"}',
            training_protocol='[{"title": "done"}]',
            hyperparameter_search_report='{"enabled": true}',
        )

        row = get_training_by_id(self.db_path, training_id)
        rows = get_training_history(self.db_path)

        self.assertEqual(row["model_name"], "linear")
        self.assertEqual(row["metrics_rmse"], 12)
        self.assertEqual(row["adequacy_report"], '{"verdict": "good"}')
        self.assertEqual(row["training_protocol"], '[{"title": "done"}]')
        self.assertEqual(row["hyperparameter_search_report"], '{"enabled": true}')
        self.assertEqual(len(rows), 1)

    def test_delete_prediction_by_id(self):
        init_db(self.db_path, create_default_admin=False)

        save_prediction(
            db_path=self.db_path,
            model_name="linear",
            features={
                "InStrSpending": 1,
                "Discount": 0.1,
                "TVSpending": 2,
                "StockRate": 0.5,
                "Price": 10,
                "Radio": 3,
                "OnlineAdsSpending": 4,
            },
            prediction=100,
            actual_sale=95,
            absolute_diff=5,
            percent_diff=5.26,
        )

        self.assertIsNotNone(get_prediction_by_id(self.db_path, 1))
        self.assertTrue(delete_prediction_by_id(self.db_path, 1))
        self.assertIsNone(get_prediction_by_id(self.db_path, 1))
        self.assertFalse(delete_prediction_by_id(self.db_path, 999))


if __name__ == "__main__":
    unittest.main()
