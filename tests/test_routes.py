import os
import tempfile
import unittest

import bootstrap  # noqa: F401

from app import create_app
from app.database.db import create_user, get_prediction_by_id, save_prediction, save_training_run


class RoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "app.db")
        self.data_dir = os.path.join(self.tmp.name, "data")
        self.model_dir = os.path.join(self.tmp.name, "models")

        self.app = create_app({
            "TESTING": True,
            "DATABASE_PATH": self.db_path,
            "DATA_DIR": self.data_dir,
            "MODEL_DIR": self.model_dir,
            "DEFAULT_ADMIN_ENABLED": True,
            "DEFAULT_ADMIN_USERNAME": "admin",
            "DEFAULT_ADMIN_EMAIL": "admin@example.com",
            "DEFAULT_ADMIN_PASSWORD": "admin123",
            "SECRET_KEY": "test-secret",
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def login(self, email="admin@example.com", password="admin123"):
        return self.client.post("/login", data={
            "email": email,
            "password": password,
        }, follow_redirects=True)

    def test_index_is_public(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Рабочая панель".encode("utf-8"), response.data)

    def test_protected_page_redirects_to_login(self):
        response = self.client.get("/predict")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_register_then_login(self):
        response = self.client.post("/register", data={
            "username": "analyst",
            "email": "Analyst@Example.COM",
            "password": "password1",
            "confirm_password": "password1",
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Вход в систему".encode("utf-8"), response.data)

        response = self.login("analyst@example.com", "password1")

        self.assertEqual(response.status_code, 200)
        self.assertIn("analyst".encode("utf-8"), response.data)

    def test_admin_can_export_history(self):
        with self.app.app_context():
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

        self.login()
        response = self.client.get("/history/export")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn("prediction_history.csv", response.headers["Content-Disposition"])

    def test_admin_can_delete_prediction_from_history(self):
        with self.app.app_context():
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

        self.login()
        response = self.client.post("/history/1/delete", follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(get_prediction_by_id(self.db_path, 1))

    def test_analyst_cannot_delete_prediction(self):
        with self.app.app_context():
            create_user(self.db_path, "analyst2", "analyst2@example.com", "password1")
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

        self.login("analyst2@example.com", "password1")
        response = self.client.post("/history/1/delete", follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(get_prediction_by_id(self.db_path, 1))

    def test_sample_row_requires_login(self):
        response = self.client.get("/predict/sample-row")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_models_page_requires_login(self):
        response = self.client.get("/models")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_logged_in_admin_can_open_models_page(self):
        self.login()
        response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        self.assertIn("models".encode("utf-8"), response.data.lower())

    def test_predict_page_shows_improved_manual_forecast_ui(self):
        self.login()
        response = self.client.get("/predict")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Статус модели".encode("utf-8"), response.data)
        self.assertIn("Сводка сценария".encode("utf-8"), response.data)
        self.assertIn("Подставить строку из датасета".encode("utf-8"), response.data)

    def test_logged_in_admin_can_open_main_pages(self):
        self.login()

        pages = [
            "/",
            "/upload",
            "/compare",
            "/models",
            "/predict",
            "/history",
            "/profile",
            "/admin",
            "/users",
        ]

        for path in pages:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

    def test_model_detail_shows_ml_evaluation_report(self):
        with self.app.app_context():
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
                train_metrics='{"r2": 0.85, "mae": 9, "rmse": 11}',
                baseline_metrics='{"strategy": "mean", "prediction_value": 20, "r2": 0, "mae": 30, "rmse": 40}',
                residual_report='{"residual_mean": 0, "residual_std": 2, "p50_abs_error": 4, "p90_abs_error": 8, "max_abs_error": 12, "mape": 5}',
                adequacy_report='{"verdict": "good", "verdict_label": "Адекватна", "summary": "OK", "baseline_improvement_percent": 70, "train_test_r2_gap": 0.05, "checks": []}',
                training_protocol='[{"title": "Обучение", "detail": "Готово", "status": "success", "duration_seconds": 0.1}]',
            )

        self.login()
        response = self.client.get(f"/models/{training_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Проверка адекватности модели".encode("utf-8"), response.data)
        self.assertIn("Протокол обучения".encode("utf-8"), response.data)


if __name__ == "__main__":
    unittest.main()
