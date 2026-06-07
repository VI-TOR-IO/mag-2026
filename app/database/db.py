import os
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

VALID_ROLES = {"admin", "analyst"}


def get_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_column(cursor, table_name: str, column_name: str, column_definition: str):
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row["name"] for row in cursor.fetchall()}
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def init_db(
    db_path: str,
    create_default_admin: bool = True,
    default_admin_username: str = "admin",
    default_admin_email: str = "admin@example.com",
    default_admin_password: str = "admin123",
):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            instrspending REAL NOT NULL,
            discount_value REAL NOT NULL,
            tvspending REAL NOT NULL,
            stockrate REAL NOT NULL,
            price REAL NOT NULL,
            radio REAL NOT NULL,
            onlineadsspending REAL NOT NULL,
            prediction REAL NOT NULL,
            actual_sale REAL,
            absolute_diff REAL,
            percent_diff REAL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'analyst',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_title TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            rows_count INTEGER NOT NULL,
            train_size INTEGER NOT NULL,
            test_size INTEGER NOT NULL,
            feature_columns TEXT NOT NULL,
            parameters TEXT,
            metrics_r2 REAL NOT NULL,
            metrics_mae REAL NOT NULL,
            metrics_rmse REAL NOT NULL,
            cv_r2_mean REAL,
            cv_r2_std REAL,
            artifact_dir TEXT,
            model_path TEXT,
            x_scaler_path TEXT,
            y_scaler_path TEXT,
            coefficients TEXT,
            loss_history TEXT,
            dataset_report TEXT,
            train_metrics TEXT,
            baseline_metrics TEXT,
            residual_report TEXT,
            adequacy_report TEXT,
            training_protocol TEXT,
            hyperparameter_search_report TEXT,
            created_at TEXT NOT NULL
        )
    """)

    ensure_column(cursor, "model_training_history", "train_metrics", "TEXT")
    ensure_column(cursor, "model_training_history", "baseline_metrics", "TEXT")
    ensure_column(cursor, "model_training_history", "residual_report", "TEXT")
    ensure_column(cursor, "model_training_history", "adequacy_report", "TEXT")
    ensure_column(cursor, "model_training_history", "training_protocol", "TEXT")
    ensure_column(cursor, "model_training_history", "hyperparameter_search_report", "TEXT")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at
        ON prediction_history(created_at)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email
        ON users(email)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_training_history_created_at
        ON model_training_history(created_at)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_training_history_model_name
        ON model_training_history(model_name)
    """)

    conn.commit()

    if create_default_admin:
        cursor.execute("SELECT COUNT(*) as cnt FROM users")
        users_count = cursor.fetchone()["cnt"]

        if users_count == 0:
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                default_admin_username.strip(),
                default_admin_email.strip().lower(),
                generate_password_hash(default_admin_password),
                "admin",
                now_string(),
            ))
            conn.commit()

    conn.close()


def create_user(db_path: str, username: str, email: str, password: str, role: str = "analyst"):
    if role not in VALID_ROLES:
        raise ValueError("Некорректная роль пользователя.")

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO users (username, email, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        username.strip(),
        email.strip().lower(),
        generate_password_hash(password),
        role,
        now_string(),
    ))

    conn.commit()
    conn.close()


def get_user_by_email(db_path: str, email: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return row


def get_user_by_id(db_path: str, user_id: int):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_user_by_username(db_path: str, username: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return row


def verify_user_password(user_row, password: str) -> bool:
    if not user_row:
        return False
    return check_password_hash(user_row["password_hash"], password)


def get_all_users(db_path: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, email, role, created_at
        FROM users
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def update_user_role(db_path: str, user_id: int, new_role: str):
    if new_role not in VALID_ROLES:
        raise ValueError("Некорректная роль пользователя.")

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE users
        SET role = ?
        WHERE id = ?
    """, (new_role, user_id))

    conn.commit()
    conn.close()


def delete_user_by_id(db_path: str, user_id: int):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM users
        WHERE id = ?
    """, (user_id,))

    conn.commit()
    conn.close()


def update_user_profile(db_path: str, user_id: int, username: str, email: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE users
        SET username = ?, email = ?
        WHERE id = ?
    """, (username.strip(), email.strip().lower(), user_id))

    conn.commit()
    conn.close()


def update_user_password(db_path: str, user_id: int, new_password: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE users
        SET password_hash = ?
        WHERE id = ?
    """, (generate_password_hash(new_password), user_id))

    conn.commit()
    conn.close()


def save_prediction(
    db_path: str,
    model_name: str,
    features: dict,
    prediction: float,
    actual_sale: float | None,
    absolute_diff: float | None,
    percent_diff: float | None,
):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_history (
            model_name,
            instrspending,
            discount_value,
            tvspending,
            stockrate,
            price,
            radio,
            onlineadsspending,
            prediction,
            actual_sale,
            absolute_diff,
            percent_diff,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name,
        features["InStrSpending"],
        features["Discount"],
        features["TVSpending"],
        features["StockRate"],
        features["Price"],
        features["Radio"],
        features["OnlineAdsSpending"],
        prediction,
        actual_sale,
        absolute_diff,
        percent_diff,
        now_string(),
    ))

    conn.commit()
    conn.close()


def get_prediction_history(db_path: str, limit: int = 20):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_prediction_by_id(db_path: str, prediction_id: int):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM prediction_history
        WHERE id = ?
    """, (prediction_id,))

    row = cursor.fetchone()
    conn.close()
    return row


def delete_prediction_by_id(db_path: str, prediction_id: int) -> bool:
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM prediction_history
        WHERE id = ?
    """, (prediction_id,))

    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_all_prediction_history(db_path: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def save_training_run(
    db_path: str,
    model_name: str,
    model_title: str,
    dataset_name: str,
    rows_count: int,
    train_size: int,
    test_size: int,
    feature_columns: str,
    parameters: str | None,
    metrics: dict,
    cv_metrics: dict | None,
    artifact_paths: dict,
    coefficients: str | None,
    loss_history: str | None,
    dataset_report: str | None,
    train_metrics: str | None = None,
    baseline_metrics: str | None = None,
    residual_report: str | None = None,
    adequacy_report: str | None = None,
    training_protocol: str | None = None,
    hyperparameter_search_report: str | None = None,
) -> int:
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_training_history (
            model_name,
            model_title,
            dataset_name,
            rows_count,
            train_size,
            test_size,
            feature_columns,
            parameters,
            metrics_r2,
            metrics_mae,
            metrics_rmse,
            cv_r2_mean,
            cv_r2_std,
            artifact_dir,
            model_path,
            x_scaler_path,
            y_scaler_path,
            coefficients,
            loss_history,
            dataset_report,
            train_metrics,
            baseline_metrics,
            residual_report,
            adequacy_report,
            training_protocol,
            hyperparameter_search_report,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name,
        model_title,
        dataset_name,
        rows_count,
        train_size,
        test_size,
        feature_columns,
        parameters,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
        cv_metrics.get("r2_mean") if cv_metrics else None,
        cv_metrics.get("r2_std") if cv_metrics else None,
        artifact_paths.get("artifact_dir"),
        artifact_paths.get("model_path"),
        artifact_paths.get("x_scaler_path"),
        artifact_paths.get("y_scaler_path"),
        coefficients,
        loss_history,
        dataset_report,
        train_metrics,
        baseline_metrics,
        residual_report,
        adequacy_report,
        training_protocol,
        hyperparameter_search_report,
        now_string(),
    ))

    conn.commit()
    training_id = cursor.lastrowid
    conn.close()
    return training_id


def get_training_history(db_path: str, limit: int | None = None):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    sql = """
        SELECT *
        FROM model_training_history
        ORDER BY id DESC
    """
    params = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_training_by_id(db_path: str, training_id: int):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM model_training_history
        WHERE id = ?
    """, (training_id,))

    row = cursor.fetchone()
    conn.close()
    return row


def get_training_summary_stats(db_path: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS cnt FROM model_training_history")
    total_training_runs = cursor.fetchone()["cnt"]

    cursor.execute("""
        SELECT model_name, model_title, COUNT(*) AS cnt
        FROM model_training_history
        GROUP BY model_name, model_title
        ORDER BY cnt DESC
    """)
    model_counts = cursor.fetchall()

    cursor.execute("""
        SELECT model_name, model_title, AVG(metrics_mae) AS avg_mae, AVG(metrics_rmse) AS avg_rmse
        FROM model_training_history
        GROUP BY model_name, model_title
        ORDER BY avg_rmse ASC
    """)
    avg_errors = cursor.fetchall()

    cursor.execute("""
        SELECT *
        FROM model_training_history
        ORDER BY id DESC
        LIMIT 5
    """)
    recent_training_runs = cursor.fetchall()

    conn.close()

    return {
        "total_training_runs": total_training_runs,
        "model_counts": model_counts,
        "avg_errors": avg_errors,
        "recent_training_runs": recent_training_runs,
    }


def get_prediction_history_stats(db_path: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT created_at, model_name, percent_diff
        FROM prediction_history
        WHERE percent_diff IS NOT NULL
        ORDER BY id ASC
    """)
    error_rows = cursor.fetchall()

    cursor.execute("""
        SELECT model_name, COUNT(*) AS cnt
        FROM prediction_history
        GROUP BY model_name
        ORDER BY cnt DESC
    """)
    model_counts = cursor.fetchall()

    cursor.execute("""
        SELECT model_name, AVG(percent_diff) AS avg_percent_diff
        FROM prediction_history
        WHERE percent_diff IS NOT NULL
        GROUP BY model_name
        ORDER BY avg_percent_diff ASC
    """)
    avg_errors = cursor.fetchall()

    conn.close()

    return {
        "error_rows": error_rows,
        "model_counts": model_counts,
        "avg_errors": avg_errors,
    }


def get_admin_dashboard_stats(db_path: str):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS cnt FROM users")
    total_users = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin'")
    total_admins = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) AS cnt FROM users WHERE role = 'analyst'")
    total_analysts = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_history")
    total_predictions = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) AS cnt FROM model_training_history")
    total_training_runs = cursor.fetchone()["cnt"]

    cursor.execute("""
        SELECT AVG(percent_diff) AS avg_percent_diff
        FROM prediction_history
        WHERE percent_diff IS NOT NULL
    """)
    avg_percent_diff = cursor.fetchone()["avg_percent_diff"]

    cursor.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 5
    """)
    recent_predictions = cursor.fetchall()

    conn.close()

    return {
        "total_users": total_users,
        "total_admins": total_admins,
        "total_analysts": total_analysts,
        "total_predictions": total_predictions,
        "total_training_runs": total_training_runs,
        "avg_percent_diff": round(avg_percent_diff, 2) if avg_percent_diff is not None else None,
        "recent_predictions": recent_predictions,
    }
