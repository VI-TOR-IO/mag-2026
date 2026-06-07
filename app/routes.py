import csv
import io
import json
import os
import time
from datetime import datetime

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from app.constants import (
    ALLOWED_EXTENSIONS,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_HELP,
    FEATURE_INPUT_RULES,
    FEATURE_LABELS,
    FEATURE_UNITS,
    MODEL_OPTIONS,
    PRESET_DATASETS,
)
from app.database.db import (
    delete_prediction_by_id,
    get_admin_dashboard_stats,
    get_all_prediction_history,
    get_prediction_history_stats,
    get_prediction_by_id,
    get_prediction_history,
    get_training_by_id,
    get_training_history,
    get_training_summary_stats,
    save_prediction,
    save_training_run,
)
from app.decorators import login_required, role_required
from app.ml.hyperparameter_search import SEARCH_MODE_OPTIONS, normalize_search_mode

main_bp = Blueprint("main", __name__)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resolve_dataset(request_file, preset_dataset: str | None, data_dir: str):
    if preset_dataset:
        preset_info = PRESET_DATASETS.get(preset_dataset)
        if not preset_info:
            raise ValueError("Выбран неизвестный предустановленный датасет.")

        file_path = os.path.join(data_dir, "samples", preset_info["filename"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {preset_info['filename']} не найден в data/samples.")

        return file_path, preset_info["title"]

    if not request_file or request_file.filename == "":
        raise ValueError("Выберите CSV-файл или предустановленный датасет.")

    if not allowed_file(request_file.filename):
        raise ValueError("Разрешён только CSV-файл.")

    filename = secure_filename(request_file.filename)
    if not filename:
        raise ValueError("Некорректное имя файла.")

    upload_dir = os.path.join(data_dir, "raw")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    request_file.save(file_path)
    return file_path, filename


def parse_float(value: str, field_label: str) -> float:
    try:
        return float(value.strip().replace(",", "."))
    except (TypeError, ValueError):
        raise ValueError(f"Поле «{field_label}» должно быть числом.") from None


def parse_prediction_form(form):
    input_data = []
    entered_values = {}
    features_dict = {}

    for feature in FEATURE_COLUMNS:
        raw_value = form.get(feature, "").strip()
        label = FEATURE_LABELS.get(feature, feature)
        if raw_value == "":
            raise ValueError(f"Заполните поле: {label}.")

        value = parse_float(raw_value, label)
        input_data.append(value)
        entered_values[feature] = raw_value
        features_dict[feature] = value

    actual_sale_raw = form.get("actual_sale", "").strip()
    actual_sale = None
    if actual_sale_raw:
        actual_sale = parse_float(actual_sale_raw, "Фактическое значение Sale")

    return input_data, entered_values, features_dict, actual_sale


def validate_prediction_inputs(features: dict, actual_sale: float | None):
    bounded_fields = {
        "Discount": "Скидка",
        "StockRate": "Доля товара в наличии",
    }
    for field, label in bounded_fields.items():
        if features[field] < 0 or features[field] > 1:
            raise ValueError(f"Поле «{label}» должно быть в диапазоне 0..1.")

    non_negative_fields = [
        "InStrSpending",
        "TVSpending",
        "Price",
        "Radio",
        "OnlineAdsSpending",
    ]
    for field in non_negative_fields:
        if features[field] < 0:
            raise ValueError(f"Поле «{FEATURE_LABELS[field]}» не может быть отрицательным.")

    if actual_sale is not None and actual_sale < 0:
        raise ValueError("Фактическое Sale не может быть отрицательным.")


def calculate_prediction_error(prediction: float, actual_sale: float | None):
    if actual_sale is None:
        return None, None

    absolute_diff = abs(prediction - actual_sale)
    percent_diff = None
    if actual_sale != 0:
        percent_diff = (absolute_diff / abs(actual_sale)) * 100

    return absolute_diff, percent_diff


def build_prediction_error_summary(absolute_diff: float | None, percent_diff: float | None):
    if absolute_diff is None:
        return None

    if percent_diff is None:
        return {
            "level": "warning",
            "label": "Факт равен 0",
            "message": "Абсолютное отклонение рассчитано, процентную ошибку корректно посчитать нельзя.",
        }

    if percent_diff <= 5:
        return {
            "level": "success",
            "label": "Высокая точность",
            "message": "Отклонение не превышает 5% от фактического Sale.",
        }

    if percent_diff <= 15:
        return {
            "level": "success",
            "label": "Нормальная точность",
            "message": "Отклонение находится в рабочем диапазоне до 15%.",
        }

    if percent_diff <= 30:
        return {
            "level": "warning",
            "label": "Среднее отклонение",
            "message": "Прогноз стоит использовать осторожно: проверьте входные признаки и актуальность модели.",
        }

    return {
        "level": "error",
        "label": "Высокое отклонение",
        "message": "Ошибка выше 30%; прогноз лучше перепроверить или переобучить модель на свежих данных.",
    }


def build_prediction_summary(features: dict, prediction: float):
    total_marketing_spend = (
        features["InStrSpending"]
        + features["TVSpending"]
        + features["Radio"]
        + features["OnlineAdsSpending"]
    )
    price_after_discount = features["Price"] * (1 - features["Discount"])
    marketing_to_prediction_percent = None
    if prediction:
        marketing_to_prediction_percent = total_marketing_spend / abs(prediction) * 100

    return {
        "total_marketing_spend": round(float(total_marketing_spend), 2),
        "price_after_discount": round(float(price_after_discount), 2),
        "discount_percent": round(float(features["Discount"] * 100), 2),
        "stock_percent": round(float(features["StockRate"] * 100), 2),
        "marketing_to_prediction_percent": (
            round(float(marketing_to_prediction_percent), 2)
            if marketing_to_prediction_percent is not None
            else None
        ),
    }


def parse_positive_int(value: str | None, default: int, minimum: int, maximum: int, field_name: str) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{field_name} должно быть целым числом.") from None

    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field_name} должно быть в диапазоне {minimum}..{maximum}.")

    return parsed


def parse_training_options(form):
    return {
        "knn_neighbors": parse_positive_int(form.get("knn_neighbors"), 5, 1, 50, "Количество соседей KNN"),
        "knn_weights": form.get("knn_weights", "uniform") if form.get("knn_weights", "uniform") in {"uniform", "distance"} else "uniform",
        "knn_p": parse_positive_int(form.get("knn_p"), 2, 1, 2, "Метрика расстояния KNN"),
        "mlp_epochs": parse_positive_int(form.get("mlp_epochs"), 300, 20, 2000, "Количество эпох MLP"),
        "mlp_patience": parse_positive_int(form.get("mlp_patience"), 25, 5, 200, "Patience MLP"),
        "mlp_learning_rate": 0.001,
        "hyperparameter_search": normalize_search_mode(form.get("hyperparameter_search", "none")),
    }


def model_parameters(model_name: str, options: dict):
    search_mode = options.get("hyperparameter_search", "none")
    if model_name == "knn":
        return {
            "n_neighbors": options["knn_neighbors"],
            "weights": options.get("knn_weights", "uniform"),
            "p": options.get("knn_p", 2),
            "hyperparameter_search": search_mode,
        }
    if model_name == "mlp":
        return {
            "epochs": options["mlp_epochs"],
            "patience": options["mlp_patience"],
            "learning_rate": options["mlp_learning_rate"],
            "early_stopping": True,
            "seed": 42,
            "hyperparameter_search": search_mode,
        }
    return {
        "estimator": options.get("linear_estimator", "linear"),
        "alpha": options.get("alpha"),
        "hyperparameter_search": search_mode,
    }


def train_model(model_name: str, X_train, y_train, options: dict | None = None, validation_data=None):
    from app.ml.train_models import train_knn, train_linear_regression, train_mlp, train_ridge_regression

    options = options or {}

    if model_name == "linear":
        estimator = options.get("linear_estimator", "linear")
        if estimator == "ridge":
            return train_ridge_regression(X_train, y_train, alpha=options.get("alpha", 1.0)), None
        return train_linear_regression(X_train, y_train), None
    if model_name == "knn":
        safe_neighbors = max(1, min(options.get("knn_neighbors", 5), len(X_train)))
        return train_knn(
            X_train,
            y_train,
            n_neighbors=safe_neighbors,
            weights=options.get("knn_weights", "uniform"),
            p=options.get("knn_p", 2),
        ), None
    if model_name == "mlp":
        model, loss_history = train_mlp(
            X_train,
            y_train,
            epochs=options.get("mlp_epochs", 300),
            lr=options.get("mlp_learning_rate", 0.001),
            validation_data=validation_data,
            patience=options.get("mlp_patience", 25),
            return_history=True,
        )
        return model, loss_history

    raise ValueError("Неизвестная модель.")


def predict_test_values(model, model_name: str, X_test):
    if model_name in {"linear", "knn"}:
        return model.predict(X_test)

    if model_name == "mlp":
        import torch

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            return model(X_test_tensor).numpy().ravel()

    raise ValueError("Неизвестная модель.")


def inverse_target_values(y_scaler, values):
    import numpy as np

    return y_scaler.inverse_transform(np.array(values).reshape(-1, 1)).ravel()


def build_linear_coefficients(model):
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return None

    items = []
    for feature, coefficient in zip(FEATURE_COLUMNS, coefficients):
        value = float(coefficient)
        items.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "coefficient": round(value, 6),
            "abs_coefficient": round(abs(value), 6),
        })

    return sorted(items, key=lambda item: item["abs_coefficient"], reverse=True)


def run_cross_validation(df, model_name: str, options: dict):
    if len(df) < 6:
        return None

    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[FEATURE_COLUMNS]
    y = df["Sale"]
    n_splits = min(5, len(df))

    if model_name == "linear":
        from sklearn.linear_model import LinearRegression

        estimator = make_pipeline(StandardScaler(), LinearRegression())
        scores = cross_val_score(estimator, X, y, cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), scoring="r2")
        return {"r2_mean": round(float(np.mean(scores)), 4), "r2_std": round(float(np.std(scores)), 4)}

    if model_name == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        estimator = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=options.get("knn_neighbors", 5)),
        )
        scores = cross_val_score(estimator, X, y, cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), scoring="r2")
        return {"r2_mean": round(float(np.mean(scores)), 4), "r2_std": round(float(np.std(scores)), 4)}

    if model_name == "mlp":
        from sklearn.preprocessing import StandardScaler
        from app.ml.train_models import train_mlp

        scores = []
        mlp_splits = min(3, len(df))
        kfold = KFold(n_splits=mlp_splits, shuffle=True, random_state=42)
        for train_index, test_index in kfold.split(X):
            X_train_raw = X.iloc[train_index]
            X_test_raw = X.iloc[test_index]
            y_train_raw = y.iloc[train_index]
            y_test_raw = y.iloc[test_index]

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_train = x_scaler.fit_transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)
            y_train = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
            y_test = y_test_raw.values

            model = train_mlp(
                X_train,
                y_train,
                epochs=min(options.get("mlp_epochs", 300), 120),
                lr=options.get("mlp_learning_rate", 0.001),
                patience=min(options.get("mlp_patience", 25), 15),
            )
            y_pred_scaled = predict_test_values(model, model_name, X_test)
            y_pred = inverse_target_values(y_scaler, y_pred_scaled)
            scores.append(r2_score(y_test, y_pred))

        return {"r2_mean": round(float(np.mean(scores)), 4), "r2_std": round(float(np.std(scores)), 4)}

    return None


def json_dump(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def json_load(value, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def safe_hyperparameter_search(model_name: str, X_train, y_train, options: dict, mode: str):
    from app.ml.hyperparameter_search import run_hyperparameter_search

    mode = normalize_search_mode(mode)
    try:
        return run_hyperparameter_search(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            options=options,
            mode=mode,
        )
    except Exception as exc:
        current_app.logger.exception("Hyperparameter search failed")
        return {
            "enabled": False,
            "mode": mode,
            "mode_title": SEARCH_MODE_OPTIONS[mode]["title"],
            "best_parameters": {},
            "candidates": [],
            "warning": f"Автоподбор не выполнен: {exc}. Используются параметры из формы.",
        }


def add_training_protocol_step(protocol: list[dict], title: str, detail: str, started_at: float, status: str = "success"):
    finished_at = time.perf_counter()
    protocol.append({
        "title": title,
        "detail": detail,
        "status": status,
        "duration_seconds": round(finished_at - started_at, 3),
    })
    return finished_at


def make_version_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def save_model_bundle(model_name: str, model, x_scaler, y_scaler, model_dir: str, version_id: str | None = None):
    import joblib

    os.makedirs(model_dir, exist_ok=True)
    version_id = version_id or make_version_id(model_name)
    artifact_dir = os.path.join(model_dir, "versions", version_id)
    os.makedirs(artifact_dir, exist_ok=True)

    if model_name in {"linear", "knn"}:
        active_model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        versioned_model_path = os.path.join(artifact_dir, f"{model_name}_model.pkl")
        joblib.dump(model, active_model_path)
        joblib.dump(model, versioned_model_path)
    else:
        import torch

        active_model_path = os.path.join(model_dir, "mlp_model.pt")
        versioned_model_path = os.path.join(artifact_dir, "mlp_model.pt")
        torch.save(model.state_dict(), active_model_path)
        torch.save(model.state_dict(), versioned_model_path)

    active_x_scaler_path = os.path.join(model_dir, f"{model_name}_x_scaler.pkl")
    active_y_scaler_path = os.path.join(model_dir, f"{model_name}_y_scaler.pkl")
    versioned_x_scaler_path = os.path.join(artifact_dir, f"{model_name}_x_scaler.pkl")
    versioned_y_scaler_path = os.path.join(artifact_dir, f"{model_name}_y_scaler.pkl")

    joblib.dump(x_scaler, active_x_scaler_path)
    joblib.dump(y_scaler, active_y_scaler_path)
    joblib.dump(x_scaler, versioned_x_scaler_path)
    joblib.dump(y_scaler, versioned_y_scaler_path)

    return {
        "artifact_dir": artifact_dir,
        "model_path": versioned_model_path,
        "x_scaler_path": versioned_x_scaler_path,
        "y_scaler_path": versioned_y_scaler_path,
    }


def model_artifacts_exist(model_name: str, model_dir: str) -> bool:
    if model_name in {"linear", "knn"}:
        model_filename = f"{model_name}_model.pkl"
    elif model_name == "mlp":
        model_filename = "mlp_model.pt"
    else:
        return False

    required_files = [
        os.path.join(model_dir, model_filename),
        os.path.join(model_dir, f"{model_name}_x_scaler.pkl"),
        os.path.join(model_dir, f"{model_name}_y_scaler.pkl"),
    ]
    return all(os.path.exists(path) for path in required_files)


def build_predict_model_summaries():
    summaries = {
        key: {
            "key": key,
            "title": title,
            "is_trained": False,
            "is_ready": False,
            "created_at": None,
            "metrics_r2": None,
            "metrics_mae": None,
            "metrics_rmse": None,
            "cv_r2_mean": None,
            "verdict": None,
            "verdict_label": None,
            "status_text": "Нет сохраненного обучения",
        }
        for key, title in MODEL_OPTIONS.items()
    }

    rows = get_training_history(current_app.config["DATABASE_PATH"])
    for row in rows:
        model_name = row["model_name"]
        if model_name not in summaries or summaries[model_name]["is_trained"]:
            continue

        adequacy_report = json_load(row["adequacy_report"], {})
        has_artifacts = model_artifacts_exist(model_name, current_app.config["MODEL_DIR"])
        summaries[model_name].update({
            "is_trained": True,
            "is_ready": has_artifacts,
            "created_at": row["created_at"],
            "metrics_r2": row["metrics_r2"],
            "metrics_mae": row["metrics_mae"],
            "metrics_rmse": row["metrics_rmse"],
            "cv_r2_mean": row["cv_r2_mean"],
            "verdict": adequacy_report.get("verdict"),
            "verdict_label": adequacy_report.get("verdict_label"),
            "status_text": "Готова к прогнозу" if has_artifacts else "Нет активных артефактов",
        })

    return summaries


def render_predict_template(**context):
    model_summaries = build_predict_model_summaries()
    default_model = next(
        (key for key, summary in model_summaries.items() if summary["is_ready"]),
        next(iter(MODEL_OPTIONS)),
    )
    defaults = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "feature_labels": FEATURE_LABELS,
        "feature_help": FEATURE_HELP,
        "feature_input_rules": FEATURE_INPUT_RULES,
        "feature_units": FEATURE_UNITS,
        "model_options": MODEL_OPTIONS,
        "model_summaries": model_summaries,
        "has_trained_models": any(summary["is_ready"] for summary in model_summaries.values()),
        "selected_model": default_model,
    }
    defaults.update(context)
    return render_template("predict.html", **defaults)


@main_bp.route("/")
def index():
    if not session.get("user_id"):
        return render_template("index.html")

    db_path = current_app.config["DATABASE_PATH"]
    return render_template(
        "index.html",
        stats=get_admin_dashboard_stats(db_path),
        model_summaries=build_predict_model_summaries(),
        training_stats=get_training_summary_stats(db_path),
    )


@main_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")
        preset_dataset = request.form.get("preset_dataset", "").strip()
        model_name = request.form.get("model_name", "").strip()
        action = request.form.get("action", "train")

        if model_name not in MODEL_OPTIONS:
            return render_template(
                "upload.html",
                error="Выберите корректную модель.",
                preset_datasets=PRESET_DATASETS,
                model_options=MODEL_OPTIONS,
                search_mode_options=SEARCH_MODE_OPTIONS,
                selected_model=model_name,
            )

        try:
            from app.ml.preprocess import build_dataset_profile, read_dataset, validate_dataset

            training_protocol = []
            step_started_at = time.perf_counter()
            options = parse_training_options(request.form)

            file_path, source_name = resolve_dataset(
                request_file=file,
                preset_dataset=preset_dataset,
                data_dir=current_app.config["DATA_DIR"],
            )

            raw_df = read_dataset(file_path)
            dataset_profile = build_dataset_profile(raw_df)
            step_started_at = add_training_protocol_step(
                training_protocol,
                "Загрузка и первичный анализ данных",
                f"{source_name}: {dataset_profile['row_count']} строк, {dataset_profile['column_count']} столбцов.",
                step_started_at,
            )

            if action == "preview":
                return render_template(
                    "upload.html",
                    preset_datasets=PRESET_DATASETS,
                    model_options=MODEL_OPTIONS,
                    search_mode_options=SEARCH_MODE_OPTIONS,
                    selected_model=model_name,
                    selected_knn_neighbors=options["knn_neighbors"],
                    selected_knn_weights=options["knn_weights"],
                    selected_knn_p=options["knn_p"],
                    selected_mlp_epochs=options["mlp_epochs"],
                    selected_mlp_patience=options["mlp_patience"],
                    selected_hyperparameter_search=options["hyperparameter_search"],
                    dataset_profile=dataset_profile,
                    source_name=source_name,
                )

            if not dataset_profile["is_valid"]:
                return render_template(
                    "upload.html",
                    error="Датасет содержит ошибки. Исправьте их перед обучением.",
                    preset_datasets=PRESET_DATASETS,
                    model_options=MODEL_OPTIONS,
                    search_mode_options=SEARCH_MODE_OPTIONS,
                    selected_model=model_name,
                    selected_knn_neighbors=options["knn_neighbors"],
                    selected_knn_weights=options["knn_weights"],
                    selected_knn_p=options["knn_p"],
                    selected_mlp_epochs=options["mlp_epochs"],
                    selected_mlp_patience=options["mlp_patience"],
                    selected_hyperparameter_search=options["hyperparameter_search"],
                    dataset_profile=dataset_profile,
                    source_name=source_name,
                )

            from app.charts import build_loss_chart, build_prediction_diagnostics
            from app.ml.evaluation import build_adequacy_report, build_baseline_metrics, build_residual_report
            from app.ml.metrics import calculate_regression_metrics
            from app.ml.preprocess import prepare_data

            df = validate_dataset(raw_df)
            issue_count = len(dataset_profile.get("issues", []))
            step_started_at = add_training_protocol_step(
                training_protocol,
                "Валидация датасета",
                f"Обязательные столбцы найдены, числовые значения приведены. Предупреждений: {issue_count}.",
                step_started_at,
            )
            X_train, X_test, y_train, y_test, x_scaler, y_scaler = prepare_data(df)
            if model_name == "knn":
                options["knn_neighbors"] = max(1, min(options["knn_neighbors"], len(X_train)))
            step_started_at = add_training_protocol_step(
                training_protocol,
                "Разделение и масштабирование",
                f"Train: {len(y_train)} строк, test: {len(y_test)} строк. Скейлеры обучены только на train.",
                step_started_at,
            )

            hyperparameter_search_report = safe_hyperparameter_search(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                options=options,
                mode=options["hyperparameter_search"],
            )
            if hyperparameter_search_report.get("best_parameters"):
                options.update(hyperparameter_search_report["best_parameters"])

            if hyperparameter_search_report.get("enabled"):
                search_detail = (
                    f"{hyperparameter_search_report['mode_title']}: "
                    f"{hyperparameter_search_report['successful_candidates_count']} из "
                    f"{hyperparameter_search_report['candidates_count']} кандидатов, "
                    f"best validation R2={hyperparameter_search_report['best_score']}."
                )
                search_status = "success"
            elif hyperparameter_search_report.get("warning"):
                search_detail = hyperparameter_search_report["warning"]
                search_status = "warning"
            else:
                search_detail = "Автоподбор отключен, используются параметры из формы."
                search_status = "success"

            step_started_at = add_training_protocol_step(
                training_protocol,
                "Подбор гиперпараметров",
                search_detail,
                step_started_at,
                status=search_status,
            )

            model, loss_history = train_model(
                model_name,
                X_train,
                y_train,
                options=options,
                validation_data=(X_test, y_test),
            )
            if loss_history:
                train_loss = loss_history.get("train_loss", [])
                val_loss = loss_history.get("val_loss", [])
                loss_detail = f"Обучение завершено за {len(train_loss)} эпох."
                if val_loss:
                    loss_detail += f" Последний validation loss: {round(float(val_loss[-1]), 6)}."
            else:
                loss_detail = "Модель обучена без итерационной loss-истории."
            step_started_at = add_training_protocol_step(
                training_protocol,
                "Обучение модели",
                loss_detail,
                step_started_at,
            )
            y_train_pred_scaled = predict_test_values(model, model_name, X_train)
            y_pred_scaled = predict_test_values(model, model_name, X_test)
            y_train_original = inverse_target_values(y_scaler, y_train)
            y_test_original = inverse_target_values(y_scaler, y_test)
            y_train_pred_original = inverse_target_values(y_scaler, y_train_pred_scaled)
            y_pred_original = inverse_target_values(y_scaler, y_pred_scaled)
            train_metrics = calculate_regression_metrics(y_train_original, y_train_pred_original)
            metrics = calculate_regression_metrics(y_test_original, y_pred_original)
            baseline_metrics = build_baseline_metrics(y_train_original, y_test_original)
            residual_report = build_residual_report(y_test_original, y_pred_original)
            diagnostics = build_prediction_diagnostics(y_test_original, y_pred_original)
            step_started_at = add_training_protocol_step(
                training_protocol,
                "Расчет метрик и baseline",
                f"Test R2={metrics['r2']}, RMSE={metrics['rmse']}; baseline RMSE={baseline_metrics['rmse']}.",
                step_started_at,
            )
            cv_metrics = run_cross_validation(df, model_name, options)
            cv_detail = "Кросс-валидация пропущена."
            if cv_metrics:
                cv_detail = f"CV R2={cv_metrics['r2_mean']} ± {cv_metrics['r2_std']}."
            step_started_at = add_training_protocol_step(
                training_protocol,
                "Кросс-валидация",
                cv_detail,
                step_started_at,
                status="success" if cv_metrics else "warning",
            )
            adequacy_report = build_adequacy_report(
                test_metrics=metrics,
                train_metrics=train_metrics,
                baseline_metrics=baseline_metrics,
                cv_metrics=cv_metrics,
                residual_report=residual_report,
            )
            coefficients = build_linear_coefficients(model) if model_name == "linear" else None
            parameters = model_parameters(model_name, options)
            artifact_paths = save_model_bundle(
                model_name=model_name,
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                model_dir=current_app.config["MODEL_DIR"],
                version_id=make_version_id(model_name),
            )
            step_started_at = add_training_protocol_step(
                training_protocol,
                "Сохранение артефактов",
                f"Файлы модели и скейлеров сохранены в {artifact_paths['artifact_dir']}.",
                step_started_at,
            )
            add_training_protocol_step(
                training_protocol,
                "Формирование отчета",
                f"Вердикт: {adequacy_report['verdict_label']}. Отчет готов к просмотру в истории моделей.",
                step_started_at,
            )

            training_id = save_training_run(
                db_path=current_app.config["DATABASE_PATH"],
                model_name=model_name,
                model_title=MODEL_OPTIONS[model_name],
                dataset_name=source_name,
                rows_count=len(df),
                train_size=len(y_train),
                test_size=len(y_test),
                feature_columns=json_dump(FEATURE_COLUMNS),
                parameters=json_dump(parameters),
                metrics=metrics,
                cv_metrics=cv_metrics,
                artifact_paths=artifact_paths,
                coefficients=json_dump(coefficients),
                loss_history=json_dump(loss_history),
                dataset_report=json_dump(dataset_profile),
                train_metrics=json_dump(train_metrics),
                baseline_metrics=json_dump(baseline_metrics),
                residual_report=json_dump(residual_report),
                adequacy_report=json_dump(adequacy_report),
                training_protocol=json_dump(training_protocol),
                hyperparameter_search_report=json_dump(hyperparameter_search_report),
            )

            return render_template(
                "result.html",
                training_id=training_id,
                model_title=MODEL_OPTIONS[model_name],
                metrics=metrics,
                train_metrics=train_metrics,
                baseline_metrics=baseline_metrics,
                residual_report=residual_report,
                adequacy_report=adequacy_report,
                training_protocol=training_protocol,
                hyperparameter_search_report=hyperparameter_search_report,
                cv_metrics=cv_metrics,
                parameters=parameters,
                rows_count=len(df),
                train_size=len(y_train),
                test_size=len(y_test),
                columns=df.columns.tolist(),
                source_name=source_name,
                diagnostics=diagnostics,
                coefficients=coefficients,
                loss_chart=build_loss_chart(loss_history),
                loss_history=loss_history,
                dataset_profile=dataset_profile,
            )

        except Exception as exc:
            current_app.logger.exception("Dataset training failed")
            return render_template(
                "upload.html",
                error=f"Ошибка обработки файла: {exc}",
                preset_datasets=PRESET_DATASETS,
                model_options=MODEL_OPTIONS,
                selected_model=model_name,
                search_mode_options=SEARCH_MODE_OPTIONS,
            )

    return render_template(
        "upload.html",
        preset_datasets=PRESET_DATASETS,
        model_options=MODEL_OPTIONS,
        search_mode_options=SEARCH_MODE_OPTIONS,
        selected_knn_neighbors=5,
        selected_knn_weights="uniform",
        selected_knn_p=2,
        selected_mlp_epochs=300,
        selected_mlp_patience=25,
        selected_hyperparameter_search="none",
    )


@main_bp.route("/compare", methods=["GET", "POST"])
@login_required
def compare_models():
    if request.method == "POST":
        file = request.files.get("dataset")
        preset_dataset = request.form.get("preset_dataset", "").strip()

        try:
            from app.charts import build_model_comparison_chart
            from app.ml.metrics import calculate_regression_metrics
            from app.ml.preprocess import build_dataset_profile, prepare_data, read_dataset, validate_dataset

            options = parse_training_options(request.form)

            file_path, source_name = resolve_dataset(
                request_file=file,
                preset_dataset=preset_dataset,
                data_dir=current_app.config["DATA_DIR"],
            )

            raw_df = read_dataset(file_path)
            dataset_profile = build_dataset_profile(raw_df)
            if not dataset_profile["is_valid"]:
                return render_template(
                    "compare.html",
                    error="Датасет содержит ошибки. Исправьте их перед сравнением.",
                    preset_datasets=PRESET_DATASETS,
                    search_mode_options=SEARCH_MODE_OPTIONS,
                    selected_knn_neighbors=options["knn_neighbors"],
                    selected_knn_weights=options["knn_weights"],
                    selected_knn_p=options["knn_p"],
                    selected_mlp_epochs=options["mlp_epochs"],
                    selected_mlp_patience=options["mlp_patience"],
                    selected_hyperparameter_search=options["hyperparameter_search"],
                    dataset_profile=dataset_profile,
                    source_name=source_name,
                )

            df = validate_dataset(raw_df)
            X_train, X_test, y_train, y_test, x_scaler, y_scaler = prepare_data(df)
            y_test_original = inverse_target_values(y_scaler, y_test)

            results = []
            for model_name, model_title in MODEL_OPTIONS.items():
                model_options_for_training = dict(options)
                search_report = safe_hyperparameter_search(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    options=model_options_for_training,
                    mode=options["hyperparameter_search"],
                )
                if search_report.get("best_parameters"):
                    model_options_for_training.update(search_report["best_parameters"])

                model, _ = train_model(
                    model_name,
                    X_train,
                    y_train,
                    options=model_options_for_training,
                    validation_data=(X_test, y_test),
                )
                y_pred_scaled = predict_test_values(model, model_name, X_test)
                y_pred_original = inverse_target_values(y_scaler, y_pred_scaled)
                metrics = calculate_regression_metrics(y_test_original, y_pred_original)
                cv_metrics = run_cross_validation(df, model_name, options)
                results.append({
                    "key": model_name,
                    "model": model_title,
                    "cv_r2_mean": cv_metrics["r2_mean"] if cv_metrics else None,
                    "cv_r2_std": cv_metrics["r2_std"] if cv_metrics else None,
                    "search_enabled": search_report.get("enabled", False),
                    "search_candidates_count": search_report.get("successful_candidates_count", 0),
                    "search_best_score": search_report.get("best_score"),
                    **metrics,
                })

            best_model = max(results, key=lambda row: row["r2"])
            worst_rmse_model = max(results, key=lambda row: row["rmse"])
            comparison_chart = build_model_comparison_chart(results)

            return render_template(
                "compare.html",
                preset_datasets=PRESET_DATASETS,
                results=results,
                best_model=best_model,
                worst_rmse_model=worst_rmse_model,
                source_name=source_name,
                rows_count=len(df),
                columns=df.columns.tolist(),
                comparison_chart=comparison_chart,
                dataset_profile=dataset_profile,
                selected_knn_neighbors=options["knn_neighbors"],
                selected_knn_weights=options["knn_weights"],
                selected_knn_p=options["knn_p"],
                selected_mlp_epochs=options["mlp_epochs"],
                selected_mlp_patience=options["mlp_patience"],
                selected_hyperparameter_search=options["hyperparameter_search"],
                search_mode_options=SEARCH_MODE_OPTIONS,
            )

        except Exception as exc:
            current_app.logger.exception("Model comparison failed")
            return render_template(
                "compare.html",
                error=f"Ошибка обработки файла: {exc}",
                preset_datasets=PRESET_DATASETS,
                search_mode_options=SEARCH_MODE_OPTIONS,
            )

    return render_template(
        "compare.html",
        preset_datasets=PRESET_DATASETS,
        search_mode_options=SEARCH_MODE_OPTIONS,
        selected_knn_neighbors=5,
        selected_knn_weights="uniform",
        selected_knn_p=2,
        selected_mlp_epochs=300,
        selected_mlp_patience=25,
        selected_hyperparameter_search="none",
    )


@main_bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict_page():
    if request.method == "POST":
        model_name = request.form.get("model_name", "").strip()

        if model_name not in MODEL_OPTIONS:
            return render_predict_template(error="Выберите корректную модель.")

        try:
            from app.ml.model_loader import load_model_bundle
            from app.ml.predict import make_prediction

            input_data, entered_values, features_dict, actual_sale = parse_prediction_form(request.form)
            validate_prediction_inputs(features_dict, actual_sale)

            model, x_scaler, y_scaler = load_model_bundle(
                model_name=model_name,
                model_dir=current_app.config["MODEL_DIR"],
            )

            prediction = make_prediction(
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                input_data=input_data,
                model_name=model_name,
            )

            absolute_diff, percent_diff = calculate_prediction_error(prediction, actual_sale)
            error_summary = build_prediction_error_summary(absolute_diff, percent_diff)
            prediction_summary = build_prediction_summary(features_dict, prediction)

            save_prediction(
                db_path=current_app.config["DATABASE_PATH"],
                model_name=model_name,
                features=features_dict,
                prediction=float(prediction),
                actual_sale=float(actual_sale) if actual_sale is not None else None,
                absolute_diff=float(absolute_diff) if absolute_diff is not None else None,
                percent_diff=float(percent_diff) if percent_diff is not None else None,
            )

            return render_predict_template(
                prediction=round(prediction, 2),
                actual_sale=round(actual_sale, 2) if actual_sale is not None else None,
                absolute_diff=round(absolute_diff, 2) if absolute_diff is not None else None,
                percent_diff=round(percent_diff, 2) if percent_diff is not None else None,
                error_summary=error_summary,
                prediction_summary=prediction_summary,
                selected_model=model_name,
                selected_model_title=MODEL_OPTIONS[model_name],
                entered_values=entered_values,
            )

        except (FileNotFoundError, ValueError) as exc:
            entered_values = {}
            try:
                _, entered_values, _, _ = parse_prediction_form(request.form)
            except ValueError:
                entered_values = {feature: request.form.get(feature, "") for feature in FEATURE_COLUMNS}
            return render_predict_template(error=str(exc), selected_model=model_name, entered_values=entered_values)
        except Exception as exc:
            current_app.logger.exception("Prediction failed")
            return render_predict_template(
                error=f"Ошибка прогнозирования: {exc}",
                selected_model=model_name,
            )

    return render_predict_template()


@main_bp.route("/history")
@login_required
def prediction_history():
    rows = get_prediction_history(
        db_path=current_app.config["DATABASE_PATH"],
        limit=30,
    )
    from app.charts import build_prediction_history_charts

    stats = get_prediction_history_stats(current_app.config["DATABASE_PATH"])
    history_charts = build_prediction_history_charts(stats, MODEL_OPTIONS)
    return render_template(
        "history.html",
        rows=rows,
        model_options=MODEL_OPTIONS,
        history_charts=history_charts,
    )


@main_bp.route("/history/<int:prediction_id>")
@login_required
def prediction_detail(prediction_id):
    row = get_prediction_by_id(
        db_path=current_app.config["DATABASE_PATH"],
        prediction_id=prediction_id,
    )

    if row is None:
        return render_template("history_detail.html", error="Запись не найдена.")

    return render_template(
        "history_detail.html",
        row=row,
        model_title=MODEL_OPTIONS.get(row["model_name"], row["model_name"]),
    )


@main_bp.route("/history/<int:prediction_id>/delete", methods=["POST"])
@login_required
@role_required("admin")
def delete_prediction(prediction_id):
    deleted = delete_prediction_by_id(
        db_path=current_app.config["DATABASE_PATH"],
        prediction_id=prediction_id,
    )

    if deleted:
        flash("Запись прогноза удалена.", "success")
    else:
        flash("Запись прогноза не найдена.", "warning")

    return redirect(url_for("main.prediction_history"))


def decode_training_row(row):
    if row is None:
        return None

    return {
        "row": row,
        "test_metrics": {
            "r2": row["metrics_r2"],
            "mae": row["metrics_mae"],
            "rmse": row["metrics_rmse"],
        },
        "feature_columns": json_load(row["feature_columns"], []),
        "parameters": json_load(row["parameters"], {}),
        "coefficients": json_load(row["coefficients"], []),
        "loss_history": json_load(row["loss_history"], None),
        "dataset_report": json_load(row["dataset_report"], None),
        "train_metrics": json_load(row["train_metrics"], None),
        "baseline_metrics": json_load(row["baseline_metrics"], None),
        "residual_report": json_load(row["residual_report"], None),
        "adequacy_report": json_load(row["adequacy_report"], None),
        "training_protocol": json_load(row["training_protocol"], None),
        "hyperparameter_search_report": json_load(row["hyperparameter_search_report"], None),
    }


def build_training_history_view(rows):
    view_rows = []
    for row in rows:
        item = {key: row[key] for key in row.keys()}
        adequacy_report = json_load(row["adequacy_report"], {})
        baseline_metrics = json_load(row["baseline_metrics"], {})
        search_report = json_load(row["hyperparameter_search_report"], {})
        item["verdict"] = adequacy_report.get("verdict")
        item["verdict_label"] = adequacy_report.get("verdict_label")
        item["baseline_improvement_percent"] = adequacy_report.get("baseline_improvement_percent")
        item["baseline_rmse"] = baseline_metrics.get("rmse")
        item["search_enabled"] = search_report.get("enabled", False)
        item["search_mode_title"] = search_report.get("mode_title")
        view_rows.append(item)
    return view_rows


@main_bp.route("/models")
@login_required
def trained_models():
    rows = get_training_history(current_app.config["DATABASE_PATH"])
    stats = get_training_summary_stats(current_app.config["DATABASE_PATH"])
    return render_template("models.html", rows=build_training_history_view(rows), stats=stats)


@main_bp.route("/models/<int:training_id>")
@login_required
def trained_model_detail(training_id):
    from app.charts import build_loss_chart

    row = get_training_by_id(current_app.config["DATABASE_PATH"], training_id)
    decoded = decode_training_row(row)
    if decoded is None:
        return render_template("model_detail.html", error="Запись обучения не найдена.")

    return render_template(
        "model_detail.html",
        **decoded,
        loss_chart=build_loss_chart(decoded["loss_history"]),
    )


@main_bp.route("/models/export")
@login_required
def export_models_csv():
    rows = get_training_history(current_app.config["DATABASE_PATH"])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id",
        "created_at",
        "model_name",
        "model_title",
        "dataset_name",
        "rows_count",
        "train_size",
        "test_size",
        "metrics_r2",
        "metrics_mae",
        "metrics_rmse",
        "cv_r2_mean",
        "cv_r2_std",
        "adequacy_verdict",
        "baseline_improvement_percent",
        "hyperparameter_search",
        "parameters",
        "artifact_dir",
    ])

    for row in rows:
        adequacy_report = json_load(row["adequacy_report"], {})
        search_report = json_load(row["hyperparameter_search_report"], {})
        writer.writerow([
            row["id"],
            row["created_at"],
            row["model_name"],
            row["model_title"],
            row["dataset_name"],
            row["rows_count"],
            row["train_size"],
            row["test_size"],
            row["metrics_r2"],
            row["metrics_mae"],
            row["metrics_rmse"],
            row["cv_r2_mean"],
            row["cv_r2_std"],
            adequacy_report.get("verdict_label"),
            adequacy_report.get("baseline_improvement_percent"),
            search_report.get("mode_title"),
            row["parameters"],
            row["artifact_dir"],
        ])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    output.close()

    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="model_training_history.csv",
    )


@main_bp.route("/models/<int:training_id>/report.csv")
@login_required
def export_model_report_csv(training_id):
    row = get_training_by_id(current_app.config["DATABASE_PATH"], training_id)
    decoded = decode_training_row(row)
    if decoded is None:
        return render_template("model_detail.html", error="Запись обучения не найдена."), 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["field", "value"])
    writer.writerow(["id", row["id"]])
    writer.writerow(["created_at", row["created_at"]])
    writer.writerow(["model", row["model_title"]])
    writer.writerow(["dataset", row["dataset_name"]])
    writer.writerow(["rows_count", row["rows_count"]])
    writer.writerow(["train_size", row["train_size"]])
    writer.writerow(["test_size", row["test_size"]])
    writer.writerow(["r2", row["metrics_r2"]])
    writer.writerow(["mae", row["metrics_mae"]])
    writer.writerow(["rmse", row["metrics_rmse"]])
    writer.writerow(["cv_r2_mean", row["cv_r2_mean"]])
    writer.writerow(["cv_r2_std", row["cv_r2_std"]])
    writer.writerow(["parameters", row["parameters"]])
    writer.writerow(["artifact_dir", row["artifact_dir"]])
    if decoded["train_metrics"]:
        writer.writerow(["train_r2", decoded["train_metrics"].get("r2")])
        writer.writerow(["train_mae", decoded["train_metrics"].get("mae")])
        writer.writerow(["train_rmse", decoded["train_metrics"].get("rmse")])
    if decoded["baseline_metrics"]:
        writer.writerow(["baseline_r2", decoded["baseline_metrics"].get("r2")])
        writer.writerow(["baseline_mae", decoded["baseline_metrics"].get("mae")])
        writer.writerow(["baseline_rmse", decoded["baseline_metrics"].get("rmse")])
    if decoded["adequacy_report"]:
        writer.writerow(["adequacy_verdict", decoded["adequacy_report"].get("verdict_label")])
        writer.writerow(["adequacy_summary", decoded["adequacy_report"].get("summary")])
        writer.writerow(["baseline_improvement_percent", decoded["adequacy_report"].get("baseline_improvement_percent")])
    if decoded["hyperparameter_search_report"]:
        writer.writerow(["hyperparameter_search_mode", decoded["hyperparameter_search_report"].get("mode_title")])
        writer.writerow(["hyperparameter_search_enabled", decoded["hyperparameter_search_report"].get("enabled")])
        writer.writerow(["hyperparameter_search_best_score", decoded["hyperparameter_search_report"].get("best_score")])
    if decoded["residual_report"]:
        writer.writerow(["residual_mean", decoded["residual_report"].get("residual_mean")])
        writer.writerow(["p90_abs_error", decoded["residual_report"].get("p90_abs_error")])
        writer.writerow(["mape", decoded["residual_report"].get("mape")])

    if decoded["coefficients"]:
        writer.writerow([])
        writer.writerow(["feature", "coefficient", "abs_coefficient"])
        for item in decoded["coefficients"]:
            writer.writerow([item["feature"], item["coefficient"], item["abs_coefficient"]])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    output.close()

    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"model_training_{training_id}_report.csv",
    )


@main_bp.route("/models/<int:training_id>/report.pdf")
@login_required
def export_model_report_pdf(training_id):
    row = get_training_by_id(current_app.config["DATABASE_PATH"], training_id)
    decoded = decode_training_row(row)
    if decoded is None:
        return render_template("model_detail.html", error="Запись обучения не найдена."), 404

    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
    except ImportError:
        return render_template(
            "model_detail.html",
            error="Для PDF-отчёта требуется matplotlib из requirements.txt.",
            **decoded,
            loss_chart=None,
        ), 500

    lines = [
        f"Отчёт обучения модели #{row['id']}",
        f"Дата: {row['created_at']}",
        f"Модель: {row['model_title']}",
        f"Датасет: {row['dataset_name']}",
        f"Строки: {row['rows_count']} (train: {row['train_size']}, test: {row['test_size']})",
        f"R2: {row['metrics_r2']}",
        f"MAE: {row['metrics_mae']}",
        f"RMSE: {row['metrics_rmse']}",
        f"CV R2: {row['cv_r2_mean']} ± {row['cv_r2_std']}",
        f"Параметры: {decoded['parameters']}",
        f"Артефакты: {row['artifact_dir']}",
    ]

    if decoded["train_metrics"]:
        lines.extend([
            "",
            "Train/Test:",
            f"Train R2: {decoded['train_metrics'].get('r2')}, MAE: {decoded['train_metrics'].get('mae')}, RMSE: {decoded['train_metrics'].get('rmse')}",
            f"Test R2: {row['metrics_r2']}, MAE: {row['metrics_mae']}, RMSE: {row['metrics_rmse']}",
        ])

    if decoded["baseline_metrics"]:
        lines.extend([
            "",
            "Baseline:",
            f"Strategy: {decoded['baseline_metrics'].get('strategy')}",
            f"R2: {decoded['baseline_metrics'].get('r2')}, MAE: {decoded['baseline_metrics'].get('mae')}, RMSE: {decoded['baseline_metrics'].get('rmse')}",
        ])

    if decoded["adequacy_report"]:
        lines.extend([
            "",
            f"Вердикт: {decoded['adequacy_report'].get('verdict_label')}",
            decoded["adequacy_report"].get("summary") or "",
        ])

    if decoded["hyperparameter_search_report"]:
        lines.extend([
            "",
            "Автоподбор гиперпараметров:",
            f"Режим: {decoded['hyperparameter_search_report'].get('mode_title')}",
            f"Кандидатов: {decoded['hyperparameter_search_report'].get('successful_candidates_count')} / {decoded['hyperparameter_search_report'].get('candidates_count')}",
            f"Best validation R2: {decoded['hyperparameter_search_report'].get('best_score')}",
        ])

    if decoded["coefficients"]:
        lines.append("")
        lines.append("Коэффициенты линейной регрессии:")
        for item in decoded["coefficients"][:10]:
            lines.append(f"{item['label']}: {item['coefficient']}")

    mem = io.BytesIO()
    with PdfPages(mem) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "\n".join(lines), va="top", ha="left", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"model_training_{training_id}_report.pdf",
    )


@main_bp.route("/history/export")
@login_required
@role_required("admin")
def export_history_csv():
    rows = get_all_prediction_history(
        db_path=current_app.config["DATABASE_PATH"],
    )

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "id",
        "created_at",
        "model_name",
        "instrspending",
        "discount_value",
        "tvspending",
        "stockrate",
        "price",
        "radio",
        "onlineadsspending",
        "prediction",
        "actual_sale",
        "absolute_diff",
        "percent_diff",
    ])

    for row in rows:
        writer.writerow([
            row["id"],
            row["created_at"],
            row["model_name"],
            row["instrspending"],
            row["discount_value"],
            row["tvspending"],
            row["stockrate"],
            row["price"],
            row["radio"],
            row["onlineadsspending"],
            row["prediction"],
            row["actual_sale"],
            row["absolute_diff"],
            row["percent_diff"],
        ])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    output.close()

    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv",
    )


@main_bp.route("/predict/sample-row", methods=["GET"])
@login_required
def predict_sample_row():
    try:
        from app.ml.preprocess import load_data

        sample_path = os.path.join(
            current_app.config["DATA_DIR"],
            "samples",
            "market_data.csv",
        )

        if not os.path.exists(sample_path):
            return jsonify({
                "success": False,
                "error": "Файл market_data.csv не найден в data/samples.",
            }), 404

        df = load_data(sample_path)
        sample = df[FEATURE_COLUMNS + ["Sale"]].sample(n=1).iloc[0]

        return jsonify({
            "success": True,
            "row": {col: float(sample[col]) for col in FEATURE_COLUMNS},
            "actual_sale": float(sample["Sale"]),
        })

    except Exception as exc:
        current_app.logger.exception("Sample row loading failed")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения строки из датасета: {exc}",
        }), 500
