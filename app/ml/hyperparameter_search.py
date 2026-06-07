SEARCH_MODE_OPTIONS = {
    "none": {
        "title": "Без автоподбора",
        "description": "Использовать параметры из формы.",
    },
    "fast": {
        "title": "Быстрый поиск",
        "description": "Небольшая сетка параметров для оперативного подбора.",
    },
    "balanced": {
        "title": "Расширенный поиск",
        "description": "Больше кандидатов, дольше обучение, выше шанс найти устойчивую конфигурацию.",
    },
}


def normalize_search_mode(mode: str | None) -> str:
    if mode in SEARCH_MODE_OPTIONS:
        return mode
    return "none"


def _dedupe_candidates(candidates):
    seen = set()
    unique = []
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def build_candidate_grid(model_name: str, mode: str, options: dict, train_size: int):
    mode = normalize_search_mode(mode)
    if mode == "none":
        return []

    if model_name == "linear":
        candidates = [
            {"linear_estimator": "linear"},
            {"linear_estimator": "ridge", "alpha": 0.1},
            {"linear_estimator": "ridge", "alpha": 1.0},
            {"linear_estimator": "ridge", "alpha": 10.0},
        ]
        if mode == "balanced":
            candidates.extend([
                {"linear_estimator": "ridge", "alpha": 0.01},
                {"linear_estimator": "ridge", "alpha": 100.0},
            ])
        return candidates

    if model_name == "knn":
        manual_neighbors = options.get("knn_neighbors", 5)
        neighbor_values = [3, manual_neighbors, 7]
        if mode == "balanced":
            neighbor_values.extend([1, 11, 15])

        candidates = []
        for n_neighbors in neighbor_values:
            safe_neighbors = max(1, min(int(n_neighbors), max(1, train_size)))
            candidates.append({"knn_neighbors": safe_neighbors, "knn_weights": "uniform", "knn_p": 2})
            candidates.append({"knn_neighbors": safe_neighbors, "knn_weights": "distance", "knn_p": 2})
            if mode == "balanced":
                candidates.append({"knn_neighbors": safe_neighbors, "knn_weights": "distance", "knn_p": 1})
        return _dedupe_candidates(candidates)

    if model_name == "mlp":
        manual_epochs = options.get("mlp_epochs", 300)
        manual_patience = options.get("mlp_patience", 25)
        candidates = [
            {
                "mlp_epochs": min(manual_epochs, 160),
                "mlp_patience": min(manual_patience, 20),
                "mlp_learning_rate": 0.001,
            },
            {
                "mlp_epochs": min(max(manual_epochs, 120), 220),
                "mlp_patience": min(max(manual_patience, 15), 30),
                "mlp_learning_rate": 0.0005,
            },
        ]
        if mode == "balanced":
            candidates.extend([
                {
                    "mlp_epochs": min(max(manual_epochs, 180), 320),
                    "mlp_patience": min(max(manual_patience, 20), 40),
                    "mlp_learning_rate": 0.0015,
                },
                {
                    "mlp_epochs": min(max(manual_epochs, 220), 420),
                    "mlp_patience": min(max(manual_patience, 25), 50),
                    "mlp_learning_rate": 0.0008,
                },
            ])
        return _dedupe_candidates(candidates)

    return []


def _regression_metrics(y_true, y_pred):
    import numpy as np

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 if ss_tot == 0 and ss_res == 0 else 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot

    return {
        "r2": round(float(r2), 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


def _train_candidate(model_name: str, X_train, y_train, parameters: dict, validation_data=None):
    from app.ml.train_models import (
        train_knn,
        train_linear_regression,
        train_mlp,
        train_ridge_regression,
    )

    if model_name == "linear":
        estimator = parameters.get("linear_estimator", "linear")
        if estimator == "ridge":
            return train_ridge_regression(X_train, y_train, alpha=parameters.get("alpha", 1.0))
        return train_linear_regression(X_train, y_train)

    if model_name == "knn":
        return train_knn(
            X_train,
            y_train,
            n_neighbors=parameters.get("knn_neighbors", 5),
            weights=parameters.get("knn_weights", "uniform"),
            p=parameters.get("knn_p", 2),
        )

    if model_name == "mlp":
        return train_mlp(
            X_train,
            y_train,
            epochs=parameters.get("mlp_epochs", 160),
            lr=parameters.get("mlp_learning_rate", 0.001),
            validation_data=validation_data,
            patience=parameters.get("mlp_patience", 20),
        )

    raise ValueError("Неизвестная модель.")


def _predict_candidate(model, model_name: str, X):
    if model_name in {"linear", "knn"}:
        return model.predict(X)

    if model_name == "mlp":
        import torch

        model.eval()
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

    raise ValueError("Неизвестная модель.")


def run_hyperparameter_search(model_name: str, X_train, y_train, options: dict, mode: str):
    mode = normalize_search_mode(mode)
    if mode == "none":
        return {
            "enabled": False,
            "mode": mode,
            "mode_title": SEARCH_MODE_OPTIONS[mode]["title"],
            "best_parameters": {},
            "candidates": [],
        }

    if len(y_train) < 8:
        return {
            "enabled": False,
            "mode": mode,
            "mode_title": SEARCH_MODE_OPTIONS[mode]["title"],
            "best_parameters": {},
            "candidates": [],
            "warning": "Автоподбор пропущен: для внутреннего validation-разбиения нужно минимум 8 train-строк.",
        }

    from sklearn.model_selection import train_test_split

    X_search_train, X_val, y_search_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
    )

    candidates = build_candidate_grid(model_name, mode, options, train_size=len(y_search_train))
    results = []

    for index, candidate in enumerate(candidates, start=1):
        try:
            model = _train_candidate(
                model_name,
                X_search_train,
                y_search_train,
                candidate,
                validation_data=(X_val, y_val),
            )
            y_pred = _predict_candidate(model, model_name, X_val)
            metrics = _regression_metrics(y_val, y_pred)
            results.append({
                "index": index,
                "parameters": candidate,
                **metrics,
            })
        except Exception as exc:
            results.append({
                "index": index,
                "parameters": candidate,
                "error": str(exc),
            })

    successful = [item for item in results if "error" not in item]
    if not successful:
        return {
            "enabled": False,
            "mode": mode,
            "mode_title": SEARCH_MODE_OPTIONS[mode]["title"],
            "best_parameters": {},
            "candidates": results,
            "warning": "Автоподбор не нашел рабочую конфигурацию.",
        }

    successful = sorted(successful, key=lambda item: (item["r2"], -item["rmse"]), reverse=True)
    best = successful[0]

    return {
        "enabled": True,
        "mode": mode,
        "mode_title": SEARCH_MODE_OPTIONS[mode]["title"],
        "metric": "validation R²",
        "best_parameters": best["parameters"],
        "best_score": best["r2"],
        "best_rmse": best["rmse"],
        "candidates_count": len(results),
        "successful_candidates_count": len(successful),
        "candidates": sorted(results, key=lambda item: item.get("r2", -999), reverse=True),
    }
