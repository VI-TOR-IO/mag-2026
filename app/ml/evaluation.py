def _to_array(values):
    import numpy as np

    return np.asarray(values, dtype=float).ravel()


def _safe_number(value, digits: int = 4):
    import numpy as np

    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _safe_percent(value):
    number = _safe_number(value, 2)
    return number if number is not None else None


def _calculate_metrics(y_true, y_pred):
    import numpy as np

    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1 - ss_res / ss_tot

    return {
        "r2": round(float(r2), 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


def build_baseline_metrics(y_train, y_test):
    import numpy as np

    y_train = _to_array(y_train)
    y_test = _to_array(y_test)
    baseline_value = float(np.mean(y_train))
    baseline_prediction = np.full_like(y_test, baseline_value, dtype=float)
    metrics = _calculate_metrics(y_test, baseline_prediction)

    return {
        "strategy": "Среднее значение Sale на train-выборке",
        "prediction_value": round(baseline_value, 4),
        **metrics,
    }


def build_residual_report(y_true, y_pred):
    import numpy as np

    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    non_zero_mask = np.abs(y_true) > 1e-12
    mape = None

    if np.any(non_zero_mask):
        mape = float(np.mean(abs_residuals[non_zero_mask] / np.abs(y_true[non_zero_mask])) * 100)

    target_mean = float(np.mean(np.abs(y_true))) if len(y_true) else 0.0
    p90_abs_error = float(np.percentile(abs_residuals, 90)) if len(abs_residuals) else 0.0
    p90_share = (p90_abs_error / target_mean * 100) if target_mean else None

    return {
        "residual_mean": _safe_number(float(np.mean(residuals))) if len(residuals) else None,
        "residual_std": _safe_number(float(np.std(residuals))) if len(residuals) else None,
        "p50_abs_error": _safe_number(float(np.percentile(abs_residuals, 50))) if len(abs_residuals) else None,
        "p90_abs_error": _safe_number(p90_abs_error) if len(abs_residuals) else None,
        "max_abs_error": _safe_number(float(np.max(abs_residuals))) if len(abs_residuals) else None,
        "mape": _safe_percent(mape),
        "p90_abs_error_share": _safe_percent(p90_share),
    }


def build_adequacy_report(test_metrics, train_metrics, baseline_metrics, cv_metrics, residual_report):
    checks = []

    model_rmse = test_metrics.get("rmse")
    baseline_rmse = baseline_metrics.get("rmse")
    improvement = None
    if baseline_rmse and baseline_rmse > 0 and model_rmse is not None:
        improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100

    if improvement is None:
        checks.append({
            "status": "warning",
            "title": "Baseline",
            "detail": "Недостаточно данных, чтобы корректно сравнить RMSE с baseline.",
        })
    elif improvement >= 10:
        checks.append({
            "status": "success",
            "title": "Baseline",
            "detail": f"RMSE лучше baseline на {round(improvement, 2)}%.",
        })
    elif improvement > 0:
        checks.append({
            "status": "warning",
            "title": "Baseline",
            "detail": f"Модель лучше baseline только на {round(improvement, 2)}%.",
        })
    else:
        checks.append({
            "status": "error",
            "title": "Baseline",
            "detail": f"Модель хуже baseline на {round(abs(improvement), 2)}%.",
        })

    train_r2 = train_metrics.get("r2")
    test_r2 = test_metrics.get("r2")
    gap = None
    if train_r2 is not None and test_r2 is not None:
        gap = train_r2 - test_r2

    if test_r2 is None:
        checks.append({
            "status": "warning",
            "title": "Test R2",
            "detail": "R2 на test-выборке не рассчитан.",
        })
    elif test_r2 >= 0.65:
        checks.append({
            "status": "success",
            "title": "Test R2",
            "detail": f"R2={test_r2}; модель объясняет заметную часть вариации Sale.",
        })
    elif test_r2 > 0:
        checks.append({
            "status": "warning",
            "title": "Test R2",
            "detail": f"R2={test_r2}; модель полезнее константы, но запас качества ограничен.",
        })
    else:
        checks.append({
            "status": "error",
            "title": "Test R2",
            "detail": f"R2={test_r2}; модель на test-выборке хуже простого среднего.",
        })

    if gap is None:
        checks.append({
            "status": "warning",
            "title": "Train/Test",
            "detail": "Нельзя оценить разрыв между train и test.",
        })
    elif gap <= 0.15:
        checks.append({
            "status": "success",
            "title": "Train/Test",
            "detail": f"Разрыв R2={round(gap, 4)}; явного переобучения не видно.",
        })
    elif gap <= 0.3:
        checks.append({
            "status": "warning",
            "title": "Train/Test",
            "detail": f"Разрыв R2={round(gap, 4)}; возможны признаки переобучения.",
        })
    else:
        checks.append({
            "status": "error",
            "title": "Train/Test",
            "detail": f"Разрыв R2={round(gap, 4)}; высок риск переобучения.",
        })

    if cv_metrics:
        cv_mean = cv_metrics.get("r2_mean")
        cv_std = cv_metrics.get("r2_std")
        if cv_std is not None and cv_std <= 0.15 and cv_mean is not None and cv_mean > 0:
            checks.append({
                "status": "success",
                "title": "Cross-validation",
                "detail": f"CV R2={cv_mean} ± {cv_std}; оценка стабильна.",
            })
        elif cv_std is not None and cv_std <= 0.3:
            checks.append({
                "status": "warning",
                "title": "Cross-validation",
                "detail": f"CV R2={cv_mean} ± {cv_std}; стабильность средняя.",
            })
        else:
            checks.append({
                "status": "error",
                "title": "Cross-validation",
                "detail": f"CV R2={cv_mean} ± {cv_std}; качество сильно зависит от разбиения.",
            })
    else:
        checks.append({
            "status": "warning",
            "title": "Cross-validation",
            "detail": "Кросс-валидация пропущена: слишком мало строк или модель не поддержана.",
        })

    p90_share = residual_report.get("p90_abs_error_share")
    if p90_share is None:
        checks.append({
            "status": "warning",
            "title": "Остатки",
            "detail": "Недостаточно данных для относительной оценки крупных ошибок.",
        })
    elif p90_share <= 20:
        checks.append({
            "status": "success",
            "title": "Остатки",
            "detail": f"90% ошибок не превышают {p90_share}% от среднего Sale.",
        })
    elif p90_share <= 40:
        checks.append({
            "status": "warning",
            "title": "Остатки",
            "detail": f"90% ошибок не превышают {p90_share}% от среднего Sale; точность умеренная.",
        })
    else:
        checks.append({
            "status": "error",
            "title": "Остатки",
            "detail": f"Крупные ошибки высоки: p90={p90_share}% от среднего Sale.",
        })

    error_count = sum(1 for check in checks if check["status"] == "error")
    warning_count = sum(1 for check in checks if check["status"] == "warning")

    if error_count:
        verdict = "bad"
        verdict_label = "Нужна доработка"
        summary = "Модель нельзя считать надежной без дополнительной настройки или улучшения данных."
    elif warning_count:
        verdict = "warning"
        verdict_label = "Условно пригодна"
        summary = "Модель можно использовать осторожно, но есть ограничения качества."
    else:
        verdict = "good"
        verdict_label = "Адекватна"
        summary = "Модель проходит базовые проверки качества и стабильности."

    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "summary": summary,
        "baseline_improvement_percent": _safe_percent(improvement),
        "train_test_r2_gap": _safe_number(gap),
        "checks": checks,
    }
