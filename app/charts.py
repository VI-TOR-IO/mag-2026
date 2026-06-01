def _as_float_list(values) -> list[float]:
    return [float(value) for value in values]


def _format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def _sample_pairs(actual_values: list[float], predicted_values: list[float], max_points: int):
    pairs = list(zip(actual_values, predicted_values))
    if len(pairs) <= max_points:
        return pairs

    step = (len(pairs) - 1) / (max_points - 1)
    indexes = sorted({round(index * step) for index in range(max_points)})
    return [pairs[index] for index in indexes]


def build_prediction_diagnostics(y_true, y_pred, max_points: int = 90, bins_count: int = 8):
    actual_values = _as_float_list(y_true)
    predicted_values = _as_float_list(y_pred)
    if not actual_values or not predicted_values:
        return None

    sampled_pairs = _sample_pairs(actual_values, predicted_values, max_points=max_points)

    all_values = actual_values + predicted_values
    min_value = min(all_values)
    max_value = max(all_values)
    padding = (max_value - min_value) * 0.06 or 1
    axis_min = min_value - padding
    axis_max = max_value + padding

    plot_left = 52
    plot_right = 390
    plot_top = 24
    plot_bottom = 220

    points = []
    for actual, predicted in sampled_pairs:
        points.append({
            "x": round(_scale(actual, axis_min, axis_max, plot_left, plot_right), 2),
            "y": round(_scale(predicted, axis_min, axis_max, plot_bottom, plot_top), 2),
            "actual": _format_number(actual),
            "predicted": _format_number(predicted),
        })

    residuals = [predicted - actual for actual, predicted in zip(actual_values, predicted_values)]
    max_abs_residual = max(abs(value) for value in residuals) or 1
    hist_min = -max_abs_residual
    hist_max = max_abs_residual
    bin_width = (hist_max - hist_min) / bins_count
    counts = [0 for _ in range(bins_count)]

    for residual in residuals:
        index = int((residual - hist_min) / bin_width) if bin_width else 0
        index = min(max(index, 0), bins_count - 1)
        counts[index] += 1

    max_count = max(counts) or 1
    histogram_left = 46
    histogram_bottom = 198
    histogram_height = 150
    bar_gap = 6
    bar_width = (330 - (bins_count - 1) * bar_gap) / bins_count

    bins = []
    for index, count in enumerate(counts):
        height = (count / max_count) * histogram_height
        x = histogram_left + index * (bar_width + bar_gap)
        y = histogram_bottom - height
        start = hist_min + index * bin_width
        end = start + bin_width
        bins.append({
            "x": round(x, 2),
            "y": round(y, 2),
            "width": round(bar_width, 2),
            "height": round(height, 2),
            "count": count,
            "label": f"{_format_number(start)}..{_format_number(end)}",
        })

    avg_abs_error = sum(abs(value) for value in residuals) / len(residuals)

    return {
        "scatter": {
            "points": points,
            "axis_min": _format_number(axis_min),
            "axis_max": _format_number(axis_max),
            "ideal_line": {
                "x1": plot_left,
                "y1": plot_bottom,
                "x2": plot_right,
                "y2": plot_top,
            },
        },
        "histogram": {
            "bins": bins,
            "max_count": max_count,
            "max_abs_residual": _format_number(max_abs_residual),
        },
        "summary": {
            "points_count": len(actual_values),
            "shown_points_count": len(points),
            "avg_abs_error": _format_number(avg_abs_error),
            "max_abs_error": _format_number(max_abs_residual),
        },
    }


def build_model_comparison_chart(results: list[dict]):
    if not results:
        return None

    chart_width = 720
    label_x = 18
    bar_x = 210
    bar_max_width = 390
    value_x = 625
    row_height = 44
    bar_height = 18

    r2_bars = []
    for index, row in enumerate(results):
        y = 34 + index * row_height
        value = float(row["r2"])
        width = max(0, min(value, 1)) * bar_max_width
        r2_bars.append({
            "key": row["key"],
            "label": row["model"],
            "value": f"{value:.4f}",
            "x": bar_x,
            "y": y,
            "width": round(width, 2),
            "height": bar_height,
            "label_x": label_x,
            "label_y": y + 14,
            "value_x": value_x,
            "value_y": y + 14,
        })

    error_rows = []
    max_mae = max(float(row["mae"]) for row in results) or 1
    max_rmse = max(float(row["rmse"]) for row in results) or 1

    for index, row in enumerate(results):
        y = 34 + index * 58
        mae = float(row["mae"])
        rmse = float(row["rmse"])
        error_rows.append({
            "key": row["key"],
            "label": row["model"],
            "label_x": label_x,
            "label_y": y + 22,
            "mae": {
                "value": _format_number(mae),
                "x": bar_x,
                "y": y,
                "width": round((mae / max_mae) * bar_max_width, 2),
                "height": 16,
            },
            "rmse": {
                "value": _format_number(rmse),
                "x": bar_x,
                "y": y + 22,
                "width": round((rmse / max_rmse) * bar_max_width, 2),
                "height": 16,
            },
            "value_x": value_x,
        })

    return {
        "width": chart_width,
        "r2_height": 58 + len(results) * row_height,
        "error_height": 62 + len(results) * 58,
        "r2_bars": r2_bars,
        "error_rows": error_rows,
    }


def build_loss_chart(loss_history: dict | None, max_points: int = 120):
    if not loss_history:
        return None

    train_loss = _as_float_list(loss_history.get("train_loss", []))
    val_loss = _as_float_list(loss_history.get("val_loss", []))
    if not train_loss:
        return None

    def sample(values):
        if len(values) <= max_points:
            return list(enumerate(values, start=1))
        step = (len(values) - 1) / (max_points - 1)
        indexes = sorted({round(index * step) for index in range(max_points)})
        return [(index + 1, values[index]) for index in indexes]

    sampled_train = sample(train_loss)
    sampled_val = sample(val_loss) if val_loss else []
    all_values = train_loss + val_loss
    min_value = min(all_values)
    max_value = max(all_values)
    padding = (max_value - min_value) * 0.08 or 0.01
    axis_min = max(0, min_value - padding)
    axis_max = max_value + padding

    plot_left = 48
    plot_right = 390
    plot_top = 24
    plot_bottom = 190
    max_epoch = max(len(train_loss), len(val_loss), 1)

    def build_points(points):
        return " ".join(
            f"{round(_scale(epoch, 1, max_epoch, plot_left, plot_right), 2)},"
            f"{round(_scale(value, axis_min, axis_max, plot_bottom, plot_top), 2)}"
            for epoch, value in points
        )

    return {
        "train_points": build_points(sampled_train),
        "val_points": build_points(sampled_val) if sampled_val else None,
        "axis_min": _format_number(axis_min),
        "axis_max": _format_number(axis_max),
        "max_epoch": max_epoch,
        "best_epoch": loss_history.get("best_epoch"),
        "stopped_epoch": loss_history.get("stopped_epoch"),
    }


def build_prediction_history_charts(stats: dict, model_titles: dict):
    error_rows = stats.get("error_rows", [])
    model_counts = stats.get("model_counts", [])
    avg_errors = stats.get("avg_errors", [])

    trend = None
    if error_rows:
        values = [float(row["percent_diff"]) for row in error_rows]
        max_points = 50
        if len(values) > max_points:
            step = (len(values) - 1) / (max_points - 1)
            indexes = sorted({round(index * step) for index in range(max_points)})
            sampled = [(index + 1, values[index]) for index in indexes]
        else:
            sampled = list(enumerate(values, start=1))

        max_value = max(values) or 1
        plot_left = 42
        plot_right = 390
        plot_top = 24
        plot_bottom = 170
        points = " ".join(
            f"{round(_scale(index, 1, len(values), plot_left, plot_right), 2)},"
            f"{round(_scale(value, 0, max_value, plot_bottom, plot_top), 2)}"
            for index, value in sampled
        )
        trend = {
            "points": points,
            "max_value": _format_number(max_value),
            "count": len(values),
        }

    def build_bar_rows(rows, value_key, max_width=220):
        if not rows:
            return []

        max_value = max(float(row[value_key]) for row in rows) or 1
        bar_rows = []
        for row in rows:
            model_name = row["model_name"]
            value = float(row[value_key])
            bar_rows.append({
                "label": model_titles.get(model_name, model_name),
                "value": _format_number(value),
                "width": round((value / max_value) * max_width, 2),
            })
        return bar_rows

    return {
        "trend": trend,
        "model_counts": build_bar_rows(model_counts, "cnt"),
        "avg_errors": build_bar_rows(avg_errors, "avg_percent_diff"),
    }
