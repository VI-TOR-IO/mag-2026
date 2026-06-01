from app.constants import FEATURE_COLUMNS

TARGET_COLUMN = "Sale"
REQUIRED_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS


def read_dataset(file_path: str):
    import pandas as pd

    return pd.read_csv(file_path)


def _numeric_series(series):
    import pandas as pd

    normalized = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    normalized = normalized.replace({"": None, "nan": None, "None": None})
    return pd.to_numeric(normalized, errors="coerce")


def validate_dataset(df, target_column: str = TARGET_COLUMN):
    required_columns = [target_column] + FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют столбцы: {', '.join(missing_columns)}")

    validated = df[required_columns].copy()

    for column in required_columns:
        validated[column] = _numeric_series(validated[column])

    if validated.isna().any().any():
        raise ValueError("Датасет содержит пустые или некорректные числовые значения.")

    return validated


def load_data(file_path: str):
    return validate_dataset(read_dataset(file_path))


def build_dataset_profile(df, target_column: str = TARGET_COLUMN, preview_rows_count: int = 10):
    required_columns = [target_column] + FEATURE_COLUMNS
    row_count = int(len(df))
    missing_columns = [column for column in required_columns if column not in df.columns]
    issues = []
    column_profiles = []

    for column in missing_columns:
        issues.append({
            "severity": "error",
            "message": f"Отсутствует обязательный столбец {column}.",
        })

    for column in required_columns:
        if column not in df.columns:
            column_profiles.append({
                "name": column,
                "dtype": "-",
                "missing": "-",
                "non_numeric": "-",
                "min": "-",
                "max": "-",
                "mean": "-",
            })
            continue

        raw = df[column]
        numeric = _numeric_series(raw)
        blank_mask = raw.isna() | raw.astype(str).str.strip().eq("")
        non_numeric_count = int(numeric.isna().sum() - blank_mask.sum())
        missing_count = int(blank_mask.sum())

        profile = {
            "name": column,
            "dtype": str(raw.dtype),
            "missing": missing_count,
            "non_numeric": max(non_numeric_count, 0),
            "min": round(float(numeric.min()), 4) if not numeric.dropna().empty else "-",
            "max": round(float(numeric.max()), 4) if not numeric.dropna().empty else "-",
            "mean": round(float(numeric.mean()), 4) if not numeric.dropna().empty else "-",
        }
        column_profiles.append(profile)

        if missing_count:
            issues.append({
                "severity": "error",
                "message": f"В столбце {column} есть пустые значения: {missing_count}.",
            })

        if non_numeric_count:
            issues.append({
                "severity": "error",
                "message": f"В столбце {column} есть нечисловые значения: {non_numeric_count}.",
            })

    if row_count == 0:
        issues.append({"severity": "error", "message": "Датасет не содержит строк."})

    def add_range_issue(column: str, mask, message: str, severity: str = "warning"):
        count = int(mask.sum())
        if count:
            issues.append({"severity": severity, "message": f"{column}: {message}: {count}."})

    if "Price" in df.columns:
        price = _numeric_series(df["Price"])
        add_range_issue("Price", price < 0, "Отрицательная цена", severity="error")

    if "Sale" in df.columns:
        sale = _numeric_series(df["Sale"])
        add_range_issue("Sale", sale < 0, "Отрицательное значение Sale", severity="warning")

    for ratio_column in ["Discount", "StockRate"]:
        if ratio_column in df.columns:
            values = _numeric_series(df[ratio_column])
            add_range_issue(
                ratio_column,
                (values < 0) | (values > 1),
                f"Значения {ratio_column} вне диапазона 0..1",
                severity="error",
            )

    for spend_column in ["InStrSpending", "TVSpending", "Radio", "OnlineAdsSpending"]:
        if spend_column in df.columns:
            values = _numeric_series(df[spend_column])
            add_range_issue(
                spend_column,
                values < 0,
                f"Отрицательные расходы в {spend_column}",
                severity="warning",
            )

    preview_df = df.head(preview_rows_count).copy()
    preview_df = preview_df.where(preview_df.notna(), "")

    return {
        "row_count": row_count,
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "required_columns": required_columns,
        "preview_columns": list(preview_df.columns),
        "preview_rows": preview_df.astype(str).to_dict(orient="records"),
        "column_profiles": column_profiles,
        "issues": issues,
        "is_valid": not any(issue["severity"] == "error" for issue in issues),
    }


def prepare_data(df, target_column: str = TARGET_COLUMN):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = validate_dataset(df, target_column=target_column)

    X = df[FEATURE_COLUMNS]
    y = df[target_column]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test_raw.values.reshape(-1, 1)).ravel()

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler
