import os


def load_model_bundle(model_name: str, model_dir: str):
    import joblib

    x_scaler_path = os.path.join(model_dir, f"{model_name}_x_scaler.pkl")
    y_scaler_path = os.path.join(model_dir, f"{model_name}_y_scaler.pkl")

    if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
        raise FileNotFoundError(
            f"Не найдены scaler-файлы для модели {model_name}. "
            f"Сначала обучите модель на странице /upload."
        )

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    if model_name in ["linear", "knn"]:
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Не найден файл модели {model_name}. "
                f"Сначала обучите модель на странице /upload."
            )
        model = joblib.load(model_path)
        return model, x_scaler, y_scaler

    if model_name == "mlp":
        import torch
        from app.ml.train_models import MLPRegressorModel

        model_path = os.path.join(model_dir, "mlp_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Не найден файл модели MLP. Сначала обучите модель на странице /upload."
            )

        input_dim = x_scaler.n_features_in_
        model = MLPRegressorModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        return model, x_scaler, y_scaler

    raise ValueError("Неизвестная модель.")
