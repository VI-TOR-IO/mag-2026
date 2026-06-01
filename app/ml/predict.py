def make_prediction(model, x_scaler, y_scaler, input_data: list[float], model_name: str) -> float:
    import numpy as np

    arr = np.array(input_data, dtype=float).reshape(1, -1)
    arr_scaled = x_scaler.transform(arr)

    if model_name in ["linear", "knn"]:
        pred_scaled = model.predict(arr_scaled)
    elif model_name == "mlp":
        import torch

        with torch.no_grad():
            tensor_input = torch.tensor(arr_scaled, dtype=torch.float32)
            pred_scaled = model(tensor_input).numpy().ravel()
    else:
        raise ValueError("Неизвестная модель для прогнозирования.")

    prediction = y_scaler.inverse_transform(
        np.array(pred_scaled).reshape(-1, 1)
    )[0][0]

    return float(prediction)
