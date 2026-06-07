import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train, alpha: float = 1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors: int = 5, weights: str = "uniform", p: int = 2):
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    model.fit(X_train, y_train)
    return model


class MLPRegressorModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_mlp(
    X_train,
    y_train,
    epochs: int = 300,
    lr: float = 0.001,
    validation_data=None,
    patience: int = 25,
    min_delta: float = 1e-5,
    seed: int = 42,
    return_history: bool = False,
):
    torch.manual_seed(seed)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    if validation_data is not None:
        X_val, y_val = validation_data
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    else:
        X_val_tensor = None
        y_val_tensor = None

    model = MLPRegressorModel(input_dim=X_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "best_epoch": None, "stopped_epoch": None}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

        train_loss = float(loss.detach().item())
        history["train_loss"].append(round(train_loss, 8))

        if X_val_tensor is None:
            continue

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_tensor), y_val_tensor).item())

        history["val_loss"].append(round(val_loss, 8))

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            history["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            history["stopped_epoch"] = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if history["best_epoch"] is None:
        history["best_epoch"] = len(history["train_loss"])

    if history["stopped_epoch"] is None:
        history["stopped_epoch"] = len(history["train_loss"])

    if return_history:
        return model, history

    return model
