"""Small PyTorch models behind a scikit-learn-style interface.

Supports early stopping, a learning-rate scheduler, seeded training and device
selection (CUDA/MPS only if available; CPU is the default and the tested path).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..device import get_device
from ..seed import seed_everything

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "mlp": {
        "hidden_dim": 64,
        "n_layers": 2,
        "dropout": 0.2,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "epochs": 60,
        "batch_size": 64,
        "patience": 10,
    },
    "cnn": {
        "hidden_dim": 16,
        "dropout": 0.1,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "epochs": 40,
        "batch_size": 64,
        "patience": 8,
    },
}


class _MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        modules: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, layers)):
            modules += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden
        modules.append(nn.Linear(dim, n_classes))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SmallCNN(nn.Module):
    def __init__(self, n_classes: int, hidden: int, dropout: float, image_shape: tuple[int, int]):
        super().__init__()
        h, w = image_shape
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = hidden * max(1, h // 4) * max(1, w // 4)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TorchClassifier:
    """sklearn-like wrapper: ``fit``, ``predict``, ``predict_proba``."""

    def __init__(
        self,
        name: str = "mlp",
        seed: int = 0,
        params: dict[str, Any] | None = None,
        input_dim: int | None = None,
        n_classes: int | None = None,
        image_shape: tuple[int, int] | None = None,
        device: str | None = None,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        if name not in DEFAULT_PARAMS:
            raise KeyError(name)
        self.name = name
        self.seed = seed
        self.params = {**DEFAULT_PARAMS[name], **(params or {})}
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.image_shape = tuple(image_shape) if image_shape else None
        self.device = device or get_device()
        self.verbose = verbose
        self.model_: nn.Module | None = None
        self.scaler_: StandardScaler | None = None
        self.history_: list[dict[str, float]] = []
        self.best_epoch_: int = 0
        self.classes_: np.ndarray | None = None

    # -- helpers ---------------------------------------------------------
    def _prepare(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if self.name != "cnn" and self.scaler_ is not None:
            arr = self.scaler_.transform(arr)
        return arr

    def _to_features(self, X: np.ndarray) -> torch.Tensor:
        arr = np.asarray(self._prepare(X), dtype=np.float32)
        if self.name == "cnn":
            if self.image_shape is None:
                side = int(round(np.sqrt(arr.shape[1])))
                self.image_shape = (side, side)
            arr = arr.reshape(-1, 1, *self.image_shape) / 16.0
        return torch.from_numpy(arr)

    def _build(self, input_dim: int, n_classes: int) -> nn.Module:
        p = self.params
        if self.name == "cnn":
            return _SmallCNN(
                n_classes=n_classes,
                hidden=int(p["hidden_dim"]),
                dropout=float(p["dropout"]),
                image_shape=self.image_shape or (8, 8),
            )
        return _MLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden=int(p["hidden_dim"]),
            layers=int(p["n_layers"]),
            dropout=float(p["dropout"]),
        )

    def _forward_proba(self, model: nn.Module, X: torch.Tensor) -> np.ndarray:
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X), 1024):
                batch = X[start : start + 1024].to(self.device)
                chunks.append(torch.softmax(model(batch), dim=1).double().cpu().numpy())
        if not chunks:
            return np.empty((0, 0))
        proba = np.concatenate(chunks, axis=0)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    # -- sklearn API -----------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchClassifier:
        seed_everything(self.seed)
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_arr)
        self.n_classes = len(self.classes_)
        self.input_dim = int(np.prod(X_arr.shape[1:]))
        self.model_ = self._build(self.input_dim, self.n_classes).to(self.device)

        # internal stratified validation split for early stopping only
        if self.name != "cnn":
            self.scaler_ = StandardScaler().fit(X_arr)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_arr, y_arr, test_size=0.15, random_state=self.seed, stratify=y_arr
        )
        Xtr = self._to_features(X_tr).to(self.device)
        ytr = torch.from_numpy(np.searchsorted(self.classes_, y_tr)).long().to(self.device)
        Xva = self._to_features(X_va).to(self.device)
        yva = torch.from_numpy(np.searchsorted(self.classes_, y_va)).long().to(self.device)

        opt = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.seed)
        batch_size = int(self.params["batch_size"])
        patience = int(self.params["patience"])
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model_.state_dict())
        bad = 0
        self.history_ = []

        for epoch in range(int(self.params["epochs"])):
            self.model_.train()
            perm = torch.randperm(len(Xtr), generator=gen)
            total = 0.0
            for start in range(0, len(perm), batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(idx)
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(Xva)
                val_loss = float(loss_fn(val_logits, yva).item())
                val_acc = float((val_logits.argmax(1) == yva).float().mean().item())
            sched.step(val_loss)
            self.history_.append(
                {"epoch": epoch, "train_loss": total / max(1, len(Xtr)), "val_loss": val_loss, "val_acc": val_acc}
            )
            if self.verbose:
                print(f"  epoch {epoch:3d} train={total / max(1, len(Xtr)):.4f} val={val_loss:.4f} acc={val_acc:.3f}")
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                self.best_epoch_ = epoch
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        return self._forward_proba(self.model_, self._to_features(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[proba.argmax(axis=1)]

    # -- extras ----------------------------------------------------------
    def param_count(self) -> int:
        if self.model_ is None:
            return 0
        return int(sum(p.numel() for p in self.model_.parameters()))

    def state_dict(self) -> dict[str, Any]:
        return {} if self.model_ is None else self.model_.state_dict()

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, **self.params}
