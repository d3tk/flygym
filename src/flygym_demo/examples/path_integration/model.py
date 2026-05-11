from __future__ import annotations

import numpy as np


class LinearModel:
    """Small dependency-free linear regressor for tutorial path integration."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearModel":
        x_aug = np.c_[np.ones(len(x)), x]
        beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:].T
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before predict")
        return x @ self.coef_.T + self.intercept_


def path_integrate(start_xytheta: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Integrate local-frame displacement deltas into global x/y/heading state."""

    states = np.zeros((len(deltas) + 1, 3), dtype=np.float64)
    states[0] = start_xytheta
    for i, (forward, lateral, dtheta) in enumerate(deltas):
        theta = states[i, 2]
        c, s = np.cos(theta), np.sin(theta)
        global_delta = np.array([c * forward - s * lateral, s * forward + c * lateral])
        states[i + 1, :2] = states[i, :2] + global_delta
        states[i + 1, 2] = states[i, 2] + dtheta
    return states
