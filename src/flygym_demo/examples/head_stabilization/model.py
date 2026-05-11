from __future__ import annotations

import pickle
from importlib.resources import files
from pathlib import Path

import numpy as np


class JointAngleScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.scale[self.scale == 0] = 1

    @classmethod
    def from_pickle(cls, path: str | Path | None = None) -> "JointAngleScaler":
        if path is None:
            path = (
                Path(str(files("flygym_demo.examples")))
                / "assets/trained_models/head_stabilization/joint_angle_scaler_params.pkl"
            )
        with open(path, "rb") as f:
            params = pickle.load(f)
        if isinstance(params, dict):
            mean = params.get("mean", params.get("mean_", 0))
            scale = params.get("scale", params.get("scale_", params.get("std", 1)))
        else:
            mean = getattr(params, "mean_", 0)
            scale = getattr(params, "scale_", 1)
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.scale


class ThreeLayerMLP:
    """Torch MLP matching the pretrained head-stabilization tutorial model shape."""

    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dim: int = 32):
        import torch

        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def __call__(self, x):
        return self.net(x)


class HeadStabilizationInferenceWrapper:
    def __init__(
        self,
        scaler: JointAngleScaler | None = None,
        model: ThreeLayerMLP | None = None,
        contact_force_thr: float = 0.5,
    ) -> None:
        self.scaler = scaler
        self.model = model
        self.contact_force_thr = contact_force_thr

    def predict(self, joint_angles: np.ndarray, contact_forces: np.ndarray) -> np.ndarray:
        contact_mask = (np.asarray(contact_forces) >= self.contact_force_thr).astype(np.float32)
        x = np.concatenate([np.asarray(joint_angles, dtype=np.float32), contact_mask])
        if self.scaler is not None:
            try:
                x = self.scaler.transform(x)
            except ValueError:
                pass
        if self.model is None:
            return np.zeros(2, dtype=np.float32)
        torch = self.model.torch
        with torch.no_grad():
            y = self.model(torch.as_tensor(x[None, :], dtype=torch.float32))
        return y.detach().cpu().numpy()[0]
