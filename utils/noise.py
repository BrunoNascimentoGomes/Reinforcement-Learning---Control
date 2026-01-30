# utils/ou_noise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class OUNoise:
    action_dim: int
    mu: float = 0.0
    theta: float = 0.15
    sigma: float = 0.20
    dt: float = 1.0
    x0: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        if self.x0 is None:
            self._x = np.zeros(self.action_dim, dtype=np.float32)
        else:
            x0 = np.asarray(self.x0, dtype=np.float32).reshape(-1)
            if x0.size != self.action_dim:
                raise ValueError("x0 must have shape (action_dim,).")
            self._x = x0.copy()

    def sample(self) -> np.ndarray:
        x = self._x
        dx = (
            self.theta * (self.mu - x) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim).astype(np.float32)
        )
        self._x = x + dx
        return self._x.copy()
