"""Segmenter interface: every backend is a *tool* that turns 2.5D stacks into class probabilities.

The agent never depends on a particular network. A backend only has to implement
``predict_batch(stacks_u8) -> (P, B, 3, S, S)`` where ``P`` is the number of
stochastic passes (TTA views x MC-dropout samples for networks, ensemble members for
the classical baseline). The spread across passes is the epistemic-uncertainty signal.
Classes are 0 background, 1 liver, 2 tumour.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

N_CLASSES = 3


class Segmenter(ABC):
    name: str = "base"
    version: str = "0"

    @property
    @abstractmethod
    def n_passes(self) -> int:
        ...

    @abstractmethod
    def predict_batch(self, stacks_u8: np.ndarray, passes: int | None = None, roi: tuple | None = None,
                      coarse: bool = False) -> np.ndarray:
        """stacks_u8: (B, 3, S, S) uint8 -> probabilities (P, B, 3, S, S) float32.

        ``passes=1`` asks for a single deterministic pass. ``roi=(r0, r1, c0, c1)`` and
        ``coarse=True`` are optional speed hints a backend may ignore (pixels outside the
        ROI may be returned as background).
        """

    def describe(self) -> dict:
        return {"backend": self.name, "version": self.version, "passes": self.n_passes}
