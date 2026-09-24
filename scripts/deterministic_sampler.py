"""Dependency-free deterministic sampler used by confirmatory training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class EpochShuffleSampler:
    """Deterministic epoch shuffle with a serializable cursor and RNG state."""

    size: int
    seed: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("sampler size must be positive")
        self.order = list(range(self.size))
        self.rng = random.Random(self.seed)
        self.rng.shuffle(self.order)
        self.position = 0

    def next(self) -> int:
        if self.position and self.position % self.size == 0:
            self.rng.shuffle(self.order)
        index = self.order[self.position % self.size]
        self.position += 1
        return index

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "seed": self.seed,
            "order": self.order,
            "position": self.position,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["size"] != self.size or state["seed"] != self.seed:
            raise ValueError("resume sampler does not match this run")
        order = list(state["order"])
        if sorted(order) != list(range(self.size)):
            raise ValueError("resume sampler order is invalid")
        self.order = order
        self.position = int(state["position"])
        self.rng.setstate(state["rng_state"])
