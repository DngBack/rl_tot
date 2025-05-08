from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def __call__(self, state: str) -> float:
        pass


class BinaryMatchReward(RewardFunction):
    """Binary reward based on exact match."""

    def __init__(self, target: str):
        self.target = target

    def __call__(self, state: str) -> float:
        return 1.0 if state == self.target else 0.0


class LengthPenaltyReward(RewardFunction):
    """Penalizes long outputs."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, state: str) -> float:
        return 1.0 - (len(state) / self.max_length)


class CompositeReward(RewardFunction):
    """Combines multiple reward functions."""

    def __init__(self, rewards: List[RewardFunction], weights: List[float]):
        self.rewards = rewards
        self.weights = weights

    def __call__(self, state: str) -> float:
        return sum(w * r(state) for r, w in zip(self.rewards, self.weights))


class RewardModel(nn.Module):
    """Wraps a reward function as a torch.nn.Module for PPOTrainer."""

    def __init__(self, reward_fn: RewardFunction):
        super().__init__()
        self.reward_fn = reward_fn

    def forward(self, state: str) -> torch.Tensor:
        return torch.tensor(self.reward_fn(state), dtype=torch.float32)

    def evaluate(self, states: List[str]) -> List[float]:
        """Evaluate a list of states."""
        return [self.reward_fn(state) for state in states]
