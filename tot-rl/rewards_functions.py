# reward_functions.py

from abc import ABC, abstractmethod
from typing import List
from core import ThoughtNode  # Assuming your ThoughtNode is in `core.py`


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, node: ThoughtNode) -> float:
        pass


class BinaryMatchReward(RewardFunction):
    def __init__(self, must_contain: List[str]):
        self.must_contain = must_contain

    def __call__(self, node: ThoughtNode) -> float:
        state = node.state.lower()
        return float(all(term.lower() in state for term in self.must_contain))


class LengthPenaltyReward(RewardFunction):
    def __init__(self, max_len: int = 100):
        self.max_len = max_len

    def __call__(self, node: ThoughtNode) -> float:
        length = len(node.state)
        return max(0.0, 1.0 - length / self.max_len)


class CompositeReward(RewardFunction):
    def __init__(self, *rewards: RewardFunction, weights: List[float] = None):
        self.rewards = rewards
        self.weights = weights or [1.0] * len(rewards)

    def __call__(self, node: ThoughtNode) -> float:
        return sum(w * r(node) for r, w in zip(self.rewards, self.weights))
