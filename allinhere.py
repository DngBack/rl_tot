"""
RL-ToT: Reinforcement Learning Tree-of-Thought for LLM Reasoning

This module implements a clean architecture for RL-ToT, combining Tree-of-Thought reasoning with Reinforcement Learning.
"""

import uuid
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import numpy as np
import math
from torch.distributions import Categorical
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Core ToT Engine
# -----------------------------------------------------------------------------


@dataclass
class ThoughtNode:
    """
    Represents a node in the reasoning tree.
    """

    state: str
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = None
    score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}


class TreeManager:
    """
    Manages the tree of thoughts, including expansion, scoring, and traversal.
    """

    def __init__(self, expand_fn: Callable, score_fn: Callable):
        self.expand_fn = expand_fn
        self.score_fn = score_fn
        self.root = None

    def create_root(self, initial_state: str) -> ThoughtNode:
        """
        Creates the root node of the tree.
        """
        self.root = ThoughtNode(state=initial_state)
        return self.root

    def expand_node(self, node: ThoughtNode) -> List[ThoughtNode]:
        """
        Expands a node by generating children using the expand_fn.
        """
        children_states = self.expand_fn(node.state)
        children = [ThoughtNode(state=state, parent=node) for state in children_states]
        node.children.extend(children)
        return children

    def score_node(self, node: ThoughtNode) -> float:
        """
        Scores a node using the score_fn.
        """
        node.score = self.score_fn(node.state)
        return node.score

    def traverse(
        self, strategy: str = "best_first", max_depth: int = 5, max_branches: int = 3
    ) -> List[ThoughtNode]:
        """
        Traverses the tree using the specified strategy.
        """
        if strategy == "best_first":
            return self._best_first_traverse(max_depth, max_branches)
        elif strategy == "dfs":
            return self._dfs_traverse(max_depth, max_branches)
        elif strategy == "random":
            return self._random_traverse(max_depth, max_branches)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _best_first_traverse(
        self, max_depth: int, max_branches: int
    ) -> List[ThoughtNode]:
        """
        Best-first traversal of the tree.
        """
        if not self.root:
            return []
        queue = [(self.root, 0)]
        visited = []
        while queue and len(visited) < max_branches:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            visited.append(node)
            self.expand_node(node)
            for child in node.children:
                self.score_node(child)
                queue.append((child, depth + 1))
            queue.sort(key=lambda x: x[0].score, reverse=True)
        return visited

    def _dfs_traverse(self, max_depth: int, max_branches: int) -> List[ThoughtNode]:
        """
        Depth-first traversal of the tree.
        """
        if not self.root:
            return []
        stack = [(self.root, 0)]
        visited = []
        while stack and len(visited) < max_branches:
            node, depth = stack.pop()
            if depth >= max_depth:
                continue
            visited.append(node)
            self.expand_node(node)
            for child in reversed(node.children):
                self.score_node(child)
                stack.append((child, depth + 1))
        return visited

    def _random_traverse(self, max_depth: int, max_branches: int) -> List[ThoughtNode]:
        """
        Random traversal of the tree.
        """
        if not self.root:
            return []
        queue = [(self.root, 0)]
        visited = []
        while queue and len(visited) < max_branches:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            visited.append(node)
            self.expand_node(node)
            for child in node.children:
                self.score_node(child)
                queue.append((child, depth + 1))
            random.shuffle(queue)
        return visited


# -----------------------------------------------------------------------------
# 2. LLM Interface (HFModel)
# -----------------------------------------------------------------------------


class HFModel:
    """
    Wraps a HuggingFace LLM and value head for PPO.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1).to(device)

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generates text using the LLM.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, prompt: str) -> float:
        """
        Evaluates the prompt using the value head.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :]
            value = self.value_head(hidden_states).item()
        return value

    def generate_with_values(
        self, prompts: List[str], max_new_tokens: int = 128, **gen_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tokenize batch
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        # Generate with value head
        outputs = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )
        input_ids = batch["input_ids"]
        sequences = outputs.sequences
        # scores: list of (batch, vocab) tensors, length = new_tokens
        # stack into (batch, seq_len, vocab)
        scores = torch.stack(outputs.scores, dim=1)
        logprobs = torch.log_softmax(scores, dim=-1)
        values = outputs.v_values  # (batch, seq_len)
        return input_ids, sequences, logprobs, values


# -----------------------------------------------------------------------------
# 3. Reward Modules
# -----------------------------------------------------------------------------


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    """

    @abstractmethod
    def __call__(self, state: str) -> float:
        pass


class BinaryMatchReward(RewardFunction):
    """
    Binary reward based on exact match.
    """

    def __init__(self, target: str):
        self.target = target

    def __call__(self, state: str) -> float:
        return 1.0 if state == self.target else 0.0


class LengthPenaltyReward(RewardFunction):
    """
    Penalizes long outputs.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, state: str) -> float:
        return 1.0 - (len(state) / self.max_length)


class CompositeReward(RewardFunction):
    """
    Combines multiple reward functions.
    """

    def __init__(self, rewards: List[RewardFunction], weights: List[float]):
        self.rewards = rewards
        self.weights = weights

    def __call__(self, state: str) -> float:
        return sum(w * r(state) for r, w in zip(self.rewards, self.weights))


class RewardModel(nn.Module):
    """
    Wraps a reward function as a torch.nn.Module for PPOTrainer.
    """

    def __init__(self, reward_fn: RewardFunction):
        super().__init__()
        self.reward_fn = reward_fn

    def forward(self, state: str) -> torch.Tensor:
        return torch.tensor(self.reward_fn(state), dtype=torch.float32)


# -----------------------------------------------------------------------------
# 4. RLVR Trainer
# -----------------------------------------------------------------------------


class RLVRTrainer:
    """
    Wraps PPOTrainer from TRL for RL training.
    """

    def __init__(
        self, model: HFModel, reward_model: RewardModel, ppo_config: PPOConfig
    ):
        self.model = model
        self.reward_model = reward_model
        self.ppo_trainer = PPOTrainer(model=model.model, config=ppo_config)

    def train_step(self, prompt: str) -> Dict[str, float]:
        """
        Runs a single PPO training step.
        """
        response = self.model.generate(prompt)
        reward = self.reward_model(response)
        self.ppo_trainer.step(prompt, response, reward)
        return {"reward": reward.item()}


# -----------------------------------------------------------------------------
# 5. Multi-Source Distillation
# -----------------------------------------------------------------------------


class TrajectoryCollector:
    """
    Collects reasoning trajectories from base LLM, RL-trained LLM, and ToT search.
    """

    def __init__(
        self, base_model: HFModel, rl_model: HFModel, tree_manager: TreeManager
    ):
        self.base_model = base_model
        self.rl_model = rl_model
        self.tree_manager = tree_manager

    def collect_trajectories(
        self, prompt: str, num_trajectories: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Collects trajectories from all sources.
        """
        trajectories = []
        for _ in range(num_trajectories):
            base_response = self.base_model.generate(prompt)
            rl_response = self.rl_model.generate(prompt)
            tot_nodes = self.tree_manager.traverse()
            trajectories.append(
                {
                    "base": base_response,
                    "rl": rl_response,
                    "tot": [node.state for node in tot_nodes],
                }
            )
        return trajectories


class StudentDistiller:
    """
    Fine-tunes a student model on collected trajectories.
    """

    def __init__(
        self, student_model: PreTrainedModel, student_tokenizer: PreTrainedTokenizer
    ):
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer

    def distill(self, trajectories: List[Dict[str, Any]], num_epochs: int = 3) -> None:
        """
        Distills knowledge from trajectories into the student model.
        """
        for epoch in range(num_epochs):
            for traj in trajectories:
                for source in ["base", "rl", "tot"]:
                    if source == "tot":
                        for state in traj[source]:
                            self._train_step(state)
                    else:
                        self._train_step(traj[source])

    def _train_step(self, state: str) -> None:
        """
        Trains the student model on a single state.
        """
        inputs = self.student_tokenizer(state, return_tensors="pt")
        self.student_model.train()
        outputs = self.student_model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        # Optimizer step would go here


# -----------------------------------------------------------------------------
# 6. Adaptive Search Scheduler
# -----------------------------------------------------------------------------


class SearchScheduler:
    """
    Dynamically adjusts search parameters based on recent performance.
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        initial_branches: int = 3,
        initial_depth: int = 5,
    ):
        self.temp = initial_temp
        self.branches = initial_branches
        self.depth = initial_depth
        self.recent_entropy = []
        self.recent_rewards = []

    def update(self, entropy: float, reward: float) -> None:
        """
        Updates search parameters based on recent entropy and reward.
        """
        self.recent_entropy.append(entropy)
        self.recent_rewards.append(reward)
        if len(self.recent_entropy) > 10:
            self.recent_entropy.pop(0)
            self.recent_rewards.pop(0)
        avg_entropy = sum(self.recent_entropy) / len(self.recent_entropy)
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        self.temp = max(0.1, min(2.0, self.temp * (1.0 + (avg_reward - 0.5) * 0.1)))
        self.branches = max(1, min(10, self.branches + int((avg_reward - 0.5) * 2)))
        self.depth = max(1, min(10, self.depth + int((avg_entropy - 0.5) * 2)))


# -----------------------------------------------------------------------------
# 7. Strategy Plugins
# -----------------------------------------------------------------------------


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.
    """

    @abstractmethod
    def search(
        self, tree_manager: TreeManager, max_depth: int, max_branches: int
    ) -> List[ThoughtNode]:
        pass


class DepthFirstStrategy(SearchStrategy):
    """
    Depth-first search strategy.
    """

    def search(
        self, tree_manager: TreeManager, max_depth: int, max_branches: int
    ) -> List[ThoughtNode]:
        return tree_manager._dfs_traverse(max_depth, max_branches)


class StochasticBeamStrategy(SearchStrategy):
    """
    Stochastic beam search strategy with Gumbel noise.
    """

    def search(
        self, tree_manager: TreeManager, max_depth: int, max_branches: int
    ) -> List[ThoughtNode]:
        if not tree_manager.root:
            return []
        queue = [(tree_manager.root, 0)]
        visited = []
        while queue and len(visited) < max_branches:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            visited.append(node)
            tree_manager.expand_node(node)
            for child in node.children:
                tree_manager.score_node(child)
                queue.append((child, depth + 1))
            gumbel_noise = torch.distributions.Gumbel(0, 1).sample((len(queue),))
            queue = [x for _, x in sorted(zip(gumbel_noise, queue), reverse=True)]
        return visited


# -----------------------------------------------------------------------------
# 8. Main Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HFModel(model_name, device)
    reward_fn = BinaryMatchReward("target")
    reward_model = RewardModel(reward_fn)
    ppo_config = PPOConfig()
    trainer = RLVRTrainer(model, reward_model, ppo_config)
    tree_manager = TreeManager(
        expand_fn=lambda x: [x + "a", x + "b"], score_fn=lambda x: len(x)
    )
    collector = TrajectoryCollector(model, model, tree_manager)
    distiller = StudentDistiller(model.model, model.tokenizer)
    scheduler = SearchScheduler()
    strategy = DepthFirstStrategy()
    trajectories = collector.collect_trajectories("start", 5)
    distiller.distill(trajectories)
    logger.info("Training complete.")
