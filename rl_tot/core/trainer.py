from typing import Dict, List
import torch
from trl import PPOTrainer, PPOConfig
from ..models.language_model import LanguageModel
from .reward import RewardModel


class RLVRTrainer:
    """Wraps PPOTrainer from TRL for RL training."""

    def __init__(
        self,
        model: LanguageModel,
        reward_model: RewardModel,
        ppo_config: PPOConfig,
    ):
        self.model = model
        self.reward_model = reward_model

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            model=model.model,
            ref_model=model.model,  # Use same model as reference
            reward_model=reward_model,
            config=ppo_config,
        )

    def train_step(self, prompt: str) -> Dict[str, float]:
        """Run a single PPO training step."""
        # Generate response
        response = self.model.generate(prompt)

        # Get reward
        reward = self.reward_model(response)

        # Update policy
        self.ppo_trainer.step(
            query_tensor=self.model.tokenizer(prompt, return_tensors="pt")["input_ids"],
            response_tensor=self.model.tokenizer(response, return_tensors="pt")[
                "input_ids"
            ],
            reward_tensor=torch.tensor([reward], dtype=torch.float32),
        )

        return {"reward": reward}

    def train_epoch(
        self,
        prompts: List[str],
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train for one epoch on a list of prompts."""
        total_reward = 0.0

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            for prompt in batch_prompts:
                metrics = self.train_step(prompt)
                total_reward += metrics["reward"]

        return {
            "avg_reward": total_reward / len(prompts),
        }

    def train(
        self,
        prompts: List[str],
        num_epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """Train for multiple epochs."""
        metrics_history = {
            "avg_reward": [],
        }

        for epoch in range(num_epochs):
            metrics = self.train_epoch(prompts, batch_size)

            for k, v in metrics.items():
                metrics_history[k].append(v)

        return metrics_history
