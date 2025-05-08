from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig
from ..models.language_model import LanguageModel
from .tree import ThoughtTree
from .reward import RewardModel, CompositeReward, LengthPenaltyReward
from .trainer import RLVRTrainer


class RLToT:
    def __init__(
        self,
        model_name: str = "gpt2",
        max_depth: int = 3,
        num_thoughts: int = 5,
        temperature: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.max_depth = max_depth
        self.num_thoughts = num_thoughts
        self.temperature = temperature
        self.device = device

        # Initialize components
        self.language_model = LanguageModel(model_name, device)
        self.tree = ThoughtTree(max_depth, num_thoughts)

        # Initialize reward model with composite rewards
        rewards = [
            LengthPenaltyReward(max_length=100),
        ]
        weights = [1.0]
        self.reward_model = RewardModel(CompositeReward(rewards, weights))

        # Initialize PPO config
        self.ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=64,
            mini_batch_size=16,
            gradient_accumulation_steps=1,
        )

        # Initialize trainer
        self.trainer = RLVRTrainer(
            model=self.language_model,
            reward_model=self.reward_model,
            ppo_config=self.ppo_config,
        )

    def generate(
        self, prompt: str, max_tokens: int = 100, num_samples: int = 1
    ) -> List[str]:
        """
        Generate responses using RL-ToT approach.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            num_samples: Number of samples to generate

        Returns:
            List of generated responses
        """
        # Initialize thought tree with prompt
        self.tree.initialize(prompt)

        # Generate thoughts and evaluate rewards
        for depth in range(self.max_depth):
            thoughts = self.language_model.generate_thoughts(
                self.tree.get_current_thoughts(),
                num_thoughts=self.num_thoughts,
                temperature=self.temperature,
            )

            # Evaluate thoughts and update tree
            rewards = [float(r) for r in self.reward_model.evaluate(thoughts)]
            self.tree.update(thoughts, rewards)

            # Select best thoughts for next iteration
            best_thoughts = self.tree.select_best_thoughts()

        # Generate final responses
        responses = []
        for _ in range(num_samples):
            response = self.language_model.generate(
                self.tree.get_best_thought(),
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            responses.append(response)

        return responses

    def train(
        self,
        training_data: List[Dict[str, Any]],
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
    ) -> Dict[str, List[float]]:
        """
        Train the RL-ToT agent on provided data.

        Args:
            training_data: List of training examples
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization

        Returns:
            Dictionary containing training metrics
        """
        # Extract prompts from training data
        prompts = [item["prompt"] for item in training_data]

        # Update learning rate
        self.ppo_config.learning_rate = learning_rate

        # Train
        metrics = self.trainer.train(
            prompts=prompts,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        return metrics


class ToTAgent:
    def __init__(
        self,
        model_name: str,
        max_depth: int = 3,
        num_thoughts: int = 5,
        temperature: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = LanguageModel(model_name, device)
        self.tree = ThoughtTree(max_depth, num_thoughts)
        self.temperature = temperature
        self.device = device

    def think(self, prompt: str) -> str:
        """
        Main thinking process using Tree of Thoughts.

        Args:
            prompt: Initial prompt to think about

        Returns:
            Best thought after exploration
        """
        # Initialize tree with prompt
        self.tree.initialize(prompt)

        # Iterate through depths
        for _ in range(self.tree.max_depth):
            # Get current thoughts
            current_thoughts = self.tree.get_current_thoughts()

            # Generate new thoughts
            new_thoughts = self.model.generate_thoughts(
                current_thoughts,
                num_thoughts=self.tree.num_thoughts,
                temperature=self.temperature,
            )

            # Evaluate thoughts (simple length-based reward for now)
            rewards = [len(thought) for thought in new_thoughts]

            # Update tree
            self.tree.update(new_thoughts, rewards)

        # Return best thought
        return self.tree.get_best_thought()

    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate a response using the best thought.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        # Get best thought
        best_thought = self.think(prompt)

        # Generate response based on best thought
        response = self.model.generate(
            best_thought, max_tokens=max_tokens, temperature=self.temperature
        )

        return response
