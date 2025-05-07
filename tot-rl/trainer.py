from typing import List, Optional, Callable
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from models.hf_models import HFModel
from core.tree_manager import ThoughtNode, TreeManager


class RewardModel(torch.nn.Module):
    """
    Wraps a scalar reward function into an nn.Module for compatibility with PPOTrainer.
    Uses the provided tokenizer for decoding sequences.
    """

    def __init__(
        self, reward_fn: Callable[[ThoughtNode], float], tokenizer: AutoTokenizer
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode each sequence batch-wise and compute scalar rewards.
        """
        rewards: List[float] = []
        # iterate over batch
        for seq in input_ids.tolist():
            # decode full sequence
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            node = ThoughtNode(state=text)
            r = self.reward_fn(node)
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)


# RLVRTrainer
class RLVRTrainer:
    """
    Integrates PPOTrainer with TreeManager and reward module.
    """

    def __init__(
        self,
        llm: HFModel,
        tree_manager: TreeManager,
        reward_fn: Callable[[ThoughtNode], float],
        train_dataset: torch.utils.data.Dataset,
        ppo_config: Optional[PPOConfig] = None,
    ):
        self.tree_manager = tree_manager
        self.llm = llm
        self.ppo_config = ppo_config or PPOConfig(
            batch_size=4,
            num_ppo_epochs=4,
        )
        # Create a reference copy of the policy model for KL penalties
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            llm.ppo_model.config._name_or_path, config=llm.ppo_model.config
        )
        # Wrap reward_fn into nn.Module, providing tokenizer
        self.reward_model = RewardModel(reward_fn, tokenizer=llm.tokenizer)
        # Prepare the PPOTrainer with required arguments
        self.trainer = PPOTrainer(
            args=self.ppo_config,
            processing_class=llm.tokenizer,
            model=llm.ppo_model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            train_dataset=train_dataset,
            value_model=llm.ppo_model.value_head
            if hasattr(llm.ppo_model, "value_head")
            else None,
            data_collator=None,
            eval_dataset=None,
        )

    def train(self):
        """
        Runs the built-in PPO training loop.
        """
        # Ensure tree_manager is used for on-policy rollouts if needed
        self.trainer.train()

    def train_step(self, prompts: List[str]) -> None:
        """
        Manual single-step PPO update (optional).
        """
        # Generate rollouts
        input_ids, sequences, logprobs, values = self.llm.generate_with_values(prompts)
        # Extract generated part only
        gen_tokens = sequences[:, input_ids.shape[-1] :]
        # Compute rewards via wrapped reward_model
        rewards_list = self.reward_model(input_ids).tolist()
        rewards_tensor = torch.tensor(
            rewards_list, dtype=torch.float32, device=values.device
        )
        # Step PPO trainer
        self.trainer.step(
            queries=input_ids, responses=gen_tokens, rewards=rewards_tensor
        )
