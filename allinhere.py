# rl_tot_core.py

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
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# -----------------------------------------------------------------------------
# 1. Core ToT Engine
# -----------------------------------------------------------------------------


class ThoughtNode:
    """
    Node in the Tree-of-Thought search, tracking state, score, and metadata.
    """

    def __init__(
        self,
        state: str,
        parent: Optional["ThoughtNode"] = None,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id: str = str(uuid.uuid4())
        self.state: str = state
        self.parent: Optional["ThoughtNode"] = parent
        self.children: List["ThoughtNode"] = []
        self.score: float = score
        self.metadata: Dict[str, Any] = metadata or {
            "visits": 0,
            "entropy": None,
            "reward": 0.0,
        }

    def add_child(self, child: "ThoughtNode") -> None:
        self.children.append(child)

    def visit(self) -> None:
        self.metadata["visits"] += 1

    def set_entropy(self, entropy: float) -> None:
        self.metadata["entropy"] = entropy

    def set_reward(self, reward: float) -> None:
        self.metadata["reward"] = reward

    def __repr__(self) -> str:
        v = self.metadata["visits"]
        r = self.metadata["reward"]
        return f"<Node {self.id[:8]} score={self.score:.3f} visits={v} reward={r:.2f}>"


class TreeManager:
    """
    Core inference engine for Tree-of-Thought with pluggable strategies.
    """

    def __init__(
        self,
        expand_fn: Callable[[ThoughtNode], List[str]],
        score_fn: Callable[[str], float],
        max_depth: int = 10,
    ):
        self.expand_fn = expand_fn
        self.score_fn = score_fn
        self.max_depth = max_depth
        self.root: Optional[ThoughtNode] = None

    def initialize(self, prompt: str) -> None:
        self.root = ThoughtNode(state=prompt)

    def generate_and_expand(self, node: ThoughtNode) -> List[ThoughtNode]:
        children: List[ThoughtNode] = []
        for state in self.expand_fn(node):
            score = self.score_fn(state)
            child = ThoughtNode(state=state, parent=node, score=score)
            node.add_child(child)
            children.append(child)
        return children

    def traverse(
        self, strategy: Union[str, "SearchStrategy"] = "best_first"
    ) -> ThoughtNode:
        assert self.root is not None, "TreeManager not initialized"
        current = self.root
        depth = 0
        while depth < self.max_depth:
            current.visit()
            kids = self.generate_and_expand(current)
            if not kids:
                break

            if isinstance(strategy, SearchStrategy):
                sel = strategy.select(kids)
                # allow beam returning multiple nodes
                current = sel[0] if isinstance(sel, list) else sel
            elif strategy == "best_first":
                current = max(kids, key=lambda n: n.score)
            elif strategy == "dfs":
                current = kids[0]
            else:  # random
                current = random.choice(kids)

            depth += 1
        return current


# -----------------------------------------------------------------------------
# 2. LLM Interface (HFModel)
# -----------------------------------------------------------------------------


class HFModel:
    """
    Wraps a HuggingFace causal LM and its value-head for PPO.
    """

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # policy-only model for plain generation/evaluation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        # policy+value model for PPO
        self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.temperature: float = 1.0

    def generate(self, prompt: str, **gen_kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        cfg = GenerationConfig(temperature=self.temperature, **gen_kwargs)
        output_ids = self.model.generate(**inputs, generation_config=cfg)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def evaluate(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, labels=inputs.input_ids)
        return -out.loss.item()

    def generate_with_values(
        self, prompts: List[str], max_new_tokens: int = 128, **gen_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tokenize batch
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.ppo_model.device
        )
        # Generate with value head
        outputs = self.ppo_model.generate(
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
    """Base for reward functions."""

    @abstractmethod
    def __call__(self, node: ThoughtNode) -> float: ...


class BinaryMatchReward(RewardFunction):
    def __init__(self, terms: List[str]):
        self.terms = [t.lower() for t in terms]

    def __call__(self, node: ThoughtNode) -> float:
        s = node.state.lower()
        return float(all(t in s for t in self.terms))


class LengthPenaltyReward(RewardFunction):
    def __init__(self, max_len: int = 100):
        self.max_len = max_len

    def __call__(self, node: ThoughtNode) -> float:
        return max(0.0, 1.0 - len(node.state) / self.max_len)


class CompositeReward(RewardFunction):
    def __init__(
        self, rewards: List[RewardFunction], weights: Optional[List[float]] = None
    ):
        self.rewards = rewards
        self.weights = weights or [1.0] * len(rewards)

    def __call__(self, node: ThoughtNode) -> float:
        return sum(w * r(node) for r, w in zip(self.rewards, self.weights))


class RewardModel(torch.nn.Module):
    """
    Wraps RewardFunction into an nn.Module for PPOTrainer.
    """

    def __init__(self, reward_fn: RewardFunction, tokenizer: AutoTokenizer):
        super().__init__()
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        rewards = [self.reward_fn(ThoughtNode(state=t)) for t in texts]
        return torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)


# -----------------------------------------------------------------------------
# 4. RLVR Trainer
# -----------------------------------------------------------------------------


class RLVRTrainer:
    """
    Wraps TRL's PPOTrainer with your ToT environment and reward logic.
    """

    def __init__(
        self,
        llm: HFModel,
        tree_manager: TreeManager,
        reward_fn: RewardFunction,
        train_dataset: Dataset,
        ppo_config: Optional[PPOConfig] = None,
    ):
        self.llm = llm
        self.tree_manager = tree_manager
        self.ppo_config = ppo_config or PPOConfig(
            batch_size=8, forward_batch_size=2, ppo_epochs=4
        )
        # Reference model for KL
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            llm.ppo_model.config._name_or_path
        )
        self.reward_model = RewardModel(reward_fn, llm.tokenizer)

        self.trainer = PPOTrainer(
            args=self.ppo_config,
            processing_class=llm.tokenizer,
            model=llm.ppo_model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            train_dataset=train_dataset,
            value_model=None,
            data_collator=None,
            eval_dataset=None,
        )

    def train(self) -> None:
        """Run full PPO training loop."""
        self.trainer.train()

    def train_step(self, prompts: List[str]) -> None:
        """
        Single PPO step: generate, score, and step the PPOTrainer.
        """
        input_ids, sequences, _, _ = self.llm.generate_with_values(prompts)
        gen_tokens = sequences[:, input_ids.size(-1) :]
        rewards = self.reward_model(input_ids).to(input_ids.device)
        self.trainer.step(queries=input_ids, responses=gen_tokens, rewards=rewards)


# -----------------------------------------------------------------------------
# 5. Multi-Source Distillation
# -----------------------------------------------------------------------------


class TrajectoryCollector:
    def __init__(
        self,
        llm: HFModel,
        rl_trainer: RLVRTrainer,
        tree_manager: TreeManager,
        k_base: int = 256,
    ):
        self.llm = llm
        self.rl_trainer = rl_trainer
        self.tree_manager = tree_manager
        self.k_base = k_base

    def collect_base(self, prompt: str) -> List[Dict[str, Any]]:
        recs = []
        for _ in range(self.k_base):
            out = self.llm.generate(prompt, do_sample=True, temperature=0.8)
            recs.append({"input": prompt, "cot": None, "output": out, "source": "base"})
        return recs

    def collect_rl(self, prompt: str) -> Dict[str, Any]:
        self.rl_trainer.train_step([prompt])
        _, seqs, _, _ = self.llm.generate_with_values([prompt])
        out = self.llm.tokenizer.decode(seqs[0], skip_special_tokens=True)
        return {"input": prompt, "cot": None, "output": out, "source": "rlvr"}

    def collect_tot(
        self, prompt: str, strategy: Union[str, "SearchStrategy"] = "best_first"
    ) -> Dict[str, Any]:
        self.tree_manager.initialize(prompt)
        final = self.tree_manager.traverse(strategy)
        path = []
        node = final
        while node:
            path.append(node.state)
            node = node.parent
        return {
            "input": prompt,
            "cot": path[::-1],
            "output": final.state,
            "source": "tot",
        }


class StudentDistiller:
    def __init__(
        self,
        student_model_name: str,
        trajectories: List[Dict[str, Any]],
        output_dir: str = "./distilled",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(student_model_name)

        examples = []
        for rec in trajectories:
            cot = "" if rec["cot"] is None else "\n".join(rec["cot"]) + "\n"
            examples.append(
                {"input_text": rec["input"], "target_text": cot + rec["output"]}
            )

        def preprocess(ex: Dict[str, str]) -> Dict[str, Any]:
            enc = self.tokenizer(
                ex["input_text"], truncation=True, padding="max_length", max_length=128
            )
            dec = self.tokenizer(
                ex["target_text"], truncation=True, padding="max_length", max_length=128
            )
            enc["labels"] = dec["input_ids"]
            return enc

        dataset = [preprocess(ex) for ex in examples]
        self.args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=500,
            logging_steps=100,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

    def fine_tune(self) -> None:
        self.trainer.train()
        self.trainer.save_model()


# -----------------------------------------------------------------------------
# 6. Adaptive Search Scheduler
# -----------------------------------------------------------------------------


class SearchScheduler:
    """
    Dynamically adjust temperature (T), branch budget (B), and depth (D)
    based on recent CoT entropy and reward.
    """

    def __init__(
        self,
        llm: HFModel,
        tree_manager: TreeManager,
        tau_low: float = 1.0,
        tau_high: float = 2.5,
        base_budget: int = 5,
        base_depth: int = 10,
    ):
        self.llm = llm
        self.tree_manager = tree_manager
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.T = llm.temperature
        self.B = base_budget
        self.D = base_depth
        self.entropy_hist: List[float] = []
        self.reward_hist: List[float] = []

    def step(self, node: ThoughtNode) -> None:
        if (ent := node.metadata.get("entropy")) is not None:
            self.entropy_hist.append(ent)
        self.reward_hist.append(node.metadata.get("reward", 0.0))

        # Adjust temperature
        if self.entropy_hist:
            avg_ent = sum(self.entropy_hist[-10:]) / len(self.entropy_hist[-10:])
            if avg_ent < self.tau_low:
                self.T *= 1.2
            elif avg_ent > self.tau_high:
                self.T *= 0.8
            self.T = max(0.1, min(self.T, 5.0))
            self.llm.temperature = self.T

        # Adjust budget & depth
        if self.reward_hist:
            avg_r = sum(self.reward_hist[-10:]) / len(self.reward_hist[-10:])
            if avg_r > 0.8:
                self.B = max(1, int(self.B * 0.8))
                self.D = max(1, int(self.D * 0.9))
            else:
                self.B = min(50, int(self.B * 1.1) + 1)
                self.D = min(50, int(self.D * 1.05) + 1)
            self.tree_manager.max_depth = self.D


# -----------------------------------------------------------------------------
# 7. Strategy Plugins
# -----------------------------------------------------------------------------


class SearchStrategy(ABC):
    @abstractmethod
    def select(
        self, children: List[ThoughtNode]
    ) -> Union[ThoughtNode, List[ThoughtNode]]: ...


class DepthFirstStrategy(SearchStrategy):
    def select(self, children: List[ThoughtNode]) -> ThoughtNode:
        return children[0]


class StochasticBeamStrategy(SearchStrategy):
    def __init__(self, beam_width: int = 5):
        self.k = beam_width

    def select(self, children: List[ThoughtNode]) -> List[ThoughtNode]:
        scores = torch.tensor([c.score for c in children], dtype=torch.float32)
        # Gumbel noise
        gumbel = -torch.log(-torch.log(torch.rand_like(scores)))
        noisy = scores + gumbel
        idx = torch.topk(noisy, min(self.k, len(children))).indices.tolist()
        return [children[i] for i in idx]
