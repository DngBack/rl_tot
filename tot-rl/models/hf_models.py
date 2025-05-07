from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Tuple
from trl import AutoModelForCausalLMWithValueHead


class HFModel:
    """
    Basic HuggingFace LLM interface for ToT and RL:
    - generate(): simple next-token sampling or beam.
    - evaluate(): log-prob sum as proxy score.
    - generate_with_values(): batched rollout with policy and value head for PPO.
    """

    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # core LM for ToT
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype=torch.float16
        )
        # extended model for PPO
        base = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name_or_path, config=base.config
        )

    def generate(self, prompt: str, **gen_kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs, labels=inputs.input_ids)
        return -output.loss.item()

    def generate_with_values(
        self, prompts: List[str], max_new_tokens: int = 128, **gen_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        outputs = self.ppo_model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # sequences: [batch, prompt_len + gen_len]
        sequences = outputs.sequences
        # scores per step: list of [batch, vocab]
        scores = torch.stack(outputs.scores, dim=1)
        logprobs = torch.log_softmax(scores, dim=-1)
        # extract values from hidden states
        values = outputs.v_values
        return batch["input_ids"], sequences, logprobs, values
