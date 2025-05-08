from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate_thoughts(
        self, prompts: List[str], num_thoughts: int = 5, temperature: float = 0.7
    ) -> List[str]:
        """
        Generate multiple thoughts from prompts.

        Args:
            prompts: List of input prompts
            num_thoughts: Number of thoughts to generate per prompt
            temperature: Sampling temperature

        Returns:
            List of generated thoughts
        """
        thoughts = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate multiple thoughts
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=num_thoughts,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Decode thoughts
            for output in outputs:
                thought = self.tokenizer.decode(output, skip_special_tokens=True)
                thoughts.append(thought)

        return thoughts

    def generate(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.7
    ) -> str:
        """
        Generate a single response from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
