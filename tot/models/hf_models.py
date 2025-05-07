import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from models.llm_interface import LLMInterface

class HFModel(LLMInterface):
    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            torch_dtype=torch.float16
        )

    def generate(self, prompt: str, **gen_kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, text: str) -> float:
        # use a simple proxy: model log-prob sum
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs, labels=inputs.input_ids)
        # negative loss as reward
        return -output.loss.item()
