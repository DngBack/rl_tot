import openai
from models.llm_interface import LLMInterface

class OpenAIModel(LLMInterface):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate(self, prompt: str, **gen_kwargs) -> str:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            **gen_kwargs
        )
        return resp.choices[0].message.content

    def evaluate(self, text: str) -> float:
        # simple length-based reward or call a reward model
        return len(text)