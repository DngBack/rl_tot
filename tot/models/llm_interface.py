from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **gen_kwargs) -> str:
        pass

    @abstractmethod
    def evaluate(self, text: str) -> float:
        pass

def get_model(config):
    if config.backend == 'openai':
        from models.hf_model import OpenAIModel
        return OpenAIModel(api_key=config.openai_api_key)
    else:
        from models.hf_model import HFModel
        return HFModel(config.hf_model)
