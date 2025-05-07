from tasks.game24 import get_prompt as game24_prompt
from tasks.creative_writing import get_prompt as creative_prompt

mapping = {
    'game24': game24_prompt,
    'creative': creative_prompt,
}

def get_prompt(name: str) -> str:
    if name in mapping:
        return mapping[name]()
    raise ValueError(f"Unknown task {name}")
