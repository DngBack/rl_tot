# Optional: custom reward heuristics

def custom_reward(text: str) -> float:
    # example: reward longer and more detailed outputs
    return len(text.split())
