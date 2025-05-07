from copy import deepcopy

class TreeNode:
    def __init__(self, prompt: str, text: str = "", score: float = 0.0):
        self.prompt = prompt
        self.text = text
        self.score = score
        self.children = []

class TreeManager:
    def __init__(self, model, max_depth: int = 3, beam_width: int = 5):
        self.model = model
        self.max_depth = max_depth
        self.beam_width = beam_width

    def solve(self, initial_prompt: str) -> TreeNode:
        root = TreeNode(prompt=initial_prompt)
        frontier = [root]
        for depth in range(self.max_depth):
            candidates = []
            for node in frontier:
                # expand
                for _ in range(self.beam_width):
                    gen = self.model.generate(node.prompt)
                    score = self.model.evaluate(gen)
                    child = TreeNode(prompt=node.prompt + "\nThought: " + gen, text=gen, score=score)
                    candidates.append(child)
            # select top-K
            candidates.sort(key=lambda x: x.score, reverse=True)
            frontier = candidates[:self.beam_width]
        # return best leaf
        return max(frontier, key=lambda x: x.score)
