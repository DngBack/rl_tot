from typing import List, Dict, Any, Optional
import numpy as np


class ThoughtNode:
    def __init__(self, thought: str, reward: float = 0.0):
        self.thought = thought
        self.reward = reward
        self.children: List[ThoughtNode] = []


class ThoughtTree:
    def __init__(self, max_depth: int, num_thoughts: int):
        self.max_depth = max_depth
        self.num_thoughts = num_thoughts
        self.root: Optional[ThoughtNode] = None
        self.current_depth = 0

    def initialize(self, prompt: str):
        """Initialize tree with root node."""
        self.root = ThoughtNode(prompt)
        self.current_depth = 0

    def get_current_thoughts(self) -> List[str]:
        """Get thoughts at current depth."""
        if not self.root:
            return []

        def get_thoughts_at_depth(node: ThoughtNode, depth: int) -> List[str]:
            if depth == 0:
                return [node.thought]
            thoughts = []
            for child in node.children:
                thoughts.extend(get_thoughts_at_depth(child, depth - 1))
            return thoughts

        return get_thoughts_at_depth(self.root, self.current_depth)

    def update(self, thoughts: List[str], rewards: List[float]):
        """Update tree with new thoughts and rewards."""
        if not self.root:
            return

        def update_at_depth(node: ThoughtNode, depth: int):
            if depth == 0:
                for thought, reward in zip(thoughts, rewards):
                    node.children.append(ThoughtNode(thought, reward))
                return

            for child in node.children:
                update_at_depth(child, depth - 1)

        update_at_depth(self.root, self.current_depth)
        self.current_depth += 1

    def select_best_thoughts(self) -> List[str]:
        """Select best thoughts based on rewards."""
        if not self.root:
            return []

        def get_best_thoughts(node: ThoughtNode, depth: int) -> List[str]:
            if depth == 0:
                return [node.thought]

            if not node.children:
                return []

            # Sort children by reward
            sorted_children = sorted(
                node.children, key=lambda x: x.reward, reverse=True
            )
            best_children = sorted_children[: self.num_thoughts]

            thoughts = []
            for child in best_children:
                thoughts.extend(get_best_thoughts(child, depth - 1))
            return thoughts

        return get_best_thoughts(self.root, self.current_depth)

    def get_best_thought(self) -> str:
        """Get the single best thought from the tree."""
        if not self.root:
            return ""

        def get_best_thought_recursive(node: ThoughtNode) -> str:
            if not node.children:
                return node.thought

            best_child = max(node.children, key=lambda x: x.reward)
            return get_best_thought_recursive(best_child)

        return get_best_thought_recursive(self.root)
