from typing import Any, Dict, List, Optional, Callable
import uuid


class ThoughtNode:
    """
    Represents a node in the Tree-of-Thought search.
    """

    def __init__(
        self,
        state: Any,
        parent: Optional["ThoughtNode"] = None,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id: str = str(uuid.uuid4())
        self.state: Any = state
        self.parent: Optional["ThoughtNode"] = parent
        self.children: List["ThoughtNode"] = []
        self.score: float = score
        # Metadata holds things like entropy, visit counts, etc.
        self.metadata: Dict[str, Any] = metadata or {
            "visits": 0,
            "entropy": None,
            "reward": 0.0,
        }

    def add_child(self, child_node: "ThoughtNode") -> None:
        self.children.append(child_node)

    def visit(self) -> None:
        self.metadata["visits"] = self.metadata.get("visits", 0) + 1

    def set_entropy(self, entropy: float) -> None:
        self.metadata["entropy"] = entropy

    def set_reward(self, reward: float) -> None:
        self.metadata["reward"] = reward

    def __repr__(self):
        return f"<ThoughtNode id={self.id[:8]} score={self.score:.4f} visits={self.metadata.get('visits', 0)} reward={self.metadata.get('reward', 0.0):.2f}>"


class TreeManager:
    """
    Core inference engine for Tree-of-Thought search.
    Provides interfaces for expansion, scoring, and backtracking.
    """

    def __init__(
        self,
        expand_fn: Callable[[ThoughtNode], List[Any]],
        score_fn: Callable[[Any], float],
        max_depth: int = 10,
    ):
        # expand_fn generates next possible states from a node
        self.expand_fn = expand_fn
        # score_fn assigns a score to a candidate state
        self.score_fn = score_fn
        self.max_depth = max_depth
        self.root: Optional[ThoughtNode] = None

    def initialize(self, initial_state: Any) -> None:
        """
        Initialize the tree with the given starting state.
        """
        self.root = ThoughtNode(state=initial_state)

    def generate_and_expand(self, node: ThoughtNode) -> List[ThoughtNode]:
        """
        Generate candidate states from `node`, wrap them as ThoughtNodes, score, and attach to node.
        """
        candidates = self.expand_fn(node)
        child_nodes: List[ThoughtNode] = []
        for state in candidates:
            score = self.score_fn(state)
            child = ThoughtNode(state=state, parent=node, score=score)
            node.add_child(child)
            child_nodes.append(child)
        return child_nodes

    def backtrack(self, node: ThoughtNode) -> ThoughtNode:
        """
        Simple backtracking to the parent.
        """
        if node.parent is None:
            return node
        return node.parent

    def traverse(self, strategy: str = "best_first") -> ThoughtNode:
        """
        Example traversal loop to depth `max_depth`, returns the final node.
        """
        if self.root is None:
            raise ValueError("TreeManager not initialized with a root node.")
        current = self.root
        depth = 0
        while depth < self.max_depth:
            current.visit()
            children = self.generate_and_expand(current)
            if not children:
                break
            if strategy == "best_first":
                current = max(children, key=lambda n: n.score)
            elif strategy == "dfs":
                current = children[0]
            else:
                import random

                current = random.choice(children)
            depth += 1
        return current
