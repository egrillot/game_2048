from __future__ import annotations
import numpy as np
import math
import random
from typing import Optional, List

from .State import State
from .MCTSAgent import MCTSAgent

class Node:

    def __init__(self, state: State, parent: Optional[Node]=None, policy=None) -> None:
        self.state = state 
        self.parent = parent
        self.children: List[Node] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.value = 0.0
        self.policy = policy
        self.depth = parent.depth + 1 if parent else 1

    def uct(self, c_puct) -> float:
        u = self.policy * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + c_puct * u
    
    def __repr__(self) -> str:
        return f"{self.state.grid}"
    
    def signature(self) -> int:
        return hash(self.state)
    
class MCTS:

    def __init__(self, model: MCTSAgent, num_simulations: int, c_upct=1.0) -> None:
        self.model = model
        self.num_simulations = num_simulations
        self.c_upct = c_upct
        self.max_depth = 1

    def selection(self, node: Node) -> Node:
        while len(node.children) > 0:
            if node.state.player == -1:
                node = random.choice(node.children)
            else:
                upcts = np.array([child.uct(self.c_upct) for child in node.children])
                coords = np.argwhere(upcts == np.max(upcts))
                node = node.children[coords[np.random.randint(coords.shape[0])][0]]
        
        return node
    
    def expansion(self, node: Node) -> float:
        state = node.state 

        if not state.is_terminal():
            policy, value = self.model.predict(state)

            if state.player == 1:
                legal_moves = state.get_legal_moves()

                for move in legal_moves:
                    child_state, _ = state.play_move(move)
                    child = Node(child_state, parent=node, policy=policy[move])
                    node.children.append(child)
        
            else:
                count = 2 * sum([1 for i in range(4) for j in range(4) if state.grid[i, j] == 0])
                for i in range(4):
                    for j in range(4):
                        if state.grid[i, j] == 0:
                            for number in [2, 4]:
                                child_state, _ = state.play_move((i, j, number))
                                child = Node(child_state, parent=node, policy=1/count)
                                node.children.append(child)

            return value
        
        else:
            return 1.0 if state.max_reached() >= 2048 else - 1.0

        
    def backpropagate(self, value: float, node: Node) -> Node:
        while node.parent:
            node.visit_count += 1
            if node.parent:
                node.total_value += value
                node.value = node.total_value / node.visit_count
                node = node.parent
        
        return node

    def run(self, node: Node) -> int:
        max_tile = 2

        for _ in range(self.num_simulations):
            leaf = self.selection(node)
            value = self.expansion(leaf)
            max_tile = max(max_tile, np.max(leaf.state.grid))
            self.max_depth = max(leaf.depth + 1, self.max_depth)
            node = self.backpropagate(value, leaf)
        
        return max_tile

    def get_depth(self) -> int:
        return self.max_depth