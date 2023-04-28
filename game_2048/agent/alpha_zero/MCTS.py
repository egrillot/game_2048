from __future__ import annotations
import numpy as np
import math
import random
from typing import Optional, List

from .NeuralNet import NeuralNet
from .State import State

A = np.array([
    [7, 6, 5, 4], 
    [6, 5, 4, 3],
    [5, 4, 3, 2],
    [4, 3, 2, 1]
])

class Node:

    def __init__(self, state: State, parent: Optional[Node]=None, policy=None, warm_up=False) -> None:
        self.state = state 
        self.parent = parent
        self.children: List[Node] = []
        self.visit_count = 1 if warm_up else 0
        self.total_value = 0.0

        if warm_up:
            for i in range(4):
                for j in range(4):
                    self.total_value += A[i, j] * state.grid[i, j]
            self.total_value /= 7 * 2048
            
        self.value = self.total_value if warm_up else 0.0
        self.warm_up = warm_up

        self.policy = policy

    def uct(self, c_puct) -> float:        
        if self.warm_up:
            return self.policy

        u = self.policy * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return (self.value + c_puct * u) * self.state.player
    
class MCTS:

    def __init__(self, neural_net: NeuralNet, num_simulations: int, c_upct=1.0, warm_up=False) -> None:
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_upct = c_upct
        self.warm_up = warm_up

    def selection(self, node: Node) -> Node:
        while len(node.children) > 0:
            upcts = np.array([child.uct(self.c_upct) for child in node.children])
            coords = np.argwhere(upcts == np.max(upcts))

            if self.warm_up:
                coord = random.choice(coords)[0]
                node = node.children[coord]

            else:
                node = node.children[coords[-1][0]]
        
        return node
    
    def expansion(self, node: Node) -> float:
        state = node.state 

        if not state.is_terminal():
            policy, value = self.neural_net.predict(state)

            if state.player == 1:
                legal_moves = state.get_legal_moves()

                for move in legal_moves:
                    child_state = state.play_move(move)
                    child = Node(child_state, parent=node, policy=policy[move], warm_up=self.warm_up)
                    node.children.append(child)
        
            else:
                count = 2 * sum([1 for i in range(4) for j in range(4) if state.grid[i, j] == 0])
                for i in range(4):
                    for j in range(4):
                        if state.grid[i, j] == 0:
                            for number in [2, 4]:
                                child_state = state.play_move((i, j, number))
                                child = Node(child_state, parent=node, policy=1/count, warm_up=self.warm_up)
                                node.children.append(child)

            return value
        
        else:
            return 1.0 if state.max_reached() >= 2048 else - 1.0

        
    def backpropagate(self, value: float, leaf: Node) -> Node:
        current = leaf
        while current.parent:
            current.visit_count += 1
            if current.parent:
                current.total_value += value
                current.value = current.total_value / current.visit_count
                current = current.parent
        
        return current

    def run(self, node: Node) -> int:
        max_tile = 2

        for _ in range(self.num_simulations):
            leaf = self.selection(node)
            value = self.expansion(leaf)
            max_tile = max(max_tile, np.max(leaf.state.grid))
            node = self.backpropagate(value, leaf)
        
        return max_tile
