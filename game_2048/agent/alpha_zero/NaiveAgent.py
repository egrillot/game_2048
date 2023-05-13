from typing import List, Tuple, Union
import numpy as np

from game_2048.agent.alpha_zero.State import State
from .State import State
from .MCTSAgent import MCTSAgent
from ...env import env_2048
from copy import deepcopy

class NaiveAgent(MCTSAgent):

    def __init__(self) -> None:
        super().__init__()

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        return np.ones(4) / 4, 0.0
    
    def train(self, states: List[State] | List[np.ndarray], policies: List[np.ndarray], values: List[float], batch_size=64, epochs=1) -> float:
        return 
    
    def save(self, path: str) -> None:
        return 
