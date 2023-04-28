from .baseline import Agent
from ..env import env_2048

from typing import Tuple
import numpy as np

class RandomAgent(Agent):

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def exploit(self, env: env_2048) -> Tuple[bool, int]:
        return True, env.sample_valid_action()
    
    def explore(self, env: env_2048) -> int:
        return env.sample_valid_action()
    
    def learn(self, grid: np.ndarray, next_grid: np.ndarray, action: int, reward: float, done: bool) -> None:
        return 
    
    def get_descritpion(self) -> str:
        return 'Uniform'

    def loss(self) -> None:
        return

    def new_episode(self) -> None:
        return

    def save(self, directory_path: str) -> None:
        return