from ..env import env_2048

from typing import Tuple
import numpy as np

class Agent:

    def __init__(self, name: str) -> None:
        """Init."""
        self.name = name

    def exploit(self, env: env_2048) -> Tuple[bool, int]:
        """Define how the agent is supposed to exploit the environment to choose an action, apply its method and return the choosen action and whether it's performable in the environment or not."""
        raise NotImplementedError

    def explore(self, env: env_2048) -> int:
        """Define how the agent will explore the environment."""
        raise NotImplementedError

    def learn(self, grid: np.ndarray, next_grid: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Define how the agent will process, after have executed its action, the environment informations to learn and optimize its further decisions."""
        raise NotImplementedError
    
    def get_descritpion(self) -> str:
        """Return the model parameters."""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Return the agent name."""
        return self.name
    
    def save(self, directory_path: str) -> None:
        """Save the agent as a pickle file with the followings structure : (params, model)."""
        raise NotImplementedError
    
    def loss(self) -> None:
        """Return the model loss."""
        raise NotImplementedError
    
    def new_episode(self) -> None:
        """Reset some parameters for the new epoch."""
        raise NotImplementedError