import numpy as np
from typing import Tuple, Union, List

from .State import State

class MCTSAgent:

    def __init__(self) -> None:
        pass

    def predict(self, state: State) -> Tuple[np.ndarray, float]:   
        raise NotImplementedError

    def train(self, states: Union[List[State], List[np.ndarray]], policies: List[np.ndarray], values: List[float], batch_size=64, epochs=1) -> float:
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        raise NotImplementedError
