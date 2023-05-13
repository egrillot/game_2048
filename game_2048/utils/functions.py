import math
import numpy as np
from typing import List, Dict

def exponential_espilon_decrease(epsilon_min: float, exponential_decay: float):
    """Return a function to decrease the epsilon during a training with the epsilon greedy search mehtod."""
    def f(epsilon: float, step_count: int) -> float:
        """Return the updated epsilon."""
        return (epsilon - epsilon_min) * math.exp((-1) * step_count / exponential_decay) + epsilon_min
    
    return f

def vectorize_action(action: int) -> np.ndarray:
    vect = np.zeros(4)
    vect[action] = 1.0

    return vect

def vectorize_w(successfull_game: bool, move_count: int) -> List[float]:
    w = 1.0 if successfull_game else -1.0

    return [w / (move_count - i) for i in range(move_count)]

def build_training_overview(overview: Dict[int, int]) -> str:
    message = 'Current Overview:\n\nmax tile reached | game count\n-----------------------------\n'

    for tile in range(1, 12):
        raw_tile = 2 ** tile
        count = overview[raw_tile] if raw_tile in overview else 0
        length_tile = len(str(raw_tile))
        length_count = len(str(count))
        raw1 = ' ' * (8 - length_tile // 2) + str(raw_tile)

        if length_tile % 2 == 0:
            raw1 += ' ' * (9 - length_tile // 2) 
        else:
            raw1 += ' ' * (8 - length_tile // 2) 

        raw2 = ' ' * (5 - length_count // 2) + str(count)
        message += f'{raw1}|{raw2}\n'

    return message
