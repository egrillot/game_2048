import math

def exponential_espilon_decrease(epsilon_min: float, exponential_decay: float):
    """Return a function to decrease the epsilon during a training with the epsilon greedy search mehtod."""
    def f(epsilon: float, step_count: int) -> float:
        """Return the updated epsilon."""
        return (epsilon - epsilon_min) * math.exp((-1) * step_count / exponential_decay) + epsilon_min
    
    return f