from .baseline import Agent
from ..env import env_2048
from ..utils import ProgressBar

from typing import Tuple, List
import numpy as np
from copy import deepcopy

class RandomBottomAgent(Agent):

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def sample_best_action(self, env: env_2048, first: bool) -> int:
        if first:
            return 3

        max_tile = 0
        best_action = -1

        for action in [2, 3, 0]:
            if not env.is_action_valid(action):
                continue

            temp_env = deepcopy(env)
            temp_grid = env.grid.copy()
            temp_env.step(action)
            new_max_tile = np.max(temp_env.grid)

            if new_max_tile > max_tile:
                max_tile = new_max_tile
                best_action = action

            env.grid = temp_grid

        if best_action == -1:
            action = np.random.choice([0, 2, 3])
            action_to_test = [0, 2, 3]
            action_to_test.remove(action)

            while not env.is_action_valid(action) and len(action_to_test) > 0:
                action = np.random.choice(action_to_test)
                action_to_test.remove(action)
            
            if len(action_to_test) == 0:
                return 1

            return action

        return best_action

    def exploit(self, env: env_2048) -> Tuple[bool, int]:
        return True, self.sample_best_action(env)
    
    def explore(self, env: env_2048) -> int:
        return self.sample_best_action(env)
    
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
    
    def play_game(self, env: env_2048) -> Tuple[List[np.ndarray], List[int]]:
        env.reset()

        grids = [env.grid]
        actions = []

        while not env.done: 
            action = self.sample_best_action(env, len(actions) == 0)
            _, grid, _, _, _ =  env.step(action)

            grids.append(grid)
            actions.append(action)

        return grids, actions 
    
    def play_games(self, env: env_2048, num_game=100000) -> List[Tuple[List[np.ndarray], List[int]]]:
        res = []
        bar = ProgressBar(target=num_game)
        max_tile = 2
        max_len = 0

        for i in range(num_game):

            grids, actions = self.play_game(env)
            res.append((grids, actions))
            max_tile = max(max_tile, np.max(grids[-1]))
            max_len = max(max_len, len(actions))
            bar.update({'max tile reached': max_tile, 'longest game': max_len}, step=i)
        
        return res
