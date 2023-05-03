from ..env import env_2048
from ..utils import ProgressBar

from typing import Tuple, List
import numpy as np
from copy import deepcopy

class MCTS:

    def __init__(self, num_sample=100, depth=50) -> None:
        self.num_sample = num_sample
        self.depth = depth 

    def chose_move(self, env: env_2048) -> int:
        scores = np.zeros(4)

        for first_move in range(4):

            if env.is_action_valid(first_move):
                temp_env = deepcopy(env)
                _, _, done, reward, _ = temp_env.step(first_move)
                scores[first_move] += reward

                if not done:

                    for _ in range(self.num_sample):
                        move_number = 1
                        search_env = deepcopy(temp_env)
                        
                        while not search_env.done and move_number < self.depth:

                            action = search_env.sample_valid_action()
                            _, _, _, reward, _ = search_env.step(action)
                            scores[first_move] += reward

        return np.argmax(scores)

    def play_game(self, game: int) -> Tuple[List[np.ndarray], List[int]]:
        actions = []
        grids = []
        env = env_2048()
        env.reset()
        max_tile = 2
        step = 1

        pbar = ProgressBar(target=2000)

        while env.has_moves_left():

            action = self.chose_move(env)
            _, grid, _, _, _ = env.step(action)

            max_tile = max(max_tile, np.max(grid))

            actions.append(action)
            grids.append(grid)

            pbar.update({'max tile reached': max_tile, 'move count': step}, step)
            step += 1

        pbar.update({'max tile reached': max_tile, 'move count': step}, step)
        print(f"\nGame {game} ended at {step} iterations - max tile reached : {max_tile}\n\n###################\n")

        return grids, actions

    def generate_dataset(self, game_number=50) -> List[Tuple[List[np.ndarray], List[int]]]:
        res = []

        for i in range(game_number):

            grids, actions = self.play_game(game=i)
            res.append((grids, actions))

        return res