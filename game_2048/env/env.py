import gym
import numpy as np

from gym import spaces
from typing import Tuple

A = np.array([
    [7, 6, 5, 4],
    [6, 5, 4, 3],
    [5, 4, 3, 2],
    [4, 3, 2, 1]
])

class env_2048(gym.Env):

    def __init__(self, winning_reward=5000.0, losing_reward=-5000.0, use_matrix_reward=True) -> None:
        super().__init__()

        self.winning_reward = winning_reward
        self.losing_reward = losing_reward
        self.reward_range = (-np.infty, np.infty)
        self.player_action_space = spaces.Discrete(4) # 0: left, 1: up, 2: right, 3: down
        self.reset()
        self.use_matrix_reward = use_matrix_reward

    def reset(self) -> None:
        self.grid = np.zeros((4, 4), dtype=int)
        coords1 = np.random.choice(np.arange(4), replace=True, size=2)
        coords2 = np.random.choice(np.arange(4), replace=True, size=2)
        while coords1[0] == coords2[0] and coords1[1] == coords2[1]:
            coords2 = np.random.choice(np.arange(4), replace=True, size=2)

        self.grid[coords1[0], coords1[1]] = 2
        self.grid[coords2[0], coords2[1]] = 2
        self.score = 0
        self.done = False

    def get_empty_tile(self) -> np.ndarray:
        return np.argwhere(self.grid == 0)
    
    def has_moves_left(self, indices_zeros: np.ndarray) -> bool:
        if indices_zeros.shape[0] > 0:
            return True

        for i in range(4):
            if self.is_action_valid(action=i):
                return True

        return False

    def introduce_new_number(self) -> None:
        indices_zeros = np.argwhere(self.grid == 0)
        coord = np.random.randint(indices_zeros.shape[0])
        number = np.random.choice([2, 4], p=[0.8, 0.2])
        self.grid[indices_zeros[coord][0], indices_zeros[coord][1]] = number

    def on_left_action(self) -> float:
        for i in range(4):
            row = np.array([x for x in self.grid[i, :] if x != 0])
            for j in range(row.shape[0]):
                if j < row.shape[0] - 1 and row[j] == row[j + 1]:
                    row[j] *= 2
                    self.score += row[j]
                    row[j + 1: ] = np.pad(row[j + 2: ], pad_width=(0, 1), mode='constant', constant_values=(0, 0))

            self.grid[i, :] = np.pad(row, pad_width=(0, 4 - row.shape[0]), mode='constant', constant_values=(0, 0))

    def on_right_action(self) -> float:
        for i in range(4):
            row = np.array([x for x in self.grid[i, :] if x != 0])
            for j in reversed(range(row.shape[0])):
                if j > 0 and row[j] == row[j - 1]:
                    row[j] *= 2
                    self.score += row[j]
                    row[:j] = np.pad(row[:j - 1], pad_width=(1, 0), mode='constant', constant_values=(0, 0))

            self.grid[i, :] = np.pad(row, pad_width=(4 - row.shape[0], 0), mode='constant', constant_values=(0, 0))

    def on_up_action(self) -> float:
        for j in range(4):
            column = np.array([x for x in self.grid[:, j] if x != 0])
            for i in range(column.shape[0]):
                if i < column.shape[0] - 1 and column[i] == column[i + 1]:
                    column[i] *= 2
                    self.score += column[i]
                    column[i + 1: ] = np.pad(column[i + 2: ], pad_width=(0, 1), mode='constant', constant_values=(0, 0))

            self.grid[:, j] = np.pad(column, pad_width=(0, 4 - column.shape[0]), mode='constant', constant_values=(0, 0))

    def on_down_action(self) -> float:
        for j in range(4):
            column = np.array([x for x in self.grid[:, j] if x != 0])
            for i in reversed(range(column.shape[0])):
                if i > 0 and column[i] == column[i - 1]:
                    column[i] *= 2
                    self.score += column[i]
                    column[:i] = np.pad(column[:i - 1], pad_width=(1, 0), mode='constant', constant_values=(0, 0))

            self.grid[:, j] = np.pad(column, pad_width=(4 - column.shape[0], 0), mode='constant', constant_values=(0, 0))

    def compute_reward(self) -> float:
        reward = 0.0

        for i in range(4):
            for j in range(4):
                weight = A[i, j] if self.use_matrix_reward else 1
                reward += self.grid[i, j] * weight

        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, float, dict]:
        temp_grid = self.grid.copy()
        if action == 0:
            reward = self.on_left_action()
        
        if action == 1:
            reward = self.on_up_action()
        
        if action == 2:
            reward = self.on_right_action()
        
        if action == 3:
            reward = self.on_down_action()

        self.introduce_new_number()        
        indices_zeros = self.get_empty_tile()
        self.done = not self.has_moves_left(indices_zeros)
        reward = self.compute_reward()

        if self.done:
            reward += self.winning_reward if np.max(self.grid) >= 2048 else self.losing_reward

        return temp_grid, self.grid, self.done, reward, {}
    
    def is_action_valid(self, action: int) -> bool:
        temp_grid = self.grid.copy()

        if action == 0:
            self.on_left_action()
        elif action == 1:
            self.on_up_action()
        elif action == 2:
            self.on_right_action()
        elif action == 3:
            self.on_down_action()

        is_valid = not np.array_equal(self.grid, temp_grid)
        self.grid = temp_grid

        return is_valid

    def sample_valid_action(self) -> int:
        action = np.random.randint(4)
        action_to_test = [0, 1, 2, 3]
        action_to_test.remove(action)

        while not self.is_action_valid(action):
            action = np.random.choice(action_to_test)
            action_to_test.remove(action)

        return action