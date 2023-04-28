from __future__ import annotations
import numpy as np
import random 

from typing import List, Union, Tuple

class State:

    def __init__(self, grid: np.ndarray, player=1, last_move=None) -> None:
        self.grid = grid
        self.player = player
        self.last_move = last_move

    def on_left_action(self, score: int=None) -> None:
        for i in range(4):
            row = np.array([x for x in self.grid[i, :] if x != 0])
            for j in range(row.shape[0]):
                if j < row.shape[0] - 1 and row[j] == row[j + 1]:
                    row[j] *= 2
                    if score:
                        score += row[j]
                    row[j + 1: ] = np.pad(row[j + 2: ], pad_width=(0, 1), mode='constant', constant_values=(0, 0))

            self.grid[i, :] = np.pad(row, pad_width=(0, 4 - row.shape[0]), mode='constant', constant_values=(0, 0))
        
        return score

    def on_right_action(self, score: int=None) -> None:
        for i in range(4):
            row = np.array([x for x in self.grid[i, :] if x != 0])
            for j in reversed(range(row.shape[0])):
                if j > 0 and row[j] == row[j - 1]:
                    row[j] *= 2
                    if score:
                        score += row[j]
                    row[:j] = np.pad(row[:j - 1], pad_width=(1, 0), mode='constant', constant_values=(0, 0))

            self.grid[i, :] = np.pad(row, pad_width=(4 - row.shape[0], 0), mode='constant', constant_values=(0, 0))

    def on_up_action(self, score: int=None) -> None:
        for j in range(4):
            column = np.array([x for x in self.grid[:, j] if x != 0])
            for i in range(column.shape[0]):
                if i < column.shape[0] - 1 and column[i] == column[i + 1]:
                    column[i] *= 2
                    if score:
                        score += column[i]
                    column[i + 1: ] = np.pad(column[i + 2: ], pad_width=(0, 1), mode='constant', constant_values=(0, 0))

            self.grid[:, j] = np.pad(column, pad_width=(0, 4 - column.shape[0]), mode='constant', constant_values=(0, 0))

    def on_down_action(self, score: int=None) -> None:
        for j in range(4):
            column = np.array([x for x in self.grid[:, j] if x != 0])
            for i in reversed(range(column.shape[0])):
                if i > 0 and column[i] == column[i - 1]:
                    column[i] *= 2
                    if score:
                        score += column[i]
                    column[:i] = np.pad(column[:i - 1], pad_width=(1, 0), mode='constant', constant_values=(0, 0))

            self.grid[:, j] = np.pad(column, pad_width=(4 - column.shape[0], 0), mode='constant', constant_values=(0, 0))

    def is_terminal(self) -> bool:
        temp_grid = self.grid.copy()
        self.on_left_action()
        is_terminal_left = np.array_equal(self.grid, temp_grid)
        self.grid = temp_grid

        temp_grid = self.grid.copy()
        self.on_up_action()
        is_terminal_up = np.array_equal(self.grid, temp_grid)
        self.grid = temp_grid
        
        temp_grid = self.grid.copy()
        self.on_right_action()
        is_terminal_right = np.array_equal(self.grid, temp_grid)
        self.grid = temp_grid
        
        temp_grid = self.grid.copy()
        self.on_down_action()
        is_terminal_down = np.array_equal(self.grid, temp_grid)
        self.grid = temp_grid

        return is_terminal_left and is_terminal_up and is_terminal_right and is_terminal_down
    
    def clone(self) -> State:
        return State(self.grid.copy())
    
    def play_move(self, move: Union[int, Tuple[int, int, int]]=None, score: int=None) -> Tuple[State, Union[int, None]]:
        if self.player == 1:
            temp_grid = self.grid.copy()
            if move == 0:
                self.on_left_action(score)
            elif move == 1:
                self.on_up_action(score)
            elif move == 2:
                self.on_right_action(score)
            elif move == 3:
                self.on_down_action(score)

            new_state = self.clone()
            self.grid = temp_grid
            new_state.player = -1
            new_state.last_move = move

        else:
            new_state = self.clone()

            if not move:
                i, j = random.choice([(i, j) for i in range(4) for j in range(4) if self.grid[i, j] == 0])
                new_tile = np.random.choice([2, 4], p=[0.9, 0.1])

            else:
                i, j, new_tile = move

            new_state.grid[i, j] = new_tile
            new_state.player = 1
        
        return new_state, score
    
    def get_legal_moves(self) -> List[int]:
        legal_moves = []

        temp_grid = self.grid.copy()
        self.on_left_action()
        if not np.array_equal(self.grid, temp_grid):
            legal_moves.append(0)
        self.grid = temp_grid

        temp_grid = self.grid.copy()
        self.on_up_action()
        if not np.array_equal(self.grid, temp_grid):
            legal_moves.append(1)
        self.grid = temp_grid
        
        temp_grid = self.grid.copy()
        self.on_right_action()
        if not np.array_equal(self.grid, temp_grid):
            legal_moves.append(2)
        self.grid = temp_grid
        
        temp_grid = self.grid.copy()
        self.on_down_action()
        if not np.array_equal(self.grid, temp_grid):
            legal_moves.append(3)
        self.grid = temp_grid

        return legal_moves
    
    def max_reached(self) -> int:
        return np.max(self.grid)