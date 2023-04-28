import numpy as np


class Wrapper:

    def __init__(self) -> None:
        pass

    def extract_features(self, grid: np.ndarray) -> np.ndarray:
        x_vert = ((grid[:2, :] << 8) + (grid[1:3, :] << 4) + grid[2:, :]).ravel()
        x_hor = ((grid[:, :2] << 8) + (grid[:, 1:3] << 4) + grid[:, 2:]).ravel()
        x_ex_00 = ((grid[1:, :3] << 8) + (grid[1:, 1:] << 4) + grid[:3, 1:]).ravel()
        x_ex_01 = ((grid[:3, :3] << 8) + (grid[1:, :3] << 4) + grid[1:, 1:]).ravel()
        x_ex_10 = ((grid[:3, :3] << 8) + (grid[:3, 1:] << 4) + grid[1:, 1:]).ravel()
        x_ex_11 = ((grid[:3, :3] << 8) + (grid[1:, :3] << 4) + grid[:3, 1:]).ravel()

        return np.concatenate([x_vert, x_hor, x_ex_00, x_ex_01, x_ex_10, x_ex_11])

class LinearRegressionAgent:

    def __init__(self) -> None:
        self.weights = np.random.randn((54, 16 ** 3)) * 0.01
        self.wrapper = Wrapper()

    def update_weights(self, grid: np.ndarray, dw: np.ndarray) -> None:
        for _ in range(4):
            for i, f in enumerate(self.wrapper.extract_features(grid)):
                self.weights[i][f] += dw
            grid = np.transpose(grid)
            for i, f in enumerate(self.wrapper.extract_features(grid)):
                self.weights[i][f] += dw
            grid = np.rot90(np.transpose(grid))

    def save(self, path: str) -> None:
        np.save(f"{path}/LinearRegressor.npy")
